/*
 * FILE:    main.c
 * AUTHORS: Colin Perkins    <csp@csperkins.org>
 *          Ladan Gharai     <ladan@isi.edu>
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <pthread.h>
#include "debug.h"
#include "host.h"
#include "perf.h"
#include "rtp/decoders.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "video_codec.h"
#include "video_capture.h"
#include "video_display.h"
#include "video_display/sdl.h"
#include "video_compress.h"
#include "video_decompress.h"
#include "pdb.h"
#include "tv.h"
#include "transmit.h"
#include "tfrc.h"
#include "ihdtv/ihdtv.h"
#include "lib_common.h"
#include "compat/platform_semaphore.h"
#include "audio/audio.h"

#if defined DEBUG && defined HAVE_LINUX
#include <mcheck.h>
#endif

#ifdef HAVE_GCOLL
#include "video_display/gcoll.h"
#endif // HAVE_GCOLL

#define EXIT_FAIL_USAGE		1
#define EXIT_FAIL_UI   		2
#define EXIT_FAIL_DISPLAY	3
#define EXIT_FAIL_CAPTURE	4
#define EXIT_FAIL_NETWORK	5
#define EXIT_FAIL_TRANSMIT	6
#define EXIT_FAIL_COMPRESS	7
#define EXIT_FAIL_DECODER	8

#define PORT_BASE               5004
#define PORT_AUDIO              5006

/* please see comments before transmit.c:audio_tx_send() */
/* also note that this actually differs from video */
#define DEFAULT_AUDIO_FEC       "mult:3"

#define AUDIO_CHANNEL_MAP (('a' << 8) | 'm')
#define AUDIO_CAPTURE_CHANNELS (('a' << 8) | 'c')
#define AUDIO_SCALE (('a' << 8) | 's')
#define ECHO_CANCELLATION (('E' << 8) | 'C')
#define CUDA_DEVICE (('C' << 8) | 'D')
#define MCAST_IF (('M' << 8) | 'I')

#ifdef HAVE_MACOSX
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  5944320
#else
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  ((4*1920*1080)*110/100)
#endif

struct state_send {
        char *requested_capture;
        char *capture_cfg;
        struct vidcap *capture_device;
        struct rtp *network_device;
        struct tx *tx;

        volatile unsigned int has_item_to_send:1;
        volatile unsigned int sender_waiting:1;
        volatile unsigned int compress_thread_waiting:1;
        volatile unsigned int should_exit_sender:1;
        pthread_mutex_t sender_lock;
        pthread_cond_t compress_thread_cv;
        pthread_cond_t sender_cv;

        struct video_frame * volatile tx_frame;
};

struct state_uv {
        char *address;
        bool use_ipv6;
        char *mcast_if;
        int recv_port_number;
        int send_port_number;
        struct rtp *recv_network_device;
        unsigned int connections_count;
        int send_idx;
        struct state_send send[CAP_DEV_COUNT];
        struct tx *tx;

        struct timeval start_time, curr_time;
        struct pdb *participants;
        
        char *decoder_mode;
        char *postprocess;
        
        uint32_t ts;
        struct display *display_device;
        char *requested_compression;
        const char *requested_display;
        unsigned requested_mtu;
        
        struct state_audio *audio;

        /* used mainly to serialize initialization */
        pthread_mutex_t master_lock;
};

long packet_rate;
volatile int wait_to_finish = FALSE;
volatile int threads_joined = FALSE;
static int exit_status = EXIT_SUCCESS;
static bool should_exit_receiver = false;
static bool should_exit_sender = false;

unsigned int cuda_device = 0;
unsigned int audio_capture_channels = 2;

uint32_t RTT = 0;               /* this is computed by handle_rr in rtp_callback */
uint32_t hd_color_spc = 0;

long frame_begin[2];

int uv_argc;
char **uv_argv;
static struct state_uv *uv_state;

//
// prototypes
//
static struct rtp *initialize_network(char *addr, int recv_port,
                int send_port, struct pdb *participants, bool use_ipv6,
                char *mcast_if);

static void list_video_display_devices(void);
static void list_video_capture_devices(void);
static void sender_finish(struct state_send *state);
static void display_buf_increase_warning(int size);

#ifndef WIN32
static void signal_handler(int signal)
{
        debug_msg("Caught signal %d\n", signal);
        exit_uv(0);
        return;
}
#endif                          /* WIN32 */

static void _exit_uv(int status);

static void _exit_uv(int status) {
        exit_status = status;
        wait_to_finish = TRUE;
        if(!threads_joined) {
                should_exit_sender = true;
                should_exit_receiver = true;
                if(uv_state->audio)
                        audio_finish(uv_state->audio);
        }
        wait_to_finish = FALSE;
}

void (*exit_uv)(int status) = _exit_uv;

static void usage(void)
{
        /* TODO -c -p -b are deprecated options */
        printf("\nUsage: uv [-d <display_device>] [-t <capture_device>] [-r <audio_playout>]\n");
        printf("          [-s <audio_caputre>] [-l <limit_bitrate>] "
                        "[-m <mtu>] [-c] [-i] [-6]\n");
        printf("          [-m <video_mode>] [-p <postprocess>] "
                        "[-f <fec_options>] [-p <port>]\n");
        printf("          [--mcast-if <iface>] address(es)\n\n");
        printf
            ("\t-d <display_device>        \tselect display device, use '-d help'\n");
        printf("\t                         \tto get list of supported devices\n");
        printf("\n");
        printf
            ("\t-t <capture_device>        \tselect capture device, use '-t help'\n");
        printf("\t                         \tto get list of supported devices\n");
        printf("\n");
        printf("\t-c <cfg>                 \tcompress video (see '-c help')\n");
        printf("\n");
#ifdef HAVE_IPv6
        printf("\t-6                       \tUse IPv6\n");
        printf("\n");
#endif //  HAVE_IPv6
        printf("\t--mcast-if <iface>       \tBind to specified interface for multicast\n");
        printf("\n");
        printf("\t-r <playback_device>     \tAudio playback device (see '-r help')\n");
        printf("\n");
        printf("\t-s <capture_device>      \tAudio capture device (see '-s help')\n");
        printf("\n");
        printf("\t-j <settings>            \tJACK Audio Connection Kit settings\n");
        printf("\n");
        printf("\t-M <video_mode>          \treceived video mode (eg tiled-4K, 3D,\n");
        printf("\t                         \tdual-link)\n");
        printf("\n");
        printf("\t-p <postprocess>         \tpostprocess module\n");
        printf("\n");
        printf("\t-f <settings>            \tFEC settings - use \"mult:<nr>\",\n");
        printf("\t                         \t\"ldgm:<max_expected_loss>%%\" or\n");
        printf("\t                         \t\"ldgm:<k>:<m>:<c>\"\n");
        printf("\n");
        printf("\t-P <port> | <recv_port>:<send_port>\n");
        printf("\t                         \tbase port number, also 3 subsequent\n");
        printf("\t                         \tports can be used for RTCP and audio\n");
        printf("\t                         \tstreams. Default: %d.\n", PORT_BASE);
        printf("\t                         \tIf one given, it will be used for both\n");
        printf("\t                         \tsending and receiving, if two, first one\n");
        printf("\t                         \twill be used for receiving, second one\n");
        printf("\t                         \tfor sending.\n");
        printf("\n");
        printf("\t-l <limit_bitrate> | unlimited\tlimit sending bitrate (aggregate)\n");
        printf("\t                         \tto limit_bitrate Mb/s\n");
        printf("\n");
        printf("\t--audio-channel-map      <mapping> | help\n");
        printf("\n");
        printf("\t--audio-scale            <factor> | <method> | help\n");
        printf("\n");
        printf("\t--audio-capture-channels <count> number of input channels that will\n");
        printf("\t                                 be captured (default 2).\n");
        printf("\n");
        printf("\t--echo-cancellation      \tapply acustic echo cancellation to audio\n");
        printf("\n");
        printf("\t--cuda-device [<index>|help]\tuse specified CUDA device\n");
        printf("\n");
        printf("\n");
        printf("\t-F                       \tGColl front camera\n");
        printf("\t-S                       \tGColl side camera\n");
        printf("\t-G                       \tGColl group camera camera (if present)\n");
        printf("\t-R                       \tGColl roam ID\n");
        printf("\taddress(es)              \tdestination address\n");
        printf("\n");
        printf("\t                         \tIf comma-separated list of addresses\n");
        printf("\t                         \tis entered, video frames are split\n");
        printf("\t                         \tand chunks are sent/received\n");
        printf("\t                         \tindependently.\n");
        printf("\n");
}

static void list_video_display_devices()
{
        int i;
        display_type_t *dt;

        printf("Available display devices:\n");
        display_init_devices();
        for (i = 0; i < display_get_device_count(); i++) {
                dt = display_get_device_details(i);
                printf("\t%s\n", dt->name);
        }
        display_free_devices();
}

struct display *initialize_video_display(const char *requested_display,
                                                char *fmt, unsigned int flags, void *udata)
{
        struct display *d;
        display_type_t *dt;
        display_id_t id = 0;
        int i;
        
        if(!strcmp(requested_display, "none"))
                 id = display_get_null_device_id();

        if (display_init_devices() != 0) {
                printf("Unable to initialise devices\n");
                abort();
        } else {
                debug_msg("Found %d display devices\n",
                          display_get_device_count());
        }
        for (i = 0; i < display_get_device_count(); i++) {
                dt = display_get_device_details(i);
                if (strcmp(requested_display, dt->name) == 0) {
                        id = dt->id;
                        debug_msg("Found device\n");
                        break;
                } else {
                        debug_msg("Device %s does not match %s\n", dt->name,
                                  requested_display);
                }
        }
        if(i == display_get_device_count()) {
                fprintf(stderr, "WARNING: Selected '%s' display card "
                        "was not found.\n", requested_display);
                return NULL;
        }
        display_free_devices();

        d = display_init(id, fmt, flags, udata);
        return d;
}

static void list_video_capture_devices()
{
        int i;
        struct vidcap_type *vt;

        printf("Available capture devices:\n");
        vidcap_init_devices();
        for (i = 0; i < vidcap_get_device_count(); i++) {
                vt = vidcap_get_device_details(i);
                printf("\t%s\n", vt->name);
        }
        vidcap_free_devices();
}

struct vidcap *initialize_video_capture(const char *requested_capture,
                                               char *fmt, unsigned int flags)
{
        struct vidcap_type *vt;
        vidcap_id_t id = 0;
        int i;
        
        if(!strcmp(requested_capture, "none"))
                id = vidcap_get_null_device_id();

        vidcap_init_devices();
        for (i = 0; i < vidcap_get_device_count(); i++) {
                vt = vidcap_get_device_details(i);
                if (strcmp(vt->name, requested_capture) == 0) {
                        id = vt->id;
                        break;
                }
        }
        if(i == vidcap_get_device_count()) {
                fprintf(stderr, "WARNING: Selected '%s' capture card "
                        "was not found.\n", requested_capture);
                return NULL;
        }
        vidcap_free_devices();

        return vidcap_init(id, fmt, flags);
}

static void display_buf_increase_warning(int size)
{
        fprintf(stderr, "\n***\n"
                        "Unable to set buffer size to %d B.\n"
                        "Please set net.core.rmem_max value to %d or greater. (see also\n"
                        "https://www.sitola.cz/igrid/index.php/Setup_UltraGrid)\n"
#ifdef HAVE_MACOSX
                        "\tsysctl -w kern.ipc.maxsockbuf=%d\n"
                        "\tsysctl -w net.inet.udp.recvspace=%d\n"
#else
                        "\tsysctl -w net.core.rmem_max=%d\n"
#endif
                        "To make this persistent, add these options (key=value) to /etc/sysctl.conf\n"
                        "\n***\n\n",
                        size, size,
#ifdef HAVE_MACOSX
                        size * 4,
#endif /* HAVE_MACOSX */
                        size);

}

static struct rtp *initialize_network(char *addr, int recv_port,
                int send_port, struct pdb *participants, bool use_ipv6,
                char *mcast_if)
{
	struct rtp *device = NULL;
        double rtcp_bw = 5 * 1024 * 1024;       /* FIXME */
	int ttl = 255;

        device = rtp_init_if(addr, mcast_if, recv_port,
                        send_port, ttl, rtcp_bw, FALSE,
                        rtp_recv_callback, (void *)participants,
                        use_ipv6);
        if (device != NULL) {
                rtp_set_option(device, RTP_OPT_WEAK_VALIDATION, 
                                TRUE);
                rtp_set_sdes(device, rtp_my_ssrc(device),
                                RTCP_SDES_TOOL,
                                PACKAGE_STRING, strlen(PACKAGE_STRING));

                int size = INITIAL_VIDEO_RECV_BUFFER_SIZE;
                int ret = rtp_set_recv_buf(device, INITIAL_VIDEO_RECV_BUFFER_SIZE);
                if(!ret) {
                        display_buf_increase_warning(size);
                }

                rtp_set_send_buf(device, 1024 * 56);

                pdb_add(participants, rtp_my_ssrc(device));
        }
        
        return device;
}

static void destroy_devices(struct rtp * network_device)
{
        rtp_done(network_device);
}

static struct tx *initialize_transmit(unsigned requested_mtu, char *fec)
{
        /* Currently this is trivial. It'll get more complex once we */
        /* have multiple codecs and/or error correction.             */
        return tx_init(requested_mtu, fec);
}

static struct vcodec_state *new_decoder(struct state_uv *uv) {
        struct vcodec_state *state = malloc(sizeof(struct vcodec_state));

        if(state) {
                state->decoder = decoder_init(uv->decoder_mode, uv->postprocess);
                state->reconfigured = false;
                state->frame_buffer = NULL; // no frame until reconfiguration

                if(!state->decoder) {
                        fprintf(stderr, "Error initializing decoder (incorrect '-M' or '-p' option).\n");
                        free(state);
                        exit_uv(1);
                        return NULL;
                } else {
                        decoder_register_video_display(state->decoder, uv->display_device);
                }
        }

        return state;
}

static void destroy_decoder(struct vcodec_state *video_decoder_state) {
        if(!video_decoder_state) {
                return;
        }

        decoder_destroy(video_decoder_state->decoder);

        free(video_decoder_state);
}

static void *receiver_thread(void *arg)
{
        struct state_uv *uv = (struct state_uv *)arg;

        struct pdb_e *cp;
        struct timeval timeout;
        int fr;
        int ret;
        int last_buf_size = INITIAL_VIDEO_RECV_BUFFER_SIZE;

        sleep(3);

        uv->recv_network_device =
                initialize_network(uv->address, uv->recv_port_number,
                                uv->send_port_number, uv->participants,
                                uv->use_ipv6, uv->mcast_if);
        if (uv->recv_network_device == NULL) {
                fprintf(stderr, "Unable to open network\n");
                exit_uv(EXIT_FAIL_NETWORK);
                return NULL;
        }

        initialize_video_decompress();

        pthread_mutex_unlock(&uv->master_lock);

        fr = 1;

        while (!should_exit_receiver) {
                /* Housekeeping and RTCP... */
                gettimeofday(&uv->curr_time, NULL);
                uv->ts = tv_diff(uv->curr_time, uv->start_time) * 90000;
                rtp_update(uv->recv_network_device, uv->curr_time);
                rtp_send_ctrl(uv->recv_network_device, uv->ts, 0, uv->curr_time);

                /* Receive packets from the network... The timeout is adjusted */
                /* to match the video capture rate, so the transmitter works.  */
                if (fr) {
                        gettimeofday(&uv->curr_time, NULL);
                        fr = 0;
                }

                timeout.tv_sec = 0;
                timeout.tv_usec = 999999 / 59.94;
                ret = rtp_recv_r(uv->recv_network_device, &timeout, uv->ts);

                /*
                   if (ret == FALSE) {
                   printf("Failed to receive data\n");
                   }
                 */
                UNUSED(ret);

                /* Decode and render for each participant in the conference... */
                cp = pdb_iter_init(uv->participants);
                while (cp != NULL) {
                        if (tfrc_feedback_is_due(cp->tfrc_state, uv->curr_time)) {
                                debug_msg("tfrc rate %f\n",
                                          tfrc_feedback_txrate(cp->tfrc_state,
                                                               uv->curr_time));
                        }

                        if(cp->video_decoder_state == NULL) {
                                cp->video_decoder_state = new_decoder(uv);
                                if(cp->video_decoder_state == NULL) {
                                        fprintf(stderr, "Fatal: unable to find decoder state for "
                                                        "participant %u.\n", cp->ssrc);
                                        exit_uv(1);
                                        break;
                                }
                        }

                        /* Decode and render video... */
                        if (pbuf_decode
                            (cp->playout_buffer, uv->curr_time, decode_frame, cp->video_decoder_state)) {
                                gettimeofday(&uv->curr_time, NULL);
                                fr = 1;
                                cp->video_decoder_state->frame_buffer->ssrc = cp->ssrc;
                                display_put_frame(uv->display_device,
                                                cp->video_decoder_state->frame_buffer);
                                cp->video_decoder_state->frame_buffer = NULL;
                        }

                        if(cp->video_decoder_state->decoded % 100 == 99) {
                                int new_size = cp->video_decoder_state->max_frame_size * 110ull / 100;
                                if(new_size >= last_buf_size) {
                                        struct rtp *device = uv->recv_network_device;
                                        int ret = rtp_set_recv_buf(device, new_size);
                                        if(!ret) {
                                                display_buf_increase_warning(new_size);
                                        }
                                        debug_msg("Recv buffer adjusted to %d\n", new_size);
                                        last_buf_size = new_size;
                                }
                        }

                        if(cp->video_decoder_state->reconfigured) {
                                struct rtp *session = uv->recv_network_device;
                                rtp_flush_recv_buf(session);
                                cp->video_decoder_state->reconfigured = false;
                        }

                        pbuf_remove(cp->playout_buffer, uv->curr_time);
                        cp = pdb_iter_next(uv->participants);
                }
                pdb_iter_done(uv->participants);
        }
        
        cp = pdb_iter_init(uv->participants);
        while (cp != NULL) {
                if(cp->video_decoder_state != NULL) {
                        destroy_decoder(cp->video_decoder_state);
                }

                cp = pdb_iter_next(uv->participants);
        }
        pdb_iter_done(uv->participants);

        display_finish(uv_state->display_device);

        return 0;
}

static void sender_finish(struct state_send *send) {
        pthread_mutex_lock(&send->sender_lock);

        send->should_exit_sender = TRUE;

        if(send->sender_waiting) {
                send->has_item_to_send = TRUE;
                pthread_cond_signal(&send->sender_cv);
        }

        pthread_mutex_unlock(&send->sender_lock);

}

static void *sender_thread(void *arg) {
        struct state_send *send = (struct state_send *)arg;

        while(!send->should_exit_sender) {
                pthread_mutex_lock(&send->sender_lock);

                while(!send->has_item_to_send && !send->should_exit_sender) {
                        send->sender_waiting = TRUE;
                        pthread_cond_wait(&send->sender_cv, &send->sender_lock);
                        send->sender_waiting = FALSE;
                }
                struct video_frame *tx_frame = send->tx_frame;

                if(send->should_exit_sender) {
                        send->has_item_to_send = FALSE;
                        pthread_mutex_unlock(&send->sender_lock);
                        goto exit;
                }

                pthread_mutex_unlock(&send->sender_lock);


                tx_send(send->tx, tx_frame, 
                                send->network_device);

                pthread_mutex_lock(&send->sender_lock);

                send->has_item_to_send = FALSE;

                if(send->compress_thread_waiting) {
                        pthread_cond_signal(&send->compress_thread_cv);
                }
                pthread_mutex_unlock(&send->sender_lock);
        }

exit:
        return NULL;
}

static void *compress_thread(void *arg)
{
        struct state_uv *uv = (struct state_uv *)arg;
        int my_idx = uv->send_idx++;
        struct state_send *send = &uv->send[my_idx];

        struct video_frame *tx_frame;
        struct audio_frame *audio;
        //struct video_frame *splitted_frames = NULL;
        pthread_t sender_thread_id;
        int i = 0;

        struct compress_state *compression; 

        if ((send->capture_device =
                                initialize_video_capture(send->requested_capture,
                                        send->capture_cfg, 0)) == NULL) {
                fprintf(stderr, "Unable to open capture device: %s\n",
                                send->requested_capture);
                exit_uv(EXIT_FAIL_CAPTURE);
                pthread_mutex_unlock(&uv->master_lock);
                return NULL;
        }
        printf("Video capture initialized-%s\n", send->requested_capture);

        compression = compress_init(uv->requested_compression);
        
        pthread_mutex_unlock(&uv->master_lock);
        /* NOTE: unlock before propagating possible error */
        if(compression == NULL) {
                fprintf(stderr, "Error initializing compression.\n");
                exit_uv(0);
                goto compress_done;
        }

        if (pthread_create
            (&sender_thread_id, NULL, sender_thread,
             (void *)send) != 0) {
                perror("Unable to create sender thread!\n");
                exit_uv(EXIT_FAILURE);
                goto join_thread;
        }

        while (!should_exit_sender) {
                /* Capture and transmit video... */
                tx_frame = vidcap_grab(send->capture_device, &audio);
                if (tx_frame != NULL) {
                        if(audio) {
                                /* logic error, in GColl mode, capture should be V4L2
                                 * or testcard etc. so this should not have embedded audio.
                                 * Moreover, audio is never requeseted
                                 */
                                abort();
                                audio_sdi_send(uv->audio, audio);
                        }
                        //TODO: Unghetto this
                        tx_frame = compress_frame(compression, tx_frame, i);
                        if(!tx_frame)
                                continue;

                        i = (i + 1) % 2;

                        /* when sending uncompressed video, we simply post it for send
                         * and wait until done */
                        if(is_compress_none(compression)) {
                                pthread_mutex_lock(&send->sender_lock);

                                send->tx_frame = tx_frame;

                                send->has_item_to_send = TRUE;
                                if(send->sender_waiting) {
                                        pthread_cond_signal(&send->sender_cv);
                                }

                                if(should_exit_sender) {
                                        pthread_mutex_unlock(&send->sender_lock);
                                        goto join_thread;
                                }

                                while(send->has_item_to_send) {
                                        send->compress_thread_waiting = TRUE;
                                        pthread_cond_wait(&send->compress_thread_cv, &send->sender_lock);
                                        send->compress_thread_waiting = FALSE;
                                }
                                pthread_mutex_unlock(&send->sender_lock);
                        }  else
                        /* we post for sending (after previous frame is done) and schedule a new one
                         * frames may overlap then */
                        {
                                pthread_mutex_lock(&send->sender_lock);
                                if(should_exit_sender) {
                                        pthread_mutex_unlock(&send->sender_lock);
                                        goto join_thread;
                                }
                                while(send->has_item_to_send) {
                                        send->compress_thread_waiting = TRUE;
                                        pthread_cond_wait(&send->compress_thread_cv, &send->sender_lock);
                                        send->compress_thread_waiting = FALSE;
                                }

                                send->tx_frame = tx_frame;

                                send->has_item_to_send = TRUE;
                                if(send->sender_waiting) {
                                        pthread_cond_signal(&send->sender_cv);
                                }
                                pthread_mutex_unlock(&send->sender_lock);
                        }
                }
        }

        vidcap_finish(send->capture_device);


join_thread:
        sender_finish(send);
        pthread_join(sender_thread_id, NULL);

compress_done:
        compress_done(compression);

        if(send->capture_device)
                vidcap_done(send->capture_device);

        return NULL;
}

int main(int argc, char *argv[])
{
#if defined HAVE_SCHED_SETSCHEDULER && defined USE_RT
        struct sched_param sp;
#endif

#ifdef HAVE_GCOLL
        struct gcoll_init_params gcoll_params;
#endif // HAVE_GCOLL
        char *display_cfg = NULL;
        char *audio_recv = NULL;
        char *audio_send = NULL;
        char *jack_cfg = NULL;
        char *requested_fec = NULL;
        char *audio_channel_map = NULL;
        char *audio_scale = "mixauto";

        bool echo_cancellation = false;

        int bitrate = 0;
        
        struct state_uv *uv;
        int ch;
        
        pthread_t receiver_thread_id,
                  compress_thread_id[CAP_DEV_COUNT];
	bool receiver_thread_started = false,
		  compress_thread_started[CAP_DEV_COUNT];
        unsigned vidcap_flags = 0,
                 display_flags = 0;

#if defined DEBUG && defined HAVE_LINUX
        mtrace();
#endif

        if (argc == 1) {
                usage();
                return EXIT_FAIL_USAGE;
        }

        uv_argc = argc;
        uv_argv = argv;

        static struct option getopt_options[] = {
                {"display", required_argument, 0, 'd'},
                {"capture", required_argument, 0, 't'},
                {"mtu", required_argument, 0, 'm'},
                {"ipv6", no_argument, 0, '6'},
                {"mode", required_argument, 0, 'M'},
                {"version", no_argument, 0, 'v'},
                {"compress", required_argument, 0, 'c'},
                {"receive", required_argument, 0, 'r'},
                {"send", required_argument, 0, 's'},
                {"help", no_argument, 0, 'h'},
                {"jack", required_argument, 0, 'j'},
                {"fec", required_argument, 0, 'f'},
                {"port", required_argument, 0, 'P'},
                {"limit-bitrate", required_argument, 0, 'l'},
                {"audio-channel-map", required_argument, 0, AUDIO_CHANNEL_MAP},
                {"audio-scale", required_argument, 0, AUDIO_SCALE},
                {"audio-capture-channels", required_argument, 0, AUDIO_CAPTURE_CHANNELS},
                {"echo-cancellation", no_argument, 0, ECHO_CANCELLATION},
                {"cuda-device", required_argument, 0, CUDA_DEVICE},
                {"mcast-if", required_argument, 0, MCAST_IF},
#ifdef HAVE_GCOLL
                {"front-camera", required_argument, 0, 'F'},
                {"side-camera", required_argument, 0, 'S'},
                {"group-camera", required_argument, 0, 'G'},
                {"room", required_argument, 0, 'R'},
#endif
                {0, 0, 0, 0}
        };
        int option_index = 0;

        //      uv = (struct state_uv *) calloc(1, sizeof(struct state_uv));
        uv = (struct state_uv *)malloc(sizeof(struct state_uv));
        uv_state = uv;

        uv->address = NULL;
        uv->mcast_if = NULL;
        uv->use_ipv6 = false;

        uv->audio = NULL;
        uv->ts = 0;
        uv->display_device = NULL;
        uv->requested_display = "none";
        for(int i = 0; i < CAP_DEV_COUNT; ++i) {
                uv->send[i].capture_cfg = NULL;
                uv->send[i].requested_capture = "none";
                compress_thread_id[i] = 0;
                compress_thread_started[i] = false;
                uv->send[i].tx = NULL;
                uv->send[i].network_device = NULL;

                uv->send[i].has_item_to_send = FALSE;
                uv->send[i].sender_waiting = FALSE;
                uv->send[i].compress_thread_waiting = FALSE;
                uv->send[i].should_exit_sender = FALSE;
                pthread_mutex_init(&uv->send[i].sender_lock, NULL);
                pthread_cond_init(&uv->send[i].compress_thread_cv, NULL);
                pthread_cond_init(&uv->send[i].sender_cv, NULL);
        }
        uv->requested_compression = "none";
        uv->decoder_mode = NULL;
        uv->postprocess = NULL;
        uv->requested_mtu = 0;
        uv->participants = NULL;
        uv->recv_port_number =
                uv->send_port_number =
                PORT_BASE;
        uv->send_idx = 0;

        pthread_mutex_init(&uv->master_lock, NULL);

        perf_init();
        perf_record(UVP_INIT, 0);

        init_lib_common();

#ifdef HAVE_GCOLL
        gcoll_params.send_group_camera = false;
        gcoll_params.send_audio = false;
#endif // HAVE_GCOLL

        while ((ch =
                getopt_long(argc, argv, "d:t:m:r:s:v6c:ihj:M:p:f:P:l:F:S:G:R:", getopt_options,
                            &option_index)) != -1) {
                switch (ch) {
                case 'd':
                        if (!strcmp(optarg, "help")) {
                                list_video_display_devices();
                                return 0;
                        }
                        uv->requested_display = optarg;
                        if(strchr(optarg, ':')) {
                                char *delim = strchr(optarg, ':');
                                *delim = '\0';
                                display_cfg = delim + 1;
                        }
                        break;
#ifndef HAVE_GCOLL
                case 't':
                        if (!strcmp(optarg, "help")) {
                                list_video_capture_devices();
                                return 0;
                        }
                        uv->requested_capture = optarg;
                        if(strchr(optarg, ':')) {
                                char *delim = strchr(optarg, ':');
                                *delim = '\0';
                                capture_cfg = delim + 1;
                        }
                        break;
#endif // ! HAVE_GCOLL
                case 'm':
                        uv->requested_mtu = atoi(optarg);
                        break;
                case 'M':
                        uv->decoder_mode = optarg;
                        break;
                case 'p':
                        uv->postprocess = optarg;
                        break;
                case 'v':
                        printf("%s", PACKAGE_STRING);
#ifdef GIT_VERSION
                        printf(" (rev %s)", GIT_VERSION);
#endif
                        printf("\n");
                        printf("\n" PACKAGE_NAME " was compiled with following features:\n");
                        printf(AUTOCONF_RESULT);
                        return EXIT_SUCCESS;
                case 'c':
                        uv->requested_compression = optarg;
                        break;
                case 'r':
                        audio_recv = optarg;                       
                        break;
                case 's':
                        audio_send = optarg;
#ifdef HAVE_GCOLL
                        gcoll_params.send_audio = true;
#endif // HAVE_GCOLL
                        break;
                case 'j':
                        jack_cfg = optarg;
                        break;
                case 'f':
                        requested_fec = optarg;
                        break;
		case 'h':
			usage();
			return 0;
                case 'P':
                        if(strchr(optarg, ':')) {
                                char *save_ptr = NULL;
                                uv->recv_port_number = atoi(strtok_r(optarg, ":", &save_ptr));
                                uv->send_port_number = atoi(strtok_r(NULL, ":", &save_ptr));
                        } else {
                                uv->recv_port_number =
                                        uv->send_port_number =
                                        atoi(optarg);
                        }
                        break;
                case 'l':
                        if(strcmp(optarg, "unlimited") == 0) {
                                bitrate = -1;
                        } else {
                                bitrate = atoi(optarg);
                                if(bitrate <= 0) {
                                        usage();
                                        return EXIT_FAIL_USAGE;
                                }
                        }
                        break;
                case '6':
                        uv->use_ipv6 = true;
                        break;
                case '?':
                        break;
                case AUDIO_CHANNEL_MAP:
                        audio_channel_map = optarg;
                        break;
                case AUDIO_SCALE:
                        audio_scale = optarg;
                        break;
                case AUDIO_CAPTURE_CHANNELS:
                        audio_capture_channels = atoi(optarg);
                        break;
                case ECHO_CANCELLATION:
                        echo_cancellation = true;
                        break;
                case CUDA_DEVICE:
#ifdef HAVE_CUDA
                        if(strcmp("help", optarg) == 0) {
                                struct compress_state *compression; 
                                compression = compress_init("JPEG:list_devices");
                                compress_done(compression);
                                return EXIT_SUCCESS;
                        } else {
                                cuda_device = atoi(optarg);
                        }
                        break;
#else
                        fprintf(stderr, "CUDA support is not enabled!\n");
                        return EXIT_FAIL_USAGE;
#endif // HAVE_CUDA
                case MCAST_IF:
                        uv->mcast_if = optarg;
                        break;
#ifdef HAVE_GCOLL
                case 'F':
                        if (!strcmp(optarg, "help")) {
                                list_video_capture_devices();
                                return 0;
                        }
                        uv->send[GCOLL_FRONT].requested_capture = optarg;
                        if(strchr(optarg, ':')) {
                                char *delim = strchr(optarg, ':');
                                *delim = '\0';
                                uv->send[GCOLL_FRONT].capture_cfg = delim + 1;
                        }
                        break;
                case 'S':
                        if (!strcmp(optarg, "help")) {
                                list_video_capture_devices();
                                return 0;
                        }
                        uv->send[GCOLL_SIDE].requested_capture = optarg;
                        if(strchr(optarg, ':')) {
                                char *delim = strchr(optarg, ':');
                                *delim = '\0';
                                uv->send[GCOLL_SIDE].capture_cfg = delim + 1;
                        }
                        break;
                case 'G':
                        if (!strcmp(optarg, "help")) {
                                list_video_capture_devices();
                                return 0;
                        }
                        uv->send[GCOLL_GROUP].requested_capture = optarg;
                        if(strchr(optarg, ':')) {
                                char *delim = strchr(optarg, ':');
                                *delim = '\0';
                                uv->send[GCOLL_GROUP].capture_cfg = delim + 1;
                        }
                        gcoll_params.send_group_camera = true;
                        break;
                case 'R':
                        gcoll_params.group_id = atoi(optarg);
                        break;
#endif // HAVE_GCOLL
                default:
                        usage();
                        return EXIT_FAIL_USAGE;
                }
        }

#ifdef HAVE_GCOLL
        if(strcmp(uv->requested_display, "none") != 0 &&
                        strcmp(uv->requested_display, "gcoll") != 0) {
                fprintf(stderr, "Only allowed display in GColl mode is \"gcoll\".\n");
                exit(EXIT_FAIL_USAGE);
        }
#endif // HAVE_GCOLL
        
        argc -= optind;
        argv += optind;

        printf("%s", PACKAGE_STRING);
#ifdef GIT_VERSION
        printf(" (rev %s)", GIT_VERSION);
#endif
        printf("\n");
        printf("Display device: %s\n", uv->requested_display);
#ifndef HAVE_GCOLL
        printf("Capture device: %s\n", uv->requested_capture[0]);
#else
        printf("Front camera  : %s\n", uv->send[GCOLL_FRONT].requested_capture);
        printf("Side camera   : %s\n", uv->send[GCOLL_SIDE].requested_capture);
        printf("Group camera  : %s\n", uv->send[GCOLL_GROUP].requested_capture);
#endif // ! GCOLL
        printf("MTU           : %d\n", uv->requested_mtu);
        printf("Compression   : %s\n", uv->requested_compression);

        printf("Network protocol: ultragrid rtp\n");

        gettimeofday(&uv->start_time, NULL);

        if(uv->requested_mtu > RTP_MAX_PACKET_LEN) {
                fprintf(stderr, "Requested MTU exceeds maximal value allowed by RTP library (%d).\n",
                                RTP_MAX_PACKET_LEN);
                return EXIT_FAIL_USAGE;
        }

        if (argc == 0) {
                uv->address = strdup("localhost");
        } else {
                uv->address = (char *) argv[0];
        }

#ifdef WIN32
	WSADATA wsaData;
	int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if(err != 0) {
		fprintf(stderr, "WSAStartup failed with error %d.", err);
		return EXIT_FAILURE;
	}
	if(LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
		fprintf(stderr, "Counld not found usable version of Winsock.\n");
		WSACleanup();
		return EXIT_FAILURE;
	}
#endif
	

        char *tmp_requested_fec = strdup(DEFAULT_AUDIO_FEC);
        uv->audio = audio_cfg_init (uv->address, uv->recv_port_number + 2,
                        uv->send_port_number + 2, audio_send, audio_recv,
                        jack_cfg, tmp_requested_fec, audio_channel_map,
                        audio_scale, echo_cancellation, uv->use_ipv6, uv->mcast_if);
        free(tmp_requested_fec);
        if(!uv->audio) {
                exit_uv(1);
                goto cleanup;
        }

        vidcap_flags |= audio_get_vidcap_flags(uv->audio);
        display_flags |= audio_get_display_flags(uv->audio);

        uv->participants = pdb_init();

        {
                int rx_port = uv->recv_port_number + 4;
                for(int i = 0; i < CAP_DEV_COUNT; ++i) {
                        rx_port += 2;
                        uv->send[i].network_device =
                                initialize_network(uv->address, rx_port,
                                                uv->send_port_number, uv->participants,
                                                uv->use_ipv6, uv->mcast_if);
                        if (uv->send[i].network_device == NULL) {
                                fprintf(stderr, "Unable to open network\n");
                                exit_uv(EXIT_FAIL_NETWORK);
                                goto cleanup_wait_audio;
                        }
                }

                if (uv->requested_mtu == 0)     // mtu wasn't specified on the command line
                {
                        uv->requested_mtu = 1500;       // the default value for RTP
                }

                if(bitrate == 0) { // else packet_rate defaults to 13600 or so
                        bitrate = 6618;
                }

                if(bitrate != -1) {
                        packet_rate = 1000 * uv->requested_mtu * 8 / bitrate;
                } else {
                        packet_rate = 0;
                }

                for(int i = 0; i < CAP_DEV_COUNT; ++i) {
                        if ((uv->send[i].tx = initialize_transmit(uv->requested_mtu, requested_fec)) == NULL) {
                                fprintf(stderr, "Unable to initialize transmitter.\n");
                                exit(EXIT_FAIL_TRANSMIT);
                        }
                }

                if ((uv->tx = initialize_transmit(uv->requested_mtu, requested_fec)) == NULL) {
                        printf("Unable to initialize transmitter.\n");
                        exit(EXIT_FAIL_TRANSMIT);
                }

                /* following block only shows help (otherwise initialized in receiver thread */
                if((uv->postprocess && strstr(uv->postprocess, "help") != NULL) || 
                                (uv->decoder_mode && strstr(uv->decoder_mode, "help") != NULL)) {
                        struct state_decoder *dec = decoder_init(uv->decoder_mode, uv->postprocess);
                        decoder_destroy(dec);
                        exit(0);
                }
                /* following block only shows help (otherwise initialized in sender thread */
                if(strstr(uv->requested_compression,"help") != NULL) {
                        struct compress_state *compression = compress_init(uv->requested_compression);
                        compress_done(compression);
                        exit(0);
                }
        }

#ifdef HAVE_GCOLL
        gcoll_params.front_ssrc = rtp_my_ssrc(uv->send[GCOLL_FRONT].network_device);
        gcoll_params.side_ssrc = rtp_my_ssrc(uv->send[GCOLL_SIDE].network_device);
        gcoll_params.group_ssrc = rtp_my_ssrc(uv->send[GCOLL_GROUP].network_device);
        // gcoll.params.send_group_camera set in getopt loop
        gcoll_params.audio_ssrc = audio_net_get_ssrc(uv->audio);
        gcoll_params.reflector_addr = uv->address;
        gcoll_params.port_number = uv->send_port_number;
        for (int i = 0; i < CAP_DEV_COUNT; i++) {
                gcoll_params.rtp_session[i] = uv->send[i].network_device;
        }
#endif // HAVE_GCOLL

        void *udata = NULL;
#ifdef HAVE_GCOLL
        udata = &gcoll_params;
#endif // HAVE_GCOLL

        if ((uv->display_device =
             initialize_video_display(uv->requested_display, display_cfg, display_flags, udata)) == NULL) {
                fprintf(stderr, "Unable to open display device: %s\n",
                       uv->requested_display);
                exit_uv(EXIT_FAIL_DISPLAY);
                goto cleanup_wait_audio;
        }

        printf("Display initialized-%s\n", uv->requested_display);

#ifndef WIN32
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        signal(SIGHUP, signal_handler);
        signal(SIGABRT, signal_handler);
#endif

#ifdef USE_RT
#ifdef HAVE_SCHED_SETSCHEDULER
        sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
        if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
                printf("WARNING: Unable to set real-time scheduling\n");
        }
#else
        printf("WARNING: System does not support real-time scheduling\n");
#endif /* HAVE_SCHED_SETSCHEDULER */
#endif /* USE_RT */         

        {
                if (strcmp("none", uv->requested_display) != 0) {
                        pthread_mutex_lock(&uv->master_lock); 
                        if (pthread_create
                            (&receiver_thread_id, NULL, receiver_thread,
                             (void *)uv) != 0) {
                                perror("Unable to create receiver thread!\n");
                                exit_uv(EXIT_FAIL_DISPLAY);
                                goto cleanup_wait_audio;
                        } else {
				receiver_thread_started = true;
			}
                }

                for(int i = 0; i < CAP_DEV_COUNT; ++i) {
                        if (strcmp("none", uv->send[i].requested_capture) != 0) {
                                pthread_mutex_lock(&uv->master_lock); 
                                if (pthread_create
                                                (&compress_thread_id[i], NULL, compress_thread,
                                                 (void *)uv) != 0) {
                                        perror("Unable to create capture thread!\n");
                                        exit_uv(EXIT_FAIL_CAPTURE);
                                        goto cleanup_wait_capture;
                                } else {
                                        compress_thread_started[i] = true;
                                }
                        }
                }
        }
        
        pthread_mutex_lock(&uv->master_lock); 

        if(audio_get_display_flags(uv->audio)) {
                audio_register_get_callback(uv->audio, (struct audio_frame * (*)(void *)) display_get_audio_frame, uv->display_device);
                audio_register_put_callback(uv->audio, (void (*)(void *, struct audio_frame *)) display_put_audio_frame, uv->display_device);
                audio_register_reconfigure_callback(uv->audio, (int (*)(void *, int, int, 
                                                        int)) display_reconfigure_audio, uv->display_device);
        }

        if (strcmp("none", uv->requested_display) != 0)
                display_run(uv->display_device);

//cleanup_wait_display:
        if (strcmp("none", uv->requested_display) != 0 && receiver_thread_started)
                pthread_join(receiver_thread_id, NULL);

cleanup_wait_capture:
        for(int i = 0; i < CAP_DEV_COUNT; ++i) {
                if (strcmp("none", uv->send[i].requested_capture) != 0 && compress_thread_started[i])
                        pthread_join(compress_thread_id[i],
                                        NULL);
        }

cleanup_wait_audio:
        /* also wait for audio threads */
        audio_join(uv->audio);

cleanup:
        while(wait_to_finish)
                ;
        threads_joined = TRUE;

        if(uv->audio)
                audio_done(uv->audio);
        for(int i = 0; i < CAP_DEV_COUNT; ++i) {
                if(uv->send[i].tx)
                        tx_done(uv->send[i].tx);
                if(uv->send[i].network_device)
                        destroy_devices(uv->send[i].network_device);
        }
        if(uv->tx)
                tx_done(uv->tx);
	if(uv->recv_network_device)
                destroy_devices(uv->recv_network_device);
        if(uv->display_device)
                display_done(uv->display_device);
        if (uv->participants != NULL) {
                struct pdb_e *cp = pdb_iter_init(uv->participants);
                while (cp != NULL) {
                        struct pdb_e *item = NULL;
                        pdb_remove(uv->participants, cp->ssrc, &item);
                        free(item);

                        cp = pdb_iter_next(uv->participants);
                }
                pdb_iter_done(uv->participants);
                pdb_destroy(&uv->participants);
        }

        for(int i = 0; i < CAP_DEV_COUNT; ++i) {
                pthread_mutex_destroy(&uv->send[i].sender_lock);
                pthread_cond_destroy(&uv->send[i].compress_thread_cv);
                pthread_cond_destroy(&uv->send[i].sender_cv);
        }

        pthread_mutex_unlock(&uv->master_lock); 

        pthread_mutex_destroy(&uv->master_lock);

        free(uv);
        
        lib_common_done();

#if defined DEBUG && defined HAVE_LINUX
        muntrace();
#endif

#ifdef WIN32
	WSACleanup();
#endif

        printf("Exit\n");

        return exit_status;
}
