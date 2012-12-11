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
#endif /* HAVE_CONFIG_H */

#include <getopt.h>
#include <string.h>
#include <string>
#include <stdlib.h>
#include <sys/prctl.h>

#include "aes_encrypt.h"
#include "audio_source.h"
#include "color_transform.h"
#include "compat/platform_semaphore.h"
#include "debug.h"
#include "gl_context.h"
#include "messaging.h"
#include "pdb.h"
#include "perf.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "server/streaming_server.h"
#include "tcp_transmit.h"
#include "tfrc.h"
#include "tile.h"
#include "transmit.h"
#include "tv.h"
#include "udt_transmit.h"
#include "video_codec.h"
#include "video_capture.h"
#include "video_compress.h"
#include "watermark.h"

// ifdef DUMP
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
// endif DUMP

using namespace std;

#define EXIT_FAIL_USAGE		1
#define EXIT_FAIL_UI   		2
#define EXIT_FAIL_CAPTURE	4
#define EXIT_FAIL_NETWORK	5
#define EXIT_FAIL_TRANSMIT	6
#define EXIT_FAIL_COMPRESS	7

#define PORT_BASE               5004
#define PORT_AUDIO              5006

#define USE_CUSTOM_TRANSMIT 1

const char * volatile video_directory = 0;
volatile int logo_hidden = 0;

struct state_uv {
#ifndef USE_CUSTOM_TRANSMIT
        struct rtp **network_devices;

        struct pdb *participants;
        struct tx *tx;
#else
        void  *transmit_state;
        void * (*transmit_init)(char *address, unsigned int *port);
        void (*transmit_accept)(void *state);
        void (*transmit_done)(void *state);
        void (*transmit_send)(void *state, struct video_frame *frame, struct audio_frame *);
#endif
        unsigned int port_number;
        unsigned int connections_count;

        struct vidcap *capture_device;
        struct timeval start_time, curr_time;

        char *postprocess;

        uint32_t ts;
        char *compress_options;
        int requested_compression;
        const char *requested_capture;
        unsigned requested_mtu;

        volatile unsigned int grab_thread_ready:1;
        volatile unsigned int accepted:1;

        struct audio_source *audio_source;

        int comm_fd;

        struct compress_state *compression;
        volatile double compress_quality;
};

long packet_rate = 13600;
volatile int should_exit = FALSE;
volatile int wait_to_finish = FALSE;
volatile int threads_joined = FALSE;
static int exit_status = EXIT_SUCCESS;

uint32_t RTT = 0;               /* this is computed by handle_rr in rtp_callback */
struct video_frame *frame_buffer = NULL;
uint32_t hd_color_spc = 0;

long frame_begin[2];

int uv_argc;
char **uv_argv;
static struct state_uv *uv_state;

void list_video_capture_devices(void);
struct vidcap *initialize_video_capture(const char *requested_capture,
                                               char *fmt, unsigned int flags);
static void *sender_thread(void *arg);

#ifndef WIN32
static void signal_handler(int signal)
{
        fprintf(stderr, "Caught signal %d\n", signal);
        exit_uv(0);
        return;
}
#endif                          /* WIN32 */

void _exit_uv(int status);

void _exit_uv(int status) {
        exit_status = status;
        wait_to_finish = TRUE;
        should_exit = TRUE;
        if(!threads_joined) {
                if(uv_state->capture_device)
                        vidcap_finish(uv_state->capture_device);
        }
        wait_to_finish = FALSE;
        close(uv_state->comm_fd);
}

void (*exit_uv)(int status) = _exit_uv;

static void usage(void)
{
        /* TODO -c -p -b are deprecated options */
        printf("\nUsage: uv [-d <display_device>] [-t <capture_device>] [-r <audio_playout>] [-s <audio_caputre>] \n");
        printf("          [-m <mtu>] [-c] [-i] [-M <video_mode>] [-p <postprocess>] [-f <FEC_options>] [-P <port>] address(es)\n\n");
        printf
            ("\t-d <display_device>        \tselect display device, use '-d help' to get\n");
        printf("\t                         \tlist of supported devices\n");
        printf("\n");
        printf
            ("\t-t <capture_device>        \tselect capture device, use '-t help' to get\n");
        printf("\t                         \tlist of supported devices\n");
        printf("\n");
        printf("\t-c <cfg>                 \tcompress video (see '-c help')\n");
        printf("\n");
        printf("\t-r <playback_device>     \tAudio playback device (see '-r help')\n");
        printf("\n");
        printf("\t-s <capture_device>      \tAudio capture device (see '-s help')\n");
        printf("\n");
        printf("\t-j <settings>            \tJACK Audio Connection Kit settings (see '-j help')\n");
        printf("\n");
        printf("\t-M <video_mode>          \treceived video mode (eg tiled-4K, 3D, dual-link)\n");
        printf("\n");
        printf("\t-p <postprocess>         \tpostprocess module\n");
        printf("\n");
        printf("\t-f <settings>            \tconfig forward error checking, currently \"XOR:<leap>:<nr_streams>\" or \"mult:<nr>\"\n");
        printf("\n");
        printf("\t-P <port>                \tbase port number, also 3 subsequent ports can be used (default: %d)\n", PORT_BASE);
        printf("\n");
        printf("\taddress(es)              \tdestination address\n");
        printf("\n");
        printf("\t                         \tIf comma-separated list of addresses\n");
        printf("\t                         \tis entered, video frames are split\n");
        printf("\t                         \tand chunks are sent/received independently.\n");
        printf("\n");
}

void list_video_capture_devices()
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

#if ! defined USE_CUSTOM_TRANSMIT
static struct rtp **initialize_network(char *addrs, int port_base, struct pdb *participants)
{
	struct rtp **devices = NULL;
        double rtcp_bw = 5 * 1024 * 1024;       /* FIXME */
	int ttl = 255;
	char *saveptr = NULL;
	char *addr;
	char *tmp;
	int required_connections, index;
        int port = port_base;

	tmp = strdup(addrs);
	if(strtok_r(tmp, ",", &saveptr) == NULL) {
		free(tmp);
		return NULL;
	}
	else required_connections = 1;
	while(strtok_r(NULL, ",", &saveptr) != NULL)
		++required_connections;

	free(tmp);
	tmp = strdup(addrs);

	devices = (struct rtp **)
		malloc((required_connections + 1) * sizeof(struct rtp *));

	for(index = 0, addr = strtok_r(addrs, ",", &saveptr);
		index < required_connections;
		++index, addr = strtok_r(NULL, ",", &saveptr), port += 2)
	{
                if (port == PORT_AUDIO)
                        port += 2;
		devices[index] = rtp_init(addr, port - 100, port, ttl, rtcp_bw,
                                FALSE, rtp_recv_callback,
                                (void *)participants);
		if (devices[index] != NULL) {
			rtp_set_option(devices[index], RTP_OPT_WEAK_VALIDATION,
				TRUE);
			rtp_set_sdes(devices[index], rtp_my_ssrc(devices[index]),
				RTCP_SDES_TOOL,
				PACKAGE_STRING, strlen(PACKAGE_STRING));
#ifdef HAVE_MACOSX
                        rtp_set_recv_buf(devices[index], 5944320);
#else
                        rtp_set_recv_buf(devices[index], 8*1024*1024);
#endif

			pdb_add(participants, rtp_my_ssrc(devices[index]));
		}
		else {
			int index_nest;
			for(index_nest = 0; index_nest < index; ++index_nest) {
				rtp_done(devices[index_nest]);
			}
			free(devices);
			devices = NULL;
		}
	}
	if(devices != NULL) devices[index] = NULL;
	free(tmp);

        return devices;
}

static void destroy_devices(struct rtp ** network_devices)
{
	struct rtp ** current = network_devices;
        if(!network_devices)
                return;
	while(*current != NULL) {
		rtp_done(*current++);
	}
	free(network_devices);
}

static struct tx *initialize_transmit(unsigned requested_mtu, char *fec)
{
        /* Currently this is trivial. It'll get more complex once we */
        /* have multiple codecs and/or error correction.             */
        return tx_init(requested_mtu, fec);
}
#endif

static void *sender_thread(void *arg)
{
        struct state_uv *uv = (struct state_uv *) arg;
        struct video_frame *tx_frame;

        aes_encrypt enc;

        prctl(PR_SET_NAME, (unsigned long) __func__, 0, 0);

        while(1) {
                tx_frame = compress_frame_pop(uv->compression);
                int len = tx_frame->tiles[0].data_len;
                unsigned char *enc_data = enc.encrypt((unsigned char *) tx_frame->tiles[0].data,
                                &len);
                tx_frame->deleter(tx_frame->tiles[0].data, tx_frame->tiles[0].data_len);
                tx_frame->deleter = default_free;
                tx_frame->tiles[0].data = (char *) enc_data;
                tx_frame->tiles[0].data_len = len;

                if(!tx_frame) {
                        break;
                }

#ifdef USE_CUSTOM_TRANSMIT
                struct audio_frame *audio = audio_source_read(uv->audio_source, tx_frame->frames);
                uv->transmit_send(uv->transmit_state, tx_frame, audio);
#else
                tx_send(uv->tx, tx_frame,
                                uv->network_devices[0]);

#endif
                vf_free_data(tx_frame);
        }

        return NULL;
}

static void *grab_thread(void *arg)
{
        pthread_t sender_thread_id;

        struct state_uv *uv = (struct state_uv *)arg;

        struct video_frame *tx_frame;
        struct audio_frame *audio;

        prctl(PR_SET_NAME, (unsigned long) __func__, 0, 0);
#if 0
        struct state_color_transform *color_transform = NULL;

        struct state_watermark *watermark;

        init_gl_context(&context);
        color_transform = color_transform_init(&context);
        watermark = watermark_init(&context);

        if(context.context == NULL) {
                fprintf(stderr, "Error initializing GL context.\n");
                abort();
        }
#endif

        uv->compression = compress_init(uv->compress_options);
        if(uv->requested_compression
                        && uv->compression == NULL) {
                fprintf(stderr, "Error initializing compression.\n");
                exit_uv(0);
        }

        // must be called after compression initialized
        if (pthread_create
            (&sender_thread_id, NULL, sender_thread,
             (void *)uv) != 0) {
                perror("Unable to create sender thread!\n");
                exit_uv(EXIT_FAILURE);
                return NULL;
        }

        uv->grab_thread_ready = TRUE;
        while(!uv->accepted && !should_exit)
            ;

        while (!should_exit) {
                /* Capture and transmit video... */
                tx_frame = vidcap_grab(uv->capture_device, &audio);
                if (tx_frame != NULL) {
#if 0
                        struct video_frame *after_transform, *with_watermark;
                        after_transform = color_transform_transform(color_transform, tx_frame);
                        with_watermark = add_watermark(watermark, after_transform);
#endif
                        //TODO: Unghetto this
                        if (uv->requested_compression) {
                                compress_frame_push(uv->compression, tx_frame, uv->compress_quality);
                        } else {
                                fprintf(stderr, "Compression needed!\n");
                                abort();
                        }
                }
        }

        // poisoned frame
        compress_frame_push(uv->compression, NULL, uv->compress_quality);

        pthread_join(sender_thread_id, NULL);

#ifdef USE_CUSTOM_TRANSMIT
        uv->transmit_done(uv->transmit_state);
#endif

        compress_done(uv->compression);
#if 0
        watermark_done(watermark);
        destroy_gl_context(&context);
#endif

        return NULL;
}

int main(int argc, char *argv[])
{
        uv_argc = argc;
        uv_argv = argv;

        if (argc == 1) {
                return main_sp();
        }

#if defined HAVE_SCHED_SETSCHEDULER && defined USE_RT
        struct sched_param sp;
#endif
        char *network_device = NULL;

        char *capture_cfg = NULL;
        char *jack_cfg = NULL;
        UNUSED(jack_cfg);
        char *requested_fec = NULL;
        UNUSED(requested_fec);
        char *save_ptr = NULL;
        sigset_t mask;
        char *audio_source = NULL;

        struct state_uv *uv;
        int ch;

        pthread_t grab_thread_id;
        unsigned vidcap_flags = 0;

        static struct option getopt_options[] = {
                {"communication", required_argument, 0, 'C'},
                {"capture", required_argument, 0, 't'},
                {"mtu", required_argument, 0, 'm'},
                {"mode", required_argument, 0, 'M'},
                {"version", no_argument, 0, 'v'},
                {"compress", required_argument, 0, 'c'},
                {"receive", required_argument, 0, 'r'},
                {"send", required_argument, 0, 's'},
                {"help", no_argument, 0, 'h'},
                {"jack", required_argument, 0, 'j'},
                {"fec", required_argument, 0, 'f'},
                {"port", required_argument, 0, 'P'},
#ifdef USE_CUSTOM_TRANSMIT
                {"tcp", no_argument, 0, 'T'},
#endif
                {0, 0, 0, 0}
        };
        int option_index = 0;

        //      uv = (struct state_uv *) calloc(1, sizeof(struct state_uv));
        uv = (struct state_uv *)malloc(sizeof(struct state_uv));
        uv_state = uv;

        uv->ts = 0;
        uv->requested_capture = "none";
        uv->requested_compression = TRUE;
        uv->compress_options = "none";
        uv->compress_quality = 1.0;
        uv->postprocess = NULL;
        uv->requested_mtu = 0;
#ifdef USE_CUSTOM_TRANSMIT
        uv->transmit_init = udt_transmit_init;
        uv->transmit_accept = udt_transmit_accept;
        uv->transmit_done = udt_transmit_done;
        uv->transmit_send = udt_send;
#else
        uv->participants = NULL;
        uv->tx = NULL;
        uv->network_devices = NULL;
#endif
        uv->port_number = PORT_BASE;
	uv->comm_fd = 0;
        uv->grab_thread_ready = FALSE;
        uv->accepted = FALSE;

        perf_init();
        perf_record(UVP_INIT, 0);

        while ((ch =
                getopt_long(argc, argv, "d:t:m:r:s:vc:ihj:M:p:f:P:C:a:"
#ifdef USE_CUSTOM_TRANSMIT
                        "T"
#endif
                        , getopt_options,
                            &option_index)) != -1) {
                switch (ch) {
                case 't':
                        if (!strcmp(optarg, "help")) {
                                list_video_capture_devices();
                                return 0;
                        }
                        uv->requested_capture = strtok_r(optarg, ":", &save_ptr);
                        if(save_ptr && strlen(save_ptr) > 0)
                                capture_cfg = save_ptr;
                        break;
                case 'm':
                        uv->requested_mtu = atoi(optarg);
                        break;
                case 'a':
                        audio_source = optarg;
                        break;
                case 'p':
                        uv->postprocess = optarg;
                        break;
                case 'v':
                        printf("%s\n", PACKAGE_STRING);
                        return EXIT_SUCCESS;
                case 'c':
                        uv->requested_compression = TRUE;
                        uv->compress_options = optarg;
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
                        uv->port_number = atoi(optarg);
                        break;
                case 'C':
                        uv->comm_fd = atoi(optarg);
                        break;
#ifdef USE_CUSTOM_TRANSMIT
                case 'T':
                        uv->transmit_init = tcp_transmit_init;
                        uv->transmit_accept = tcp_transmit_accept;
                        uv->transmit_done = tcp_transmit_done;
                        uv->transmit_send = tcp_send;
                        break;
#endif
                case '?':
                        break;
                default:
                        usage();
                        return EXIT_FAIL_USAGE;
                }
        }

        argc -= optind;
        argv += optind;

        prctl(PR_SET_NAME, (unsigned long) __func__, 0, 0);

        sigemptyset(&mask);
        sigaddset(&mask, SIGINT);
        sigaddset(&mask, SIGTERM);
        sigaddset(&mask, SIGQUIT);
        sigaddset(&mask, SIGHUP);
        sigaddset(&mask, SIGABRT);
        pthread_sigmask(SIG_BLOCK, &mask, NULL);

        if (argc == 0) {
                network_device = strdup("localhost");
        } else {
                network_device = (char *) argv[0];
        }

        uv->audio_source = audio_source_init(audio_source);
        
        if(audio_source && !uv->audio_source) {
                fprintf(stderr, "Unable to initialize audio source.\n");
                return EXIT_FAILURE;
        }

        printf("%s\n", PACKAGE_STRING);
        printf("Capture device: %s\n", uv->requested_capture);
        printf("MTU           : %d\n", uv->requested_mtu);
        /*printf("Compression   : ");
        if (uv->requested_compression) {
                printf("%s", get_compress_name(uv->compression));
        } else {
                printf("none");
        }
        printf("\n");*/

        printf("Network protocol: ultragrid rtp\n");

        gettimeofday(&uv->start_time, NULL);

        if ((uv->capture_device =
                        initialize_video_capture(uv->requested_capture, capture_cfg, vidcap_flags)) == NULL) {
                printf("Unable to open capture device: %s\n",
                       uv->requested_capture);
                exit_status = EXIT_FAIL_CAPTURE;
                goto cleanup;
        }
        printf("Video capture initialized-%s\n", uv->requested_capture);


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

        if (uv->requested_mtu == 0)     // mtu wasn't specified on the command line
        {
                uv->requested_mtu = 1500;       // the default value for rpt
        }

#ifndef USE_CUSTOM_TRANSMIT
        if ((uv->network_devices =
                                initialize_network(network_device, uv->port_number, uv->participants)) == NULL) {
                printf("Unable to open network\n");
                exit_status = EXIT_FAIL_NETWORK;
                goto cleanup;
        } else {
                struct rtp **item;
                uv->connections_count = 0;
                /* only count how many connections has initialize_network opened */
                for(item = uv->network_devices; *item != NULL; ++item)
                        ++uv->connections_count;
        }
        uv->participants = pdb_init();

        if ((uv->tx = initialize_transmit(uv->requested_mtu, requested_fec)) == NULL) {
                printf("Unable to initialize transmitter\n");
                exit_status = EXIT_FAIL_TRANSMIT;
                goto cleanup;
        }
#else
        uv->connections_count = 1;
        uv->transmit_state = uv->transmit_init(network_device, &uv->port_number);
        if(!uv->transmit_state) {
                exit_status = EXIT_FAILURE;
                fprintf(stderr, "Unable to connect to peer.\n");
                goto cleanup;
        }
#endif

        /* following block only shows help (otherwise initialized in sender thread */
        if(uv->requested_compression && strstr(uv->compress_options,"help") != NULL) {
                struct compress_state *compression = compress_init(uv->compress_options);
                compress_done(compression);
                exit_status = EXIT_SUCCESS;
                goto cleanup;
        }

        if (strcmp("none", uv->requested_capture) != 0) {
                if (pthread_create
                                (&grab_thread_id, NULL, grab_thread,
                                 (void *)uv) != 0) {
                        perror("Unable to create capture thread!\n");
                        exit_status = 1;
                        goto cleanup;
                }
        }

        pthread_sigmask(SIG_UNBLOCK, &mask, NULL);
#ifndef WIN32
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        signal(SIGQUIT, signal_handler);
        signal(SIGHUP, signal_handler);
        signal(SIGABRT, signal_handler);
#endif
        {
                int len;
                ssize_t ret;
                ssize_t total = 0;

                while(!uv->grab_thread_ready && !should_exit)
                        ;

                char buff[1000];
                snprintf(buff, sizeof(buff), "%d", uv->port_number);
                len = strlen(buff);

                do {
                        ret = write(uv->comm_fd, (void *) &len, sizeof(int));
                        assert(ret > 0);
                        total += ret;
                } while (total < (int) sizeof(int));

                total = 0;
                do {
                        ret = write(uv->comm_fd, (const char *) &buff + total, len - total);
                        assert(ret > 0);
                        total += ret;
                } while (total < len);
        }

#ifdef DEBUG
        fprintf(stderr, "Sent port\n");
#endif

        uv->transmit_accept(uv->transmit_state);

        uv->accepted = TRUE;

        /* FlashNET loop */
        while(!should_exit) {
                char buff[1024];
                ssize_t ret;
                int len;
		if(uv->comm_fd != 0) {
			ret = read(uv->comm_fd, (void *) &len, sizeof(int));
		} else {
			len = 1024;
			ret = 0;
		}
                if(ret != -1) {
                        ret = read(uv->comm_fd, buff, len);
			len = ret;
                        buff[len] = '\0';
                        fprintf(stderr, "main: command %s\n", buff);
                        if(ret != -1) {
                                if(strncmp(buff, "PAUSE", strlen("PAUSE")) == 0) {
                                        vidcap_command(uv->capture_device, VIDCAP_PAUSE, NULL);
                                } else
                                /* note PLAY is prefix of PLAYONE, so this must be first */
                                if(strncmp(buff, "PLAYONE", strlen("PLAYONE")) == 0) {
                                        int count = atoi(buff + strlen("PLAYONE") + 1);
                                        vidcap_command(uv->capture_device, VIDCAP_PLAYONE, &count);
                                } else if(strncmp(buff, "PLAY", strlen("PLAY")) == 0) {
                                        vidcap_command(uv->capture_device, VIDCAP_PLAY, NULL);
                                } else if(strncmp(buff, "FPS", strlen("FPS")) == 0) {
                                        float fps = atof(buff + strlen("FPS") + 1);
                                        vidcap_command(uv->capture_device, VIDCAP_FPS, (void *) &fps);
                                        double fps_dbl = fps;
                                        audio_source_set_property(uv->audio_source,
                                                        AUDIO_SOURCE_PROPERTY_FPS, (void *) &fps_dbl, sizeof(fps_dbl));
                                } else if(strncmp(buff, "SETPOS", strlen("SETOPS")) == 0) {
                                        int pos = atoi(buff + strlen("SETPOS") + 1);
                                        vidcap_command(uv->capture_device, VIDCAP_POS, (void *) &pos);
                                } else if(strncmp(buff, "LOOP", strlen("LOOP")) == 0) {
                                        int val;
                                        if(strcmp(buff + strlen("LOOP") + 1, "ON") == 0)
                                                val = TRUE;
                                        else
                                                val = FALSE;
                                        vidcap_command(uv->capture_device, VIDCAP_LOOP, (void *) &val);
                                } else if(strncmp(buff, "SPEED", strlen("SPEED")) == 0) {
                                        float speed = atof(buff + strlen("SPEED") + 1);
                                        vidcap_command(uv->capture_device, VIDCAP_SPEED, (void *) &speed);
                                } else if(strncmp(buff, "MODULE", strlen("MODULE")) == 0) {
                                        struct text_message message;
                                        message.text = string(buff + strlen("MODULE") + 1);
                                        message_manager.broadcast(&message);
                                } else if(strncmp(buff, "QUALITY", strlen("QUALITY")) == 0) {
                                        uv->compress_quality =
                                                atof(buff + strlen("QUALITY") + 1);
                                } else if(strncmp(buff, "HIDE_LOGO", strlen("HIDE_LOGO")) == 0) {
                                        logo_hidden = 1;
                                }
                        }
                }
        }

        if (strcmp("none", uv->requested_capture) != 0 || uv->requested_compression)
                pthread_join(grab_thread_id, NULL);

        /* also wait for audio threads */

cleanup:
        while(wait_to_finish)
                ;
        threads_joined = TRUE;

        if(uv->capture_device)
                vidcap_done(uv->capture_device);
#ifndef USE_CUSTOM_TRANSMIT
        if(uv->tx)
                tx_done(uv->tx);
	if(uv->network_devices)
                destroy_devices(uv->network_devices);
        if (uv->participants != NULL)
                pdb_destroy(&uv->participants);
#endif
        printf("Exit\n");

        return exit_status;
}
