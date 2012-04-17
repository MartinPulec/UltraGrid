#include "../include/UGReceiver.h"

#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <sys/time.h>

#include <string>

#include "../client_guiMain.h"
#include "../include/GLView.h"

/* UltraGrid bits */
extern "C" {
#include "debug.h"
#include "pdb.h"
#include "rtp/decoders.h"
#include "rtp/pbuf.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "tfrc.h"
#include "tv.h"
#include "video_capture.h"
#include "video_display.h"
#include "video_decompress.h"

#include "udt_receive.h"
#include "tcp_receive.h"
};

#define USE_CUSTOM_TRANSMIT

enum cmd {
    CMD_ACCEPT,
    CMD_QUIT,
    CMD_DISCONNECT,
    CMD_NONE
};

enum state {
    ST_NONE,
    ST_ACCEPTED,
    ST_EXIT
};


struct state_uv {
#ifndef USE_CUSTOM_TRANSMIT
        struct rtp **network_devices;
        struct pdb *participants;
#else
        void *receive_state;
#endif
        unsigned int connections_count;

        struct timeval start_time, curr_time;

        char *decoder_mode;
        char *postprocess;

        pthread_mutex_t lock;
        pthread_cond_t boss_cv;
        pthread_cond_t worker_cv;
        volatile bool boss_waiting;
        volatile bool worker_waiting;
        enum cmd command;
        enum state state;

        uint32_t ts;
        struct display *display_device;

        void * (* receive_init)(const char *address, unsigned int port);
        void (*receive_done)(void *state);
        int (*receive)(void *state, char *buffer, int *len);
        int (*receive_accept)(void *state, const char *remote_host, int remote_port);
        int (*receive_disconnect)(void *state);

        char * remote_host;
        int remote_port;

        bool use_tcp;
};

volatile int should_exit = FALSE;
uint32_t RTT = 0;               /* this is computed by handle_rr in rtp_callback */
struct video_frame *frame_buffer = NULL;
long packet_rate = 13600;
int uv_argc;
char **uv_argv;

void _exit_uv(int status) {
        should_exit = 1;
}

void (*exit_uv)(int status) = _exit_uv;

extern "C" {
struct display *initialize_video_display(const char *requested_display,
                                                char *fmt, unsigned int flags);


struct vidcap *initialize_video_capture(const char *requested_capture,
                                               char *fmt, unsigned int flags);

struct display *initialize_video_display(const char *requested_display,
                                                char *fmt, unsigned int flags)
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

        d = display_init(id, fmt, flags);
        return d;
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
};


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
		devices[index] = rtp_init(addr, port, port, ttl, rtcp_bw,
                                FALSE, rtp_recv_callback,
                                (uint8_t *)participants);
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

static void *receiver_thread(void *arg)
{
        struct state_uv *uv = (struct state_uv *)arg;

        struct pdb_e *cp;
        struct timeval timeout;
        int fr;
        int ret;
        unsigned int tiles_post = 0;
        struct timeval last_tile_received = {0, 0};
        struct pbuf_video_data pbuf_data;

        initialize_video_decompress();
        pbuf_data.decoder = decoder_init(uv->decoder_mode, uv->postprocess);
        if(!pbuf_data.decoder) {
                fprintf(stderr, "Error initializing decoder ('-M' option).\n");
                exit_uv(1);
        } else {
                decoder_register_video_display(pbuf_data.decoder, uv->display_device);
        }
        pbuf_data.frame_buffer = frame_buffer;

        fr = 1;

        while (1) {
#ifndef USE_CUSTOM_TRANSMIT
                abort();

                /* Housekeeping and RTCP... */
                gettimeofday(&uv->curr_time, NULL);
                uv->ts = tv_diff(uv->curr_time, uv->start_time) * 90000;
                rtp_update(uv->network_devices[0], uv->curr_time);
                rtp_send_ctrl(uv->network_devices[0], uv->ts, 0, uv->curr_time);

                /* Receive packets from the network... The timeout is adjusted */
                /* to match the video capture rate, so the transmitter works.  */
                if (fr) {
                        gettimeofday(&uv->curr_time, NULL);
                        fr = 0;
                }

                timeout.tv_sec = 0;
                timeout.tv_usec = 999999 / 59.94;
                ret = rtp_recv_poll_r(uv->network_devices, &timeout, uv->ts);

                /*
                   if (ret == FALSE) {
                   printf("Failed to receive data\n");
                   }
                 */

                /* Decode and render for each participant in the conference... */
                cp = pdb_iter_init(uv->participants);
                while (cp != NULL) {
                        if (tfrc_feedback_is_due(cp->tfrc_state, uv->curr_time)) {
                                debug_msg("tfrc rate %f\n",
                                          tfrc_feedback_txrate(cp->tfrc_state,
                                                               uv->curr_time));
                        }

                        /* Decode and render video... */
                        if (pbuf_decode
                            (cp->playout_buffer, uv->curr_time, decode_frame, &pbuf_data, TRUE)) {
                                tiles_post++;
                                /* we have data from all connections we need */
                                if(tiles_post == uv->connections_count)
                                {
                                        tiles_post = 0;
                                        gettimeofday(&uv->curr_time, NULL);
                                        fr = 1;
                                        display_put_frame(uv->display_device,
                                                          (char *) pbuf_data.frame_buffer);
                                        pbuf_data.frame_buffer =
                                            display_get_frame(uv->display_device);
                                }
                                last_tile_received = uv->curr_time;
                        }
                        pbuf_remove(cp->playout_buffer, uv->curr_time);
                        cp = pdb_iter_next(uv->participants);
                }
                pdb_iter_done(uv->participants);

                /* dual-link TIMEOUT - we won't wait for next tiles */
                if(tiles_post > 1 && tv_diff(uv->curr_time, last_tile_received) >
                                999999 / 59.94 / uv->connections_count) {
                        tiles_post = 0;
                        gettimeofday(&uv->curr_time, NULL);
                        fr = 1;
                        display_put_frame(uv->display_device,
                                          pbuf_data.frame_buffer->tiles[0].data);
                        pbuf_data.frame_buffer =
                            display_get_frame(uv->display_device);
                        last_tile_received = uv->curr_time;
                }
#else
                // LOCK - LOCK - LOCK
                pthread_mutex_lock(&uv->lock);

                bool accept = false;

                // cannot be while !!!
                // TODO: figure out more cleaner way
                if(uv->boss_waiting) {
                    uv->worker_waiting = true;
                    pthread_cond_wait(&uv->worker_cv, &uv->lock);
                    uv->worker_waiting = false;

                    switch(uv->command) {
                        case CMD_ACCEPT:
                            accept = true;
                            std::cerr << "UGReceiver command: accept" << std::endl;
                            uv->state = ST_ACCEPTED;
                            break;
                        case CMD_QUIT:
                            uv->state = ST_EXIT;
                            break;
                        case CMD_DISCONNECT:
                            ret = uv->receive_disconnect(uv->receive_state);
                            std::cerr << "UGReceiver command: disconnect" << std::endl;
                            uv->state = ST_NONE;
                            break;
                        case CMD_NONE:
                            break;
                    }

                    uv->command = CMD_NONE;

                    pthread_cond_signal(&uv->boss_cv);
                }

                // UNLOCK - UNLOCK - UNLOCK
                pthread_mutex_unlock(&uv->lock);

                if(accept) {
                    while(uv->command == CMD_NONE) {
                        ret = uv->receive_accept(uv->receive_state, uv->remote_host, uv->remote_port);

                        if(!ret) {
                            fprintf(stderr, "Failed to accept\n");
                        } else {
                            uv->state = ST_ACCEPTED;
                            fprintf(stderr, "Accepted\n");
                            break;
                        }
                    }
                }

                if(uv->state == ST_EXIT) {
                    goto quit;
                } else if(uv->state == ST_NONE) {
                    continue; // jump to next iteration and wait for command
                } else if(uv->state == ST_ACCEPTED) {
                }
                // accepted

                if(uv->use_tcp) {
                    int res, data_len;
                    int total_received;
                    video_payload_hdr_t header;

                    int len = sizeof(header);
                    char *buffer;
                    res = uv->receive(uv->receive_state, (char *) &header, &len);

                    if(!res) {
                        std::cerr << "(res: " << res << ", len: " << len << ", sizeof(video_payload_hdr_t): " << sizeof(video_payload_hdr_t) << ")"  << std::endl;
                        goto error;
                    }

                    data_len = decoder_reconfigure((char *) &header, len, &pbuf_data);
                    decoder_get_buffer(&pbuf_data, &buffer, &len);


                    total_received = 0;

                    while(total_received < data_len) {
                        len = data_len - total_received;

                        res = uv->receive(uv->receive_state, buffer + total_received, &len);
                        if(!res) {
                            std::cerr << "(res: " << res << ")" << std::endl;
                        }

                        total_received += len;
                    }

                    decoder_decode(pbuf_data.decoder, &pbuf_data, buffer, len, pbuf_data.frame_buffer);

                    display_put_frame(uv->display_device, (char *) pbuf_data.frame_buffer);
                    pbuf_data.frame_buffer = display_get_frame(uv->display_device);
                } else { //UDT
                    int res, data_len;
                    struct {
                        video_payload_hdr_t header;
                        char pad[8];
                    } a;
                    int len = sizeof(a) + 1;
                    char *buffer;
                    res = uv->receive(uv->receive_state, (char *) &a.header, &len);
                    if(!res) {
                        std::cerr << "(res: " << res << ", len: " << len << ", sizeof(video_payload_hdr_t): " << sizeof(video_payload_hdr_t) << ")"  << std::endl;
                        goto error;
                    }
                    if(len != sizeof(video_payload_hdr_t)) {
                        std::cerr << "(len: " << len << ", sizeof(video_payload_hdr_t): " << sizeof(video_payload_hdr_t) << ")"  << std::endl;
                        goto error;
                    }

                    data_len = decoder_reconfigure((char *) &a.header, len, &pbuf_data);
                    decoder_get_buffer(&pbuf_data, &buffer, &len);
                    res = uv->receive(uv->receive_state, buffer, &len);
                    if(!res) {
                        std::cerr << "(res: " << res << ")" << std::endl;
                        goto error;
                    }
                    if(len != data_len) {
                        std::cerr << "(len: " << len << ", data_len: " << data_len  << ")" << std::endl;
                        goto error;
                    }

                    decoder_decode(pbuf_data.decoder, &pbuf_data, buffer, len, pbuf_data.frame_buffer);

                    display_put_frame(uv->display_device, (char *) pbuf_data.frame_buffer);
                    pbuf_data.frame_buffer = display_get_frame(uv->display_device);
                }
error:
                ;

no_err:
                ;
#endif
        }

quit:

        decoder_destroy(pbuf_data.decoder);


        std::cerr << "UGRECEIVER THREAD EXITED";

        return 0;
}



UGReceiver::UGReceiver(const char *display, VideoBuffer *buffer, bool use_tcp)
{
    pthread_t receiver_thread_id;

    uv = (struct state_uv *) malloc(sizeof(struct state_uv));
#ifndef USE_CUSTOM_TRANSMIT
    uv->participants = pdb_init();
    uv->network_devices = initialize_network(strdup("localhost"), 5004, uv->participants);
#endif

    uv->use_tcp = use_tcp;
    uv->receive_state = 0;

    if(uv->use_tcp) {
        std::cerr << "Tramnsmit: TCP" << std::endl;
        uv->receive_init = tcp_receive_init;
        uv->receive_done = tcp_receive_done;
        uv->receive = tcp_receive;
        uv->receive_accept = tcp_receive_accept;
        uv->receive_disconnect = tcp_receive_disconnect;
    } else {
        std::cerr << "Tramnsmit: UDT" << std::endl;
        uv->receive_init = udt_receive_init;
        uv->receive_done = udt_receive_done;
        uv->receive = udt_receive;
        uv->receive_accept = udt_receive_accept;
        uv->receive_disconnect = udt_receive_disconnect;
    }

    uv->receive_state = uv->receive_init("localhost", 5004);

    uv->connections_count = 1;
    gettimeofday(&uv->start_time, NULL);
    uv->decoder_mode = NULL;
    uv->postprocess = NULL;
    uv->remote_host = NULL;

    pthread_mutex_init(&uv->lock, NULL);
    pthread_cond_init(&uv->boss_cv, NULL);
    pthread_cond_init(&uv->worker_cv, NULL);
    uv->boss_waiting = true;
    uv->worker_waiting = true;

    uv->command = CMD_NONE;
    uv->state = ST_NONE;

    if(strcmp(display, "wxgl") == 0) {
        uv->display_device = initialize_video_display(display, (char *) buffer, 0 /*flags */);
    } else {
        abort(); // is still supported ?
        uv->display_device = initialize_video_display(display, NULL, 0 /*flags */);
    }

    sigset_t mask;
    sigset_t oldmask;

    sigfillset(&mask);
    pthread_sigmask(SIG_BLOCK, &mask, &oldmask);


    if (pthread_create
        (&receiver_thread_id, NULL, receiver_thread,
         (void *)uv) != 0) {
            perror("Unable to create display thread!\n");
            // TODO handle error
    }

    pthread_sigmask(SIG_SETMASK, &oldmask, &mask);
}

void UGReceiver::Accept(const char *remote_host, int remote_port)
{
    pthread_mutex_lock(&uv->lock);

    uv->command = CMD_ACCEPT;
    free(uv->remote_host);

    uv->remote_host = strdup(remote_host);
    uv->remote_port = remote_port;

    uv->boss_waiting = true;
    pthread_mutex_unlock(&uv->lock);

    while(!uv->worker_waiting)
        ;

    pthread_mutex_lock(&uv->lock);
    if(uv->worker_waiting)
        pthread_cond_signal(&uv->worker_cv);

    while(!uv->worker_waiting) {
        pthread_cond_wait(&uv->boss_cv, &uv->lock);
    }
    uv->boss_waiting = false;

    pthread_mutex_unlock(&uv->lock);
}

void UGReceiver::Disconnect()
{
    pthread_mutex_lock(&uv->lock);

    uv->command = CMD_DISCONNECT;

    uv->boss_waiting = true;
    pthread_mutex_unlock(&uv->lock);

    while(!uv->worker_waiting)
        ;

    pthread_mutex_lock(&uv->lock);

    if(uv->worker_waiting)
        pthread_cond_signal(&uv->worker_cv);

    while(!uv->worker_waiting) {
        pthread_cond_wait(&uv->boss_cv, &uv->lock);
    }
    uv->boss_waiting = false;

    pthread_mutex_unlock(&uv->lock);
}

UGReceiver::~UGReceiver()
{
#ifdef DEBUG
    std::cerr << "STARTING UGRECEIVER EXIT" << std::endl;
#endif
    pthread_mutex_lock(&uv->lock);

    uv->command = CMD_QUIT;

    uv->boss_waiting = true;
    pthread_mutex_unlock(&uv->lock);

    while(!uv->worker_waiting)
        ;

    pthread_mutex_lock(&uv->lock);

    if(uv->worker_waiting)
        pthread_cond_signal(&uv->worker_cv);


    while(!uv->worker_waiting) {
        pthread_cond_wait(&uv->boss_cv, &uv->lock);
    }
    uv->boss_waiting = false;

    pthread_mutex_unlock(&uv->lock);

#ifdef DEBUG
    std::cerr << "UGRECEIVER SHOULD HAVE EXITED" << std::endl;
#endif
}
