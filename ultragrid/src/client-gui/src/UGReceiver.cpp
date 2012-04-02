#include <stdint.h>
#include <sys/time.h>

#include "../include/UGReceiver.h"
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
};

#define USE_UDT 1

struct state_uv {
#ifndef USE_UDT
        struct rtp **network_devices;
        struct pdb *participants;
#else
        struct udt_recv *udt_receive;
#endif
        unsigned int connections_count;

        struct timeval start_time, curr_time;

        char *decoder_mode;
        char *postprocess;

        uint32_t ts;
        struct display *display_device;
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

#ifdef USE_UDT
        uv->udt_receive = udt_receive_init("localhost", 5004);
#endif

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

        /*struct video_frame *frame = vf_alloc(1);
        frame->tiles[0].data_len = 5 * 1024 *1024;
        frame->tiles[0].data =  new char[5 * 1024 *1024];*/
        ret = udt_receive_accept(uv->udt_receive);

        if(!ret) {
            fprintf(stderr, "Failed to accept");
        }

        while (!should_exit) {
#ifndef USE_UDT
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
                int res, data_len;
                video_payload_hdr_t header;
                int len = sizeof(header);
                char *buffer;
                res = udt_receive(uv->udt_receive, (char *) &header, &len);
                if(!res || len != sizeof(video_payload_hdr_t))
                    goto error;
                data_len = decoder_reconfigure((char *) &header, len, &pbuf_data);
                decoder_get_buffer(&pbuf_data, &buffer, &len);
                res = udt_receive(uv->udt_receive, buffer, &len);
                if(!res || len != data_len) {
                    goto error;
                }
                decoder_decode(pbuf_data.decoder, &pbuf_data, buffer, len, pbuf_data.frame_buffer);

                display_put_frame(uv->display_device, (char *) pbuf_data.frame_buffer);
                pbuf_data.frame_buffer = display_get_frame(uv->display_device);

                goto no_err;
error:
                std::cerr << "Connection lost, waiting for new connection" << std::endl;
                std::cerr << "Trying to accpet" << std::endl;

                ret = udt_receive_accept(uv->udt_receive);

no_err:
                ;
#endif
        }

        decoder_destroy(pbuf_data.decoder);

        return 0;
}



UGReceiver::UGReceiver(client_guiFrame * const p, const char *display, GLView *gl)
    : parent(p)
{
    pthread_t receiver_thread_id;

    uv = (struct state_uv *) malloc(sizeof(struct state_uv));
#ifndef USE_UDT
    uv->participants = pdb_init();
    uv->network_devices = initialize_network(strdup("localhost"), 5004, uv->participants);
#endif
    uv->connections_count = 1;
    gettimeofday(&uv->start_time, NULL);
    uv->decoder_mode = NULL;
    uv->postprocess = NULL;
    if(strcmp(display, "wxgl") == 0)
        uv->display_device = initialize_video_display(display, (char *) gl, 0 /*flags */);
    else
        uv->display_device = initialize_video_display(display, NULL, 0 /*flags */);

    if (pthread_create
        (&receiver_thread_id, NULL, receiver_thread,
         (void *)uv) != 0) {
            perror("Unable to create display thread!\n");
            // TODO handle error
    }
}
