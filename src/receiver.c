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

#include "debug.h"
#include "host.h"
#include "pdb.h"
#include "receiver.h"
#include "rtp/decoders.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "tfrc.h"
#include "tv.h"
#include "video_decompress.h"
#include "video_display.h"

static struct vcodec_state *new_decoder(struct state_receiver *uv) {
        struct vcodec_state *state = malloc(sizeof(struct vcodec_state));

        if(state) {
                state->decoder = decoder_init(uv->decoder_mode, uv->postprocess, uv->display_device);
                state->reconfigured = false;
                state->frame_buffer = NULL; // no frame until reconfiguration

                if(!state->decoder) {
                        fprintf(stderr, "Error initializing decoder (incorrect '-M' or '-p' option).\n");
                        free(state);
                        exit_uv(1);
                        return NULL;
                } else {
                        //decoder_register_video_display(state->decoder, uv->display_device);
                }
        }

        return state;
}

void destroy_decoder(struct vcodec_state *video_decoder_state) {
        if(!video_decoder_state) {
                return;
        }

        decoder_destroy(video_decoder_state->decoder);

        free(video_decoder_state);
}

void *receiver_thread(void *arg)
{
        struct state_receiver *uv = (struct state_receiver *)arg;

        struct pdb_e *cp;
        struct timeval timeout;
        struct timeval curr_time;
        struct timeval start_time;
        uint32_t ts;
        int fr;
        int ret;
        unsigned int tiles_post = 0;
        struct timeval last_tile_received = {0, 0};
        int last_buf_size = INITIAL_VIDEO_RECV_BUFFER_SIZE;
#ifdef SHARED_DECODER
        struct vcodec_state *shared_decoder = new_decoder(uv);
        if(shared_decoder == NULL) {
                fprintf(stderr, "Unable to create decoder!\n");
                exit_uv(1);
                return NULL;
        }
#endif // SHARED_DECODER


        initialize_video_decompress();

        pthread_mutex_unlock(uv->master_lock);

        fr = 1;

        gettimeofday(&start_time, NULL);

        while (!should_exit_receiver) {
                /* Housekeeping and RTCP... */
                gettimeofday(&curr_time, NULL);
                ts = tv_diff(curr_time, start_time) * 90000;
                rtp_update(uv->network_devices[0], curr_time);
                rtp_send_ctrl(uv->network_devices[0], ts, 0, curr_time);

                /* Receive packets from the network... The timeout is adjusted */
                /* to match the video capture rate, so the transmitter works.  */
                if (fr) {
                        gettimeofday(&curr_time, NULL);
                        fr = 0;
                }

                timeout.tv_sec = 0;
                timeout.tv_usec = 999999 / 59.94;
                ret = rtp_recv_poll_r(uv->network_devices, &timeout, ts);

                /*
                   if (ret == FALSE) {
                   printf("Failed to receive data\n");
                   }
                 */
                UNUSED(ret);

                /* Decode and render for each participant in the conference... */
                cp = pdb_iter_init(uv->participants);
                while (cp != NULL) {
                        if (tfrc_feedback_is_due(cp->tfrc_state, curr_time)) {
                                debug_msg("tfrc rate %f\n",
                                          tfrc_feedback_txrate(cp->tfrc_state,
                                                               curr_time));
                        }

                        if(cp->video_decoder_state == NULL) {
#ifdef SHARED_DECODER
                                cp->video_decoder_state = shared_decoder;
#else
                                cp->video_decoder_state = new_decoder(uv);
#endif // SHARED_DECODER
                                if(cp->video_decoder_state == NULL) {
                                        fprintf(stderr, "Fatal: unable to find decoder state for "
                                                        "participant %u.\n", cp->ssrc);
                                        exit_uv(1);
                                        break;
                                }
                                cp->video_decoder_state->display = uv->display_device;
                        }

                        /* Decode and render video... */
                        if (pbuf_decode
                            (cp->playout_buffer, curr_time, decode_frame, cp->video_decoder_state)) {
                                tiles_post++;
                                /* we have data from all connections we need */
                                if(tiles_post == uv->connections_count) 
                                {
                                        tiles_post = 0;
                                        gettimeofday(&curr_time, NULL);
                                        fr = 1;
#if 0
                                        display_put_frame(uv->display_device,
                                                          cp->video_decoder_state->frame_buffer);
                                        cp->video_decoder_state->frame_buffer =
                                            display_get_frame(uv->display_device);
#endif
                                }
                                last_tile_received = curr_time;
                        }

                        /* dual-link TIMEOUT - we won't wait for next tiles */
                        if(tiles_post > 1 && tv_diff(curr_time, last_tile_received) > 
                                        999999 / 59.94 / uv->connections_count) {
                                tiles_post = 0;
                                gettimeofday(&curr_time, NULL);
                                fr = 1;
#if 0
                                display_put_frame(uv->display_device,
                                                cp->video_decoder_state->frame_buffer);
                                cp->video_decoder_state->frame_buffer =
                                        display_get_frame(uv->display_device);
#endif
                                last_tile_received = curr_time;
                        }

                        if(cp->video_decoder_state->decoded % 100 == 99) {
                                int new_size = cp->video_decoder_state->max_frame_size * 110ull / 100;
                                if(new_size >= last_buf_size) {
                                        struct rtp **device = uv->network_devices;
                                        while(*device) {
                                                int ret = rtp_set_recv_buf(*device, new_size);
                                                if(!ret) {
                                                        recv_buf_increase_warning(new_size);
                                                }
                                                debug_msg("Recv buffer adjusted to %d\n", new_size);
                                                device++;
                                        }
                                        last_buf_size = new_size;
                                }
                        }

                        if(cp->video_decoder_state->reconfigured) {
                                struct rtp **session = uv->network_devices;
                                while(*session) {
                                        rtp_flush_recv_buf(*session);
                                        ++session;
                                }
                                cp->video_decoder_state->reconfigured = false;
                        }

                        pbuf_remove(cp->playout_buffer, curr_time);
                        cp = pdb_iter_next(uv->participants);
                }
                pdb_iter_done(uv->participants);
        }
        
#ifdef SHARED_DECODER
        destroy_decoder(shared_decoder);
#endif //  SHARED_DECODER

        display_finish(uv->display_device);

        return 0;
}

