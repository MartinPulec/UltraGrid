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
#include "vo_postprocess.h"

typedef void (*change_il_t)(char *dst, char *src, int linesize, int height);

static int parse_video_mode(const char *str);

static int parse_video_mode(const char *requested_mode) {
        if(requested_mode) {
                /* these are data comming from newtork ! */
                if(strcasecmp(requested_mode, "help") == 0) {
                        printf("Video mode options\n\n");
                        printf("-M {tiled-4K | 3D | dual-link }\n");
                        return VIDEO_UNSET;
                } else if(strcasecmp(requested_mode, "tiled-4K") == 0) {
                        return VIDEO_4K;
                } else if(strcasecmp(requested_mode, "3D") == 0) {
                        return VIDEO_STEREO;
                } else if(strcasecmp(requested_mode, "dual-link") == 0) {
                        return VIDEO_DUAL;
                } else {
                        fprintf(stderr, "[decoder] Unknown video mode (see -M help)\n");
                        return VIDEO_UNSET;
                }
        } else {
                return VIDEO_NORMAL; // default value
        }
}

struct state_receiver {
        struct video_display   *display;

        codec_t                *native_codecs;
        size_t                  native_count;

        struct video_frame     *ldgm_frame;
        void                   *ldgm_state;

        struct state_decompress *ext_decoder;
        bool                    ext_decoder_accepts_corrupted_frames;
        char                  **ext_recv_buffer[2];
        int                     ext_recv_buffer_index_network;
        pthread_mutex_t         lock;
        pthread_cond_t          boss_cv;
        pthread_cond_t          worker_cv;
        volatile bool           work_to_do;
        volatile bool           boss_waiting;
        volatile bool           worker_waiting;

        struct vo_postprocess_state *postprocess;
        struct video_frame *pp_frame;
        int pp_output_frames_count;

        enum interlacing_t     *disp_supported_il;
        size_t                  disp_supported_il_cnt;
        change_il_t             change_il;

        struct video_desc       display_desc;

        int                     video_mode;
};
static struct state_receiver *receiver_state_alloc(struct receiver_param *param);
static void receiver_state_destroy(struct state_receiver *receiver_state);
static struct state_receiver *receiver_state_alloc(struct receiver_param *param)
{
        struct state_receiver *ret = (struct state_receiver *)
                calloc(1, sizeof(struct state_receiver));

        ret->display = param->display;

        ret->ext_recv_buffer[0] = ret->ext_recv_buffer[1] = NULL;
        ret->ext_recv_buffer_index_network = 0;
        ret->ext_decoder = NULL;
        ret->ext_decoder_accepts_corrupted_frames = false;
        pthread_mutex_init(&ret->lock, NULL);
        pthread_cond_init(&ret->boss_cv, NULL);
        pthread_cond_init(&ret->worker_cv, NULL);
        ret->work_to_do = false;
        ret->boss_waiting = false;
        ret->worker_waiting = false;

        ret->native_codecs = NULL;
        ret->native_count = 0;

        ret->video_mode = parse_video_mode(param->decoder_mode);
        if(ret->video_mode == VIDEO_UNSET) {
                free(ret);
                return NULL;
        }

        ret->postprocess = NULL;
        ret->pp_frame = NULL;

        ret->disp_supported_il = NULL;
        ret->disp_supported_il_cnt = 0;
        ret->change_il = NULL;

        memset(&ret->display_desc, 0, sizeof(ret->display_desc));

        return ret;
}
static void receiver_state_destroy(struct state_receiver *receiver_state) {
        if(receiver_state) {
                pthread_mutex_destroy(&receiver_state->lock);
                pthread_cond_destroy(&receiver_state->boss_cv);
                pthread_cond_destroy(&receiver_state->worker_cv);
        }

        free(receiver_state);
}

static struct vcodec_state *new_decoder(struct receiver_param *uv, struct state_receiver *receiver_state) {
        struct vcodec_state *state = malloc(sizeof(struct vcodec_state));

        if(state) {
                state->decoder = decoder_init(uv->decoder_mode, uv->postprocess, uv->display_device);
                state->reconfigured = false;
                state->frame_buffer = NULL; // no frame until reconfiguration
                state->receiver_state = receiver_state;

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

static codec_t choose_codec_and_decoder(struct state_receiver * receiver_state, struct video_desc desc,
                                codec_t *in_codec, decoder_t *decode_line)
{
        codec_t out_codec = (codec_t) -1;
        *decode_line = NULL;
        *in_codec = desc.color_spec;
        bool decoder_found = false;
        
        /* first deal with aliases */
        if(*in_codec == DVS8 || *in_codec == Vuy2) {
                *in_codec = UYVY;
        }
        
        size_t native;
        /* first check if the codec is natively supported */
        for(native = 0u; native < receiver_state->native_count; ++native)
        {
                out_codec = receiver_state->native_codecs[native];
                if(out_codec == DVS8 || out_codec == Vuy2)
                        out_codec = UYVY;
                if(*in_codec == out_codec) {
                        if((out_codec == DXT1 || out_codec == DXT1_YUV ||
                                        out_codec == DXT5)
                                        && receiver_state->video_mode != VIDEO_NORMAL)
                                continue; /* it is a exception, see NOTES #1 */
                        if(*in_codec == RGBA || /* another exception - we may change shifts */
                                        *in_codec == RGB)
                                continue;
                        
                        *decode_line = (decoder_t) memcpy;
                        //receiver_state->decoder_type = LINE_DECODER;
                        decoder_found = true;
                        
                        goto after_linedecoder_lookup;
                }
        }
        /* otherwise if we have line decoder */
        int trans;
        for(trans = 0; line_decoders[trans].line_decoder != NULL;
                                ++trans) {
                
                for(native = 0; native < receiver_state->native_count; ++native)
                {
                        out_codec = receiver_state->native_codecs[native];
                        if(out_codec == DVS8 || out_codec == Vuy2)
                                out_codec = UYVY;
                        if(*in_codec == line_decoders[trans].from &&
                                        out_codec == line_decoders[trans].to) {
                                                
                                *decode_line = line_decoders[trans].line_decoder;
                                
                                decoder_found = true;
                                goto after_linedecoder_lookup;
                        }
                }
        }
        
after_linedecoder_lookup:

        /* we didn't find line decoder. So try now regular (aka DXT) decoder */
        if(*decode_line == NULL) {
                for(native = 0; native < receiver_state->native_count; ++native)
                {
                        int trans;
                        out_codec = receiver_state->native_codecs[native];
                        if(out_codec == DVS8 || out_codec == Vuy2)
                                out_codec = UYVY;
                                
                        for(trans = 0; trans < decoders_for_codec_count;
                                        ++trans) {
                                if(*in_codec == decoders_for_codec[trans].from &&
                                                out_codec == decoders_for_codec[trans].to) {
                                        receiver_state->ext_decoder = decompress_init(decoders_for_codec[trans].decompress_index);

                                        if(!receiver_state->ext_decoder) {
                                                debug_msg("Decompressor with magic %x was not found.\n");
                                                continue;
                                        }

                                        int res = 0, ret;
                                        size_t size = sizeof(res);
                                        ret = decompress_get_property(receiver_state->ext_decoder,
                                                        DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME,
                                                        &res,
                                                        &size);
                                        receiver_state->ext_decoder_accepts_corrupted_frames = (ret && res);

                                        decoder_found = true;

                                        goto after_decoder_lookup;
                                }
                        }
                }
        }
after_decoder_lookup:

        if(!decoder_found) {
                fprintf(stderr, "Unable to find decoder for input codec \"%s\"!!!\n", get_codec_name(desc.color_spec));
                exit_uv(128);
                return (codec_t) -1;
        }
        
        return out_codec;
}


void reconfigure_video(struct state_receiver *receiver_state,
                struct video_desc *desc) {
        codec_t out_codec, in_codec;
        decoder_t decode_line;
        enum interlacing_t display_il = PROGRESSIVE;
        //struct video_frame *frame;
        int render_mode;

        pthread_mutex_lock(&receiver_state->lock);
        {
                while (receiver_state->work_to_do) {
                        receiver_state->boss_waiting = TRUE;
                        pthread_cond_wait(&receiver_state->boss_cv, &receiver_state->lock);
                        receiver_state->boss_waiting = FALSE;
                }
        }
        pthread_mutex_unlock(&receiver_state->lock);

        assert(receiver_state->native_codecs != NULL);
        
        if(receiver_state->ext_decoder) {
                decompress_done(receiver_state->ext_decoder);
                receiver_state->ext_decoder = NULL;
        }
        for(int i = 0; i < 2; ++i) {
                if(receiver_state->ext_recv_buffer[i]) {
                        char **buf = receiver_state->ext_recv_buffer[i];
                        while(*buf != NULL) {
                                free(*buf);
                                buf++;
                        }
                        free(receiver_state->ext_recv_buffer[i]);
                        receiver_state->ext_recv_buffer[i] = NULL;
                }
        }

#if 0
        desc.tile_count = get_video_mode_tiles_x(receiver_state->video_mode)
                        * get_video_mode_tiles_y(receiver_state->video_mode);
#endif
        
        out_codec = choose_codec_and_decoder(receiver_state, *desc, &in_codec, &decode_line);
        if(out_codec == (codec_t) -1)
                return NULL;
        struct video_desc display_desc = *desc;

        int display_mode;
        size_t len = sizeof(int);
        int ret;

        ret = display_get_property(receiver_state->display, DISPLAY_PROPERTY_VIDEO_MODE,
                        &display_mode, &len);
        if(!ret) {
                debug_msg("Failed to get video display mode.");
                display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        }

        bool pp_does_change_tiling_mode = false;

        if (receiver_state->postprocess) {
                size_t len = sizeof(pp_does_change_tiling_mode);
                if(vo_postprocess_get_property(receiver_state->postprocess, VO_PP_DOES_CHANGE_TILING_MODE,
                                        &pp_does_change_tiling_mode, &len)) {
                        if(len == 0) {
                                // just for sake of completness since it shouldn't be a case
                                fprintf(stderr, "[Receiver] Warning: unable to get pp tiling mode!\n");
                        }
                }
        }

        if(receiver_state->postprocess) {
                struct video_desc pp_desc = *desc;
                pp_desc.color_spec = out_codec;
                if(!pp_does_change_tiling_mode) {
                        pp_desc.width *= get_video_mode_tiles_x(receiver_state->video_mode);
                        pp_desc.height *= get_video_mode_tiles_y(receiver_state->video_mode);
                        pp_desc.tile_count = 1;
                }
                vo_postprocess_reconfigure(receiver_state->postprocess, pp_desc);
                receiver_state->pp_frame = vo_postprocess_getf(receiver_state->postprocess);
                vo_postprocess_get_out_desc(receiver_state->postprocess, &display_desc,
                                &render_mode, &receiver_state->pp_output_frames_count);
        }
        
        if(!is_codec_opaque(out_codec)) {
                receiver_state->change_il = select_il_func(desc->interlacing, receiver_state->disp_supported_il, receiver_state->disp_supported_il_cnt, &display_il);
        } else {
                receiver_state->change_il = NULL;
        }

        if (!receiver_state->postprocess || !pp_does_change_tiling_mode) { /* otherwise we need postprocessor mode, which we obtained before */
                render_mode = display_mode;
        }

        display_desc.color_spec = out_codec;
        display_desc.interlacing = display_il;

        if(!video_desc_eq(receiver_state->display_desc, display_desc))
        {
                int ret;
                /*
                 * TODO: put frame should be definitely here. On the other hand, we cannot be sure
                 * that vo driver is initialized so far:(
                 */
                //display_put_frame(decoder->display, frame);
                /* reconfigure VO and give it opportunity to pass us pitch */        
                ret = display_reconfigure(receiver_state->display, display_desc);
                if(!ret) {
                        fprintf(stderr, "[decoder] Unable to reconfigure display.\n");
                        exit_uv(128);
                        return NULL;
                }
                frame_display = display_get_frame(receiver_state->display);
                decoder->display_desc = display_desc;
        }
        /*if(decoder->postprocess) {
                frame = decoder->pp_frame;
        } else {
                frame = frame_display;
        }*/
        
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_RSHIFT,
                        &decoder->rshift, &len);
        if(!ret) {
                debug_msg("Failed to get rshift property from video driver.\n");
                decoder->rshift = 0;
        }
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_GSHIFT,
                        &decoder->gshift, &len);
        if(!ret) {
                debug_msg("Failed to get gshift property from video driver.\n");
                decoder->gshift = 8;
        }
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_BSHIFT,
                        &decoder->bshift, &len);
        if(!ret) {
                debug_msg("Failed to get bshift property from video driver.\n");
                decoder->bshift = 16;
        }
        
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_BUF_PITCH,
                        &decoder->requested_pitch, &len);
        if(!ret) {
                debug_msg("Failed to get pitch from video driver.\n");
                decoder->requested_pitch = PITCH_DEFAULT;
        }
        
        int linewidth;
        if(render_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
                linewidth = desc.width; 
        } else {
                linewidth = desc.width * get_video_mode_tiles_x(decoder->video_mode);
        }

        if(!decoder->postprocess) {
                if(decoder->requested_pitch == PITCH_DEFAULT)
                        decoder->pitch = vc_get_linesize(linewidth, out_codec);
                else
                        decoder->pitch = decoder->requested_pitch;
        } else {
                decoder->pitch = vc_get_linesize(linewidth, out_codec);
        }

        if(decoder->requested_pitch == PITCH_DEFAULT) {
                decoder->display_pitch = vc_get_linesize(display_desc.width, out_codec);
        } else {
                decoder->display_pitch = decoder->requested_pitch;
        }

        int src_x_tiles = get_video_mode_tiles_x(decoder->video_mode);
        int src_y_tiles = get_video_mode_tiles_y(decoder->video_mode);

        if(decoder->decoder_type == LINE_DECODER) {
                decoder->line_decoder = malloc(src_x_tiles * src_y_tiles *
                                        sizeof(struct line_decoder));                
                if(render_mode == DISPLAY_PROPERTY_VIDEO_MERGED && decoder->video_mode == VIDEO_NORMAL) {
                        struct line_decoder *out = &decoder->line_decoder[0];
                        out->base_offset = 0;
                        out->src_bpp = get_bpp(in_codec);
                        out->dst_bpp = get_bpp(out_codec);
                        out->rshift = decoder->rshift;
                        out->gshift = decoder->gshift;
                        out->bshift = decoder->bshift;
                
                        out->decode_line = decode_line;
                        out->dst_pitch = decoder->pitch;
                        out->src_linesize = vc_get_linesize(desc.width, in_codec);
                        out->dst_linesize = vc_get_linesize(desc.width, out_codec);
                        decoder->merged_fb = TRUE;
                } else if(render_mode == DISPLAY_PROPERTY_VIDEO_MERGED
                                && decoder->video_mode != VIDEO_NORMAL) {
                        int x, y;
                        for(x = 0; x < src_x_tiles; ++x) {
                                for(y = 0; y < src_y_tiles; ++y) {
                                        struct line_decoder *out = &decoder->line_decoder[x + 
                                                        src_x_tiles * y];
                                        out->base_offset = y * (desc.height)
                                                        * decoder->pitch + 
                                                        vc_get_linesize(x * desc.width, out_codec);

                                        out->src_bpp = get_bpp(in_codec);
                                        out->dst_bpp = get_bpp(out_codec);

                                        out->rshift = decoder->rshift;
                                        out->gshift = decoder->gshift;
                                        out->bshift = decoder->bshift;
                
                                        out->decode_line = decode_line;

                                        out->dst_pitch = decoder->pitch;
                                        out->src_linesize =
                                                vc_get_linesize(desc.width, in_codec);
                                        out->dst_linesize =
                                                vc_get_linesize(desc.width, out_codec);
                                }
                        }
                        decoder->merged_fb = TRUE;
                } else if (render_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
                        int x, y;
                        for(x = 0; x < src_x_tiles; ++x) {
                                for(y = 0; y < src_y_tiles; ++y) {
                                        struct line_decoder *out = &decoder->line_decoder[x + 
                                                        src_x_tiles * y];
                                        out->base_offset = 0;
                                        out->src_bpp = get_bpp(in_codec);
                                        out->dst_bpp = get_bpp(out_codec);
                                        out->rshift = decoder->rshift;
                                        out->gshift = decoder->gshift;
                                        out->bshift = decoder->bshift;
                
                                        out->decode_line = decode_line;
                                        out->src_linesize =
                                                vc_get_linesize(desc.width, in_codec);
                                        out->dst_pitch = 
                                                out->dst_linesize =
                                                vc_get_linesize(desc.width, out_codec);
                                }
                        }
                        decoder->merged_fb = FALSE;
                }
        } else if (decoder->decoder_type == EXTERNAL_DECODER) {
                int buf_size;
                
                buf_size = decompress_reconfigure(decoder->ext_decoder, desc, 
                                decoder->rshift, decoder->gshift, decoder->bshift, decoder->pitch , out_codec);
                if(!buf_size) {
                        return NULL;
                }
                for(int i = 0; i < 2; ++i) {
                        int j;
                        decoder->ext_recv_buffer[i] = malloc((src_x_tiles * src_y_tiles + 1) * sizeof(char *));
                        for (j = 0; j < src_x_tiles * src_y_tiles; ++j)
                                decoder->ext_recv_buffer[i][j] = malloc(buf_size);
                        decoder->ext_recv_buffer[i][j] = NULL;
                }
                decoder->ext_recv_buffer_index_network = 0;
                if(render_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
                        decoder->merged_fb = FALSE;
                } else {
                        decoder->merged_fb = TRUE;
                }
        }
        
        return frame_display;
}



}

void update_decoder_state(struct vcodec_state *original_vcodec_state,
                struct video_desc *video_desc, 
                struct ldgm_desc *ldgm_desc,
                int max_substreams
                )
{
        struct state_receiver *receiver_state = original_vcodec_state->receiver_state;

        if(max_substreams >= decoder->max_substreams) {
                fprintf(stderr, "[decoder] received substream ID %d. Expecting at most %d substreams. Did you
                                set -M option?\n",
                                max_substream, decoder->max_substream);
                if(max_substreams == 1 || max_substreams == 3) {
                        fprintf(stderr, "[decoder] Guessing mode: ");
                        if(max_substreams == 1) {
                                decoder_set_video_mode(decoder, VIDEO_STEREO);
                        } else {
                                decoder_set_video_mode(decoder, VIDEO_4K);
                        }
                        fprintf(stderr, "%s\n", get_video_mode_description(decoder->video_mode));
                } else {
                        exit_uv(1);
                }
        }

        if(video_desc) {
                reconfigure_video(receiver_state, video_desc);
        } else if (ldgm_desc) {
                if(receiver_state->ldgm_state) {
                        ldgm_decoder_destroy(receiver_state->ldgm_state);
                        receiver_state->ldgm_state = NULL;
                        vf_free_data(receiver_state->ldgm_frame);
                        receiver_state->ldgm_frame = NULL;
                }
                receiver_state->ldgm_state = ldgm_decoder_init(ldgm_desc->k,
                                ldgm_desc->m, ldgm_desc->c, ldgm_desc->seed);
                original_vcodec_state->line_decoder.base_offset = 0;
                original_vcodec_state->line_decoder.src_bpp = 1.0;
                original_vcodec_state->line_decoder.dst_bpp = 1.0;
                /// {r,g,b}shift unused
                original_vcodec_state->line_decoder.decode_line = NULL;

                struct video_desc ldgm_desc;
                memset(&ldgm_desc, 0, sizeof(ldgm_desc));
                ldgm_desc.tile_count = max_substreams;
                // data will be allocated by decoder itself according to size
                receiver_state->ldgm_frame = vf_alloc_desc(ldgm_desc);
                original_vcodec_state->frame_buffer = receiver_state->ldgm_frame;



                        //a jeste rict, ze se bude potreba zreinicializovat prilezitostne
        }
}

void *receiver_thread(void *arg)
{
        struct receiver_param *uv = (struct receiver_param *)arg;
        struct state_receiver *receiver_state;
        receiver_state = receiver_state_alloc(uv);

        struct pdb_e *cp;
        struct timeval timeout;
        struct timeval start_time;
        struct timeval curr_time;
        uint32_t ts;
        int fr;
        int ret;
        unsigned int tiles_post = 0;
        struct timeval last_tile_received = {0, 0};
        int last_buf_size = INITIAL_VIDEO_RECV_BUFFER_SIZE;
#ifdef SHARED_DECODER
        struct vcodec_state *shared_decoder = new_decoder(uv, receiver_state);
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
                                cp->video_decoder_state = new_decoder(uv, &receiver_state);
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

        receiver_state_destroy(receiver_state);

        return 0;
}

