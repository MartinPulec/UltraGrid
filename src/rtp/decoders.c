/*
 * AUTHOR:   Ladan Gharai/Colin Perkins
 * 
 * Copyright (c) 2003-2004 University of Southern California
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
 *      California Information Sciences Institute.
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
#include "perf.h"
#include "rtp/ldgm.h"
#include "rtp/ll.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/decoders.h"
#include "video.h"
#include "video_codec.h"
#include "video_decompress.h"
#include "video_display.h"
#include "vo_postprocess.h"

#include <pthread.h>

struct state_decoder;

static struct video_frame * reconfigure_decoder(struct state_decoder * const decoder, struct video_desc desc,
                                struct video_frame *frame);
typedef void (*change_il_t)(char *dst, char *src, int linesize, int height);
static void restrict_returned_codecs(codec_t *display_codecs,
                size_t *display_codecs_count, codec_t *pp_codecs,
                int pp_codecs_count);
static void decoder_set_video_mode(struct state_decoder *decoder, unsigned int video_mode);
static int check_for_mode_change(struct state_decoder *decoder, uint32_t *hdr, int packet_type);

enum decoder_type_t {
        UNSET,
        LINE_DECODER,
        EXTERNAL_DECODER
};

struct fec {
        int               k, m, c;
        int               seed;
        void             *state;
};

struct extra_decompress_data;

struct state_decoder {
        pthread_t decompress_thread_id;
        struct video_desc received_vid_desc;
        struct ldgm_desc received_ldgm_desc;
        struct video_desc display_desc;

        struct video_frame *frame;
        
        /* requested values */
        int               requested_pitch;
        int               rshift, gshift, bshift;
        unsigned int      max_substreams;
        
        struct display   *display;
        codec_t          *native_codecs;
        size_t            native_count;
        enum interlacing_t    *disp_supported_il;
        size_t            disp_supported_il_cnt;
        change_il_t       change_il;
        
        /* actual values */
        enum decoder_type_t decoder_type; 
        struct {
                struct line_decoder *line_decoder;
                struct {                           /* OR - it is not union for easier freeing*/
                        struct state_decompress *ext_decoder;
                        unsigned int accepts_corrupted_frame:1;
                        char **ext_recv_buffer[2];
                        int ext_recv_buffer_index_network;
                        pthread_mutex_t lock;
                        pthread_cond_t boss_cv;
                        pthread_cond_t worker_cv;
                        volatile bool work_to_do;
                        volatile bool boss_waiting;
                        volatile bool worker_waiting;
                        volatile bool thread_has_processed_input;
                        struct extra_decompress_data *extra_decompress_data;
                };
        };
        codec_t           out_codec;
        // display or postprocessor
        int               pitch;
        // display pitch, can differ from previous if we have postprocessor
        int               display_pitch;
        
        struct {
                struct vo_postprocess_state *postprocess;
                struct video_frame *pp_frame;
                int pp_output_frames_count;
        };

        unsigned int      video_mode;
        
        unsigned          merged_fb:1;

        struct fec        fec_state;

        // for statistics
        unsigned long int   displayed;
        unsigned long int   corrupted;
};

struct extra_decompress_data {
        char **ext_recv_buffer;
        int *buffer_len;
        char **fec_buffers;
};

void *decompress_thread(void *args);
static volatile bool should_exit_decompress_thread = false;

void *decompress_thread(void *args) {
        struct state_decoder *decoder = args;

        struct tile *tile;

        while(!should_exit_decompress_thread) {
                pthread_mutex_lock(&decoder->lock);
                {
                        while (decoder->work_to_do == false && !should_exit_decompress_thread) {
                                decoder->worker_waiting = true;
                                pthread_cond_wait(&decoder->worker_cv, &decoder->lock);
                                decoder->worker_waiting = false;
                        }
                        // we have double buffering so we can signal immediatelly....
                        if(decoder->decoder_type == EXTERNAL_DECODER) {
                                decoder->thread_has_processed_input = true;
                                if(decoder->boss_waiting) {
                                        pthread_cond_signal(&decoder->boss_cv);
                                }
                        }
                }
                pthread_mutex_unlock(&decoder->lock);

                if(should_exit_decompress_thread)
                        break;


                if(decoder->decoder_type == EXTERNAL_DECODER) {
                        int tile_width = decoder->received_vid_desc.width; // get_video_mode_tiles_x(decoder->video_mode);
                        int tile_height = decoder->received_vid_desc.height; // get_video_mode_tiles_y(decoder->video_mode);
                        int x, y;
                        struct video_frame *output;
                        if(decoder->postprocess) {
                                output = decoder->pp_frame;
                        } else {
                                output = decoder->frame;
                        }
                        for (x = 0; x < get_video_mode_tiles_x(decoder->video_mode); ++x) {
                                for (y = 0; y < get_video_mode_tiles_y(decoder->video_mode); ++y) {
                                        int pos = x + get_video_mode_tiles_x(decoder->video_mode) * y;
                                        char *out;
                                        if(decoder->merged_fb) {
                                                tile = vf_get_tile(output, 0);
                                                // TODO: OK when rendering directly to display FB, otherwise, do not reflect pitch (we use PP)
                                                out = tile->data + y * decoder->pitch * tile_height +
                                                        vc_get_linesize(tile_width, decoder->out_codec) * x;
                                        } else {
                                                tile = vf_get_tile(output, x);
                                                out = tile->data;
                                        }
                                        decompress_frame(decoder->ext_decoder,
                                                        (unsigned char *) out,
                                                        (unsigned char *) decoder->extra_decompress_data->ext_recv_buffer[pos],
                                                        decoder->extra_decompress_data->buffer_len[pos]);
                                }
                        }
                }

                if(decoder->postprocess) {
                        int i;
                        bool pp_ret;

                        pp_ret = vo_postprocess(decoder->postprocess,
                                        decoder->pp_frame,
                                        decoder->frame,
                                        decoder->display_pitch);

                        if(!pp_ret) {
                                decoder->pp_frame = vo_postprocess_getf(decoder->postprocess);
                                goto skip_frame;
                        }

                        for (i = 1; i < decoder->pp_output_frames_count; ++i) {
                                display_put_frame(decoder->display, decoder->frame);
                                decoder->frame = display_get_frame(decoder->display);
                                pp_ret = vo_postprocess(decoder->postprocess,
                                                NULL,
                                                decoder->frame,
                                                decoder->display_pitch);
                                if(!pp_ret) {
                                        decoder->pp_frame = vo_postprocess_getf(decoder->postprocess);
                                        goto skip_frame;
                                }
                        }

                        /* get new postprocess frame */
                        decoder->pp_frame = vo_postprocess_getf(decoder->postprocess);
                }

                if(decoder->change_il) {
                        unsigned int i;
                        for(i = 0; i < decoder->frame->tile_count; ++i) {
                                struct tile *tile = vf_get_tile(decoder->frame, i);
                                decoder->change_il(tile->data, tile->data, vc_get_linesize(tile->width, decoder->out_codec), tile->height);
                        }
                }

                display_put_frame(decoder->display,
                                decoder->frame);
                decoder->frame =
                        display_get_frame(decoder->display);

skip_frame:

                for(unsigned int i = 0; i < decoder->max_substreams; ++i) {
                        free(decoder->extra_decompress_data->fec_buffers[i]);
                }
                free(decoder->extra_decompress_data->fec_buffers);

                free(decoder->extra_decompress_data->ext_recv_buffer);
                free(decoder->extra_decompress_data->buffer_len);
                free(decoder->extra_decompress_data);

                pthread_mutex_lock(&decoder->lock);
                {
                        decoder->work_to_do = false;

                        if (decoder->boss_waiting) {
                                pthread_cond_signal(&decoder->boss_cv);
                        }

                        // we have double buffering so we can signal immediatelly....
                        if(decoder->decoder_type != EXTERNAL_DECODER) {
                                decoder->thread_has_processed_input = true;
                                if(decoder->boss_waiting) {
                                        pthread_cond_signal(&decoder->boss_cv);
                                }
                        }
                }
                pthread_mutex_unlock(&decoder->lock);
        }

        return NULL;
}

static void decoder_set_video_mode(struct state_decoder *decoder, unsigned int video_mode)
{
        decoder->video_mode = video_mode;
        decoder->max_substreams = get_video_mode_tiles_x(decoder->video_mode)
                        * get_video_mode_tiles_y(decoder->video_mode);
}

struct state_decoder *decoder_init(char *requested_mode, char *postprocess, struct display *display)
{
        struct state_decoder *s;
        
        s = (struct state_decoder *) calloc(1, sizeof(struct state_decoder));
        s->native_codecs = NULL;
        s->disp_supported_il = NULL;
        s->postprocess = NULL;
        s->change_il = NULL;
        s->display = display;

        s->fec_state.state = NULL;
        s->fec_state.k = s->fec_state.m = s->fec_state.c = s->fec_state.seed = 0;

        s->displayed = s->corrupted = 0ul;
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->boss_cv, NULL);
        pthread_cond_init(&s->worker_cv, NULL);
        s->work_to_do = false;
        s->boss_waiting = false;
        s->worker_waiting = false;
        
        int video_mode = VIDEO_NORMAL;

        if(requested_mode) {
                /* these are data comming from newtork ! */
                if(strcasecmp(requested_mode, "help") == 0) {
                        printf("Video mode options\n\n");
                        printf("-M {tiled-4K | 3D | dual-link }\n");
                        free(s);
                        exit_uv(129);
                        return NULL;
                } else if(strcasecmp(requested_mode, "tiled-4K") == 0) {
                        video_mode = VIDEO_4K;
                } else if(strcasecmp(requested_mode, "3D") == 0) {
                        video_mode = VIDEO_STEREO;
                } else if(strcasecmp(requested_mode, "dual-link") == 0) {
                        video_mode = VIDEO_DUAL;
                } else {
                        fprintf(stderr, "[decoder] Unknown video mode (see -M help)\n");
                        free(s);
                        exit_uv(129);
                        return NULL;
                }
        }

        decoder_set_video_mode(s, video_mode);

        if(postprocess) {
                s->postprocess = vo_postprocess_init(postprocess);
                if(strcmp(postprocess, "help") == 0) {
                        exit_uv(0);
                        return NULL;
                }
                if(!s->postprocess) {
                        fprintf(stderr, "Initializing postprocessor \"%s\" failed.\n", postprocess);
                        free(s);
                        exit_uv(129);
                        return NULL;
                }
        }

        decoder_register_video_display(s, display);

        if(pthread_create(&s->decompress_thread_id, NULL, decompress_thread, s) != 0) {
                perror("Unable to create thread");
                return NULL;
        }
        
        return s;
}

static void restrict_returned_codecs(codec_t *display_codecs,
                size_t *display_codecs_count, codec_t *pp_codecs,
                int pp_codecs_count)
{
        int i;

        for (i = 0; i < (int) *display_codecs_count; ++i) {
                int j;

                int found = FALSE;

                for (j = 0; j < pp_codecs_count; ++j) {
                        if(display_codecs[i] == pp_codecs[j]) {
                                found = TRUE;
                        }
                }

                if(!found) {
                        memmove(&display_codecs[i], (const void *) &display_codecs[i + 1],
                                        sizeof(codec_t) * (*display_codecs_count - i - 1));
                        --*display_codecs_count;
                        --i;
                }
        }
}

void decoder_register_video_display(struct state_decoder *decoder, struct display *display)
{
        int ret, i;
        decoder->display = display;
        
        free(decoder->native_codecs);
        decoder->native_count = 20 * sizeof(codec_t);
        decoder->native_codecs = (codec_t *)
                malloc(decoder->native_count * sizeof(codec_t));
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_CODECS, decoder->native_codecs, &decoder->native_count);
        decoder->native_count /= sizeof(codec_t);
        if(!ret) {
                fprintf(stderr, "Failed to query codecs from video display.\n");
                exit_uv(129);
                return;
        }
        
        /* next check if we didn't receive alias for UYVY */
        for(i = 0; i < (int) decoder->native_count; ++i) {
                assert(decoder->native_codecs[i] != Vuy2 &&
                                decoder->native_codecs[i] != DVS8);
        }

        if(decoder->postprocess) {
                codec_t postprocess_supported_codecs[20];
                size_t len = sizeof(postprocess_supported_codecs);
                bool ret = vo_postprocess_get_property(decoder->postprocess, VO_PP_PROPERTY_CODECS,
                                (void *) postprocess_supported_codecs, &len);

                if(ret) {
                        if(len == 0) { // problem detected
                                fprintf(stderr, "[Decoder] Unable to get supported codecs.\n");
                                exit_uv(1);
                                return;

                        }
                        restrict_returned_codecs(decoder->native_codecs, &decoder->native_count,
                                        postprocess_supported_codecs, len / sizeof(codec_t));
                }
        }


        free(decoder->disp_supported_il);
        decoder->disp_supported_il_cnt = 20 * sizeof(enum interlacing_t);
        decoder->disp_supported_il = malloc(decoder->disp_supported_il_cnt);
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_SUPPORTED_IL_MODES, decoder->disp_supported_il, &decoder->disp_supported_il_cnt);
        if(ret) {
                decoder->disp_supported_il_cnt /= sizeof(enum interlacing_t);
        } else {
                enum interlacing_t tmp[] = { PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME}; /* default if not said othervise */
                memcpy(decoder->disp_supported_il, tmp, sizeof(tmp));
                decoder->disp_supported_il_cnt = sizeof(tmp) / sizeof(enum interlacing_t);
        }
}

void decoder_destroy(struct state_decoder *decoder)
{
        if(!decoder)
                return;

        pthread_mutex_lock(&decoder->lock);
        {
                should_exit_decompress_thread = true;
                if(decoder->worker_waiting)
                        pthread_cond_signal(&decoder->worker_cv);
        }
        pthread_mutex_unlock(&decoder->lock);

        pthread_join(decoder->decompress_thread_id, NULL);

        pthread_mutex_destroy(&decoder->lock);
        pthread_cond_destroy(&decoder->boss_cv);
        pthread_cond_destroy(&decoder->worker_cv);

        if(decoder->ext_decoder) {
                decompress_done(decoder->ext_decoder);
                decoder->ext_decoder = NULL;
        }
        for (int i = 0; i < 2; ++i) {
                if(decoder->ext_recv_buffer[i]) {
                        char **buf = decoder->ext_recv_buffer[i];
                        while(*buf != NULL) {
                                free(*buf);
                                buf++;
                        }
                        free(decoder->ext_recv_buffer[i]);
                        decoder->ext_recv_buffer[i] = NULL;
                }
        }
        if(decoder->pp_frame) {
                vo_postprocess_done(decoder->postprocess);
                decoder->pp_frame = NULL;
        }
        if(decoder->line_decoder) {
                free(decoder->line_decoder);
        }
        free(decoder->native_codecs);
        free(decoder->disp_supported_il);

        if(decoder->fec_state.state)
                ldgm_decoder_destroy(decoder->fec_state.state);

        fprintf(stderr, "Decoder statistics: %lu displayed frames / %lu frames corrupted\n",
                        decoder->displayed, decoder->corrupted);
        free(decoder);
}

static codec_t choose_codec_and_decoder(struct state_decoder * const decoder, struct video_desc desc,
                                codec_t *in_codec, decoder_t *decode_line)
{
        codec_t out_codec = (codec_t) -1;
        *decode_line = NULL;
        *in_codec = desc.color_spec;
        
        /* first deal with aliases */
        if(*in_codec == DVS8 || *in_codec == Vuy2) {
                *in_codec = UYVY;
        }
        
        size_t native;
        /* first check if the codec is natively supported */
        for(native = 0u; native < decoder->native_count; ++native)
        {
                out_codec = decoder->native_codecs[native];
                if(out_codec == DVS8 || out_codec == Vuy2)
                        out_codec = UYVY;
                if(*in_codec == out_codec) {
                        if((out_codec == DXT1 || out_codec == DXT1_YUV ||
                                        out_codec == DXT5)
                                        && decoder->video_mode != VIDEO_NORMAL)
                                continue; /* it is a exception, see NOTES #1 */
                        if(*in_codec == RGBA || /* another exception - we may change shifts */
                                        *in_codec == RGB)
                                continue;
                        
                        *decode_line = (decoder_t) memcpy;
                        decoder->decoder_type = LINE_DECODER;
                        
                        goto after_linedecoder_lookup;
                }
        }
        /* otherwise if we have line decoder */
        int trans;
        for(trans = 0; line_decoders[trans].line_decoder != NULL;
                                ++trans) {
                
                for(native = 0; native < decoder->native_count; ++native)
                {
                        out_codec = decoder->native_codecs[native];
                        if(out_codec == DVS8 || out_codec == Vuy2)
                                out_codec = UYVY;
                        if(*in_codec == line_decoders[trans].from &&
                                        out_codec == line_decoders[trans].to) {
                                                
                                *decode_line = line_decoders[trans].line_decoder;
                                
                                decoder->decoder_type = LINE_DECODER;
                                goto after_linedecoder_lookup;
                        }
                }
        }
        
after_linedecoder_lookup:

        /* we didn't find line decoder. So try now regular (aka DXT) decoder */
        if(*decode_line == NULL) {
                for(native = 0; native < decoder->native_count; ++native)
                {
                        int trans;
                        out_codec = decoder->native_codecs[native];
                        if(out_codec == DVS8 || out_codec == Vuy2)
                                out_codec = UYVY;
                                
                        for(trans = 0; trans < decoders_for_codec_count;
                                        ++trans) {
                                if(*in_codec == decoders_for_codec[trans].from &&
                                                out_codec == decoders_for_codec[trans].to) {
                                        decoder->ext_decoder = decompress_init(decoders_for_codec[trans].decompress_index);

                                        if(!decoder->ext_decoder) {
                                                debug_msg("Decompressor with magic %x was not found.\n");
                                                continue;
                                        }

                                        int res = 0, ret;
                                        size_t size = sizeof(res);
                                        ret = decompress_get_property(decoder->ext_decoder,
                                                        DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME,
                                                        &res,
                                                        &size);
                                        if(ret && res) {
                                                decoder->accepts_corrupted_frame = TRUE;
                                        } else {
                                                decoder->accepts_corrupted_frame = FALSE;
                                        }

                                        decoder->decoder_type = EXTERNAL_DECODER;

                                        goto after_decoder_lookup;
                                }
                        }
                }
        }
after_decoder_lookup:

        if(decoder->decoder_type == UNSET) {
                fprintf(stderr, "Unable to find decoder for input codec \"%s\"!!!\n", get_codec_name(desc.color_spec));
                exit_uv(128);
                return (codec_t) -1;
        }
        
        decoder->out_codec = out_codec;
        return out_codec;
}

static change_il_t select_il_func(enum interlacing_t in_il, enum interlacing_t *supported, int il_out_cnt, /*out*/ enum interlacing_t *out_il)
{
        struct transcode_t { enum interlacing_t in; enum interlacing_t out; change_il_t func; };

        struct transcode_t transcode[] = {
                {UPPER_FIELD_FIRST, INTERLACED_MERGED, il_upper_to_merged},
                {INTERLACED_MERGED, UPPER_FIELD_FIRST, il_merged_to_upper}
        };

        int i;
        /* first try to check if it can be nativelly displayed */
        for (i = 0; i < il_out_cnt; ++i) {
                if(in_il == supported[i]) {
                        *out_il = in_il;
                        return NULL;
                }
        }

        for (i = 0; i < il_out_cnt; ++i) {
                size_t j;
                for (j = 0; j < sizeof(transcode) / sizeof(struct transcode_t); ++j) {
                        if(in_il == transcode[j].in && supported[i] == transcode[j].out) {
                                *out_il = transcode[j].out;
                                return transcode[j].func;
                        }
                }
        }

        return NULL;
}

static struct video_frame * reconfigure_decoder(struct state_decoder * const decoder,
                struct video_desc desc, struct video_frame *frame_display)
{
        codec_t out_codec, in_codec;
        decoder_t decode_line;
        enum interlacing_t display_il = PROGRESSIVE;
        //struct video_frame *frame;
        int render_mode;

        pthread_mutex_lock(&decoder->lock);
        {
                while (decoder->work_to_do) {
                        decoder->boss_waiting = TRUE;
                        pthread_cond_wait(&decoder->boss_cv, &decoder->lock);
                        decoder->boss_waiting = FALSE;
                }
        }
        pthread_mutex_unlock(&decoder->lock);

        assert(decoder != NULL);
        assert(decoder->native_codecs != NULL);
        
        free(decoder->line_decoder);
        decoder->line_decoder = NULL;
        decoder->decoder_type = UNSET;
        if(decoder->ext_decoder) {
                decompress_done(decoder->ext_decoder);
                decoder->ext_decoder = NULL;
        }
        for(int i = 0; i < 2; ++i) {
                if(decoder->ext_recv_buffer[i]) {
                        char **buf = decoder->ext_recv_buffer[i];
                        while(*buf != NULL) {
                                free(*buf);
                                buf++;
                        }
                        free(decoder->ext_recv_buffer[i]);
                        decoder->ext_recv_buffer[i] = NULL;
                }
        }

        desc.tile_count = get_video_mode_tiles_x(decoder->video_mode)
                        * get_video_mode_tiles_y(decoder->video_mode);
        
        out_codec = choose_codec_and_decoder(decoder, desc, &in_codec, &decode_line);
        if(out_codec == (codec_t) -1)
                return NULL;
        struct video_desc display_desc = desc;

        int display_mode;
        size_t len = sizeof(int);
        int ret;

        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_VIDEO_MODE,
                        &display_mode, &len);
        if(!ret) {
                debug_msg("Failed to get video display mode.");
                display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        }

        bool pp_does_change_tiling_mode = false;

        if (decoder->postprocess) {
                size_t len = sizeof(pp_does_change_tiling_mode);
                if(vo_postprocess_get_property(decoder->postprocess, VO_PP_DOES_CHANGE_TILING_MODE,
                                        &pp_does_change_tiling_mode, &len)) {
                        if(len == 0) {
                                // just for sake of completness since it shouldn't be a case
                                fprintf(stderr, "[Decoder] Warning: unable to get pp tiling mode!\n");
                        }
                }
        }

        if(decoder->postprocess) {
                struct video_desc pp_desc = desc;
                pp_desc.color_spec = out_codec;
                if(!pp_does_change_tiling_mode) {
                        pp_desc.width *= get_video_mode_tiles_x(decoder->video_mode);
                        pp_desc.height *= get_video_mode_tiles_y(decoder->video_mode);
                        pp_desc.tile_count = 1;
                }
                vo_postprocess_reconfigure(decoder->postprocess, pp_desc);
                decoder->pp_frame = vo_postprocess_getf(decoder->postprocess);
                vo_postprocess_get_out_desc(decoder->postprocess, &display_desc, &render_mode, &decoder->pp_output_frames_count);
        } else {
                if(display_mode == DISPLAY_PROPERTY_VIDEO_MERGED) {
                        display_desc.width *= get_video_mode_tiles_x(decoder->video_mode);
                        display_desc.height *= get_video_mode_tiles_y(decoder->video_mode);
                        display_desc.tile_count = 1;
                }
        }
        
        if(!is_codec_opaque(out_codec)) {
                decoder->change_il = select_il_func(desc.interlacing, decoder->disp_supported_il, decoder->disp_supported_il_cnt, &display_il);
        } else {
                decoder->change_il = NULL;
        }

        if (!decoder->postprocess || !pp_does_change_tiling_mode) { /* otherwise we need postprocessor mode, which we obtained before */
                render_mode = display_mode;
        }

        display_desc.color_spec = out_codec;
        display_desc.interlacing = display_il;

        if(!video_desc_eq(decoder->display_desc, display_desc))
        {
                int ret;
                /*
                 * TODO: put frame should be definitely here. On the other hand, we cannot be sure
                 * that vo driver is initialized so far:(
                 */
                //display_put_frame(decoder->display, frame);
                /* reconfigure VO and give it opportunity to pass us pitch */        
                ret = display_reconfigure(decoder->display, display_desc);
                if(!ret) {
                        fprintf(stderr, "[decoder] Unable to reconfigure display.\n");
                        exit_uv(128);
                        return NULL;
                }
                frame_display = display_get_frame(decoder->display);
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

static int check_for_mode_change(struct state_decoder *decoder, uint32_t *hdr, int packet_type)
{
        int mode_changed = FALSE;

        if(packet_type == PT_VIDEO) {
                uint32_t tmp;
                unsigned int width, height;
                codec_t color_spec;
                enum interlacing_t interlacing;
                double fps;
                int fps_pt, fpsd, fd, fi;

                width = ntohl(hdr[3]) >> 16;
                height = ntohl(hdr[3]) & 0xffff;
                color_spec = get_codec_from_fcc(hdr[4]);
                if(color_spec == (codec_t) -1) {
                        fprintf(stderr, "Unknown FourCC \"%4s\"!\n", (char *) &hdr[4]);
                }

                tmp = ntohl(hdr[5]);
                interlacing = (enum interlacing_t) (tmp >> 29);
                fps_pt = (tmp >> 19) & 0x3ff;
                fpsd = (tmp >> 15) & 0xf;
                fd = (tmp >> 14) & 0x1;
                fi = (tmp >> 13) & 0x1;

                fps = compute_fps(fps_pt, fpsd, fd, fi);
                if (!(decoder->received_vid_desc.width == width &&
                                        decoder->received_vid_desc.height == height &&
                                        decoder->received_vid_desc.color_spec == color_spec &&
                                        decoder->received_vid_desc.interlacing == interlacing  &&
                                        //decoder->received_vid_desc.video_type == video_type &&
                                        decoder->received_vid_desc.fps == fps
                     )) {
                        decoder->received_vid_desc = (struct video_desc) {
                                .width = width, 
                                        .height = height,
                                        .color_spec = color_spec,
                                        .interlacing = interlacing,
                                        .fps = fps };

                        mode_changed = TRUE;
                }
        } else { // PT_VIDEO_LDGM
                int tmp;
                int k, m, c, seed;

                tmp = ntohl(hdr[3]);
                k = tmp >> 19;
                m = 0x1fff & (tmp >> 6);
                c = 0x3f & tmp;
                seed = ntohl(hdr[4]);

                if(decoder->received_ldgm_desc.k != k ||
                                decoder->received_ldgm_desc.m != m ||
                                decoder->received_ldgm_desc.c != c ||
                                decoder->received_ldgm_desc.seed != seed
                  ) {
                        decoder->received_ldgm_desc.k = k;
                        decoder->received_ldgm_desc.m = m;
                        decoder->received_ldgm_desc.c = c;
                        decoder->received_ldgm_desc.seed = seed;
                        mode_changed = TRUE;
                }
        }

        return mode_changed;
}

int decode_frame(struct coded_data *cdata, void *decode_data)
{
        struct vcodec_state *pbuf_data = (struct vcodec_state *) decode_data;
        struct video_frame *frame = pbuf_data->frame_buffer;
        struct state_decoder *decoder = pbuf_data->decoder;

        int ret = TRUE;
        uint32_t offset;
        int len;
        rtp_packet *pckt = NULL;
        unsigned char *source;
        char *data;
        uint32_t data_pos;
        int prints=0;
        struct tile *tile = NULL;
        uint32_t tmp;
        uint32_t substream;

        int i;
        struct linked_list *pckt_list[decoder->max_substreams];
        uint32_t buffer_len[decoder->max_substreams];
        uint32_t buffer_num[decoder->max_substreams];

        for (i = 0; i < (int) decoder->max_substreams; ++i) {
                pckt_list[i] = ll_create();
                buffer_len[i] = 0;
                buffer_num[i] = 0;
        }

        perf_record(UVP_DECODEFRAME, frame);
        
        int k, m, c, seed; // LDGM
        int buffer_number, buffer_length;

        int pt;

        while (cdata != NULL) {
                uint32_t *hdr;
                pckt = cdata->data;

                pt = pckt->pt;
                hdr = (uint32_t *) pckt->data;
                data_pos = ntohl(hdr[1]);
                tmp = ntohl(hdr[0]);

                substream = tmp >> 22;
                buffer_number = tmp & 0x3ffff;
                buffer_length = ntohl(hdr[2]);

                if(pt == PT_VIDEO) {
                        len = pckt->data_len - sizeof(video_payload_hdr_t);
                        data = (char *) hdr + sizeof(video_payload_hdr_t);
                } else if (pt == PT_VIDEO_LDGM) {
                        len = pckt->data_len - sizeof(ldgm_video_payload_hdr_t);
                        data = (char *) hdr + sizeof(ldgm_video_payload_hdr_t);
                } else {
                        fprintf(stderr, "[decoder] Unknown packet type: %d.\n", pckt->pt);
                        exit_uv(1);
                        ret = FALSE;
                        goto cleanup;
                }

                if(substream >= decoder->max_substreams) {
                        fprintf(stderr, "[decoder] received substream ID %d. Expecting at most %d substreams. Did you set -M option?\n",
                                        substream, decoder->max_substreams);
                        // the guess is valid - we start with highest substream number (anytime - since it holds a m-bit)
                        // in next iterations, index is valid
                        if(substream == 1 || substream == 3) {
                                fprintf(stderr, "[decoder] Guessing mode: ");
                                if(substream == 1) {
                                        decoder_set_video_mode(decoder, VIDEO_STEREO);
                                } else {
                                        decoder_set_video_mode(decoder, VIDEO_4K);
                                }
                                decoder->received_vid_desc.width = 0; // just for sure, that we reconfigure in next iteration
                                fprintf(stderr, "%s\n", get_video_mode_description(decoder->video_mode));
                        } else {
                                exit_uv(1);
                        }
                        // we need skip this frame (variables are illegal in this iteration
                        // and in case that we got unrecognized number of substreams - exit
                        ret = FALSE;
                        goto cleanup;
                }


                buffer_num[substream] = buffer_number;
                buffer_len[substream] = buffer_length;

                ll_insert(pckt_list[substream], data_pos, len);
                
                if(check_for_mode_change(decoder, (uint32_t *) pckt->data, pt)) {
                        pbuf_data = (struct vcodec_state *) update_decoder_state(
                                        pbuf_data,
                                        pt == PT_VIDEO ? &decoder->received_vid_desc : NULL,
                                        pt == PT_VIDEO ? NULL : &decoder->received_ldgm_desc,
                                        decoder->max_substreams);
                        frame = pbuf_data->frame_buffer;
                        decoder = pbuf_data->decoder;

                        pbuf_data->reconfigured = true;
                        pbuf_data->max_frame_size = 0u;
                        pbuf_data->decoded = 0u;
                }
#if 0
                if (pt == PT_VIDEO) {
                } else if (pt == PT_VIDEO_LDGM) {
                        if(!decoder->fec_state.state || decoder->fec_state.k != k ||
                                        decoder->fec_state.m != m ||
                                        decoder->fec_state.c != c ||
                                        decoder->fec_state.seed != seed
                          ) {
                                if(decoder->fec_state.state) {
                                        ldgm_decoder_destroy(decoder->fec_state.state);
                                }
                                decoder->fec_state.k = k;
                                decoder->fec_state.m = m;
                                decoder->fec_state.c = c;
                                decoder->fec_state.seed = seed;
                                decoder->fec_state.state = ldgm_decoder_init(k, m, c, seed);
                                if(decoder->fec_state.state == NULL) {
                                        fprintf(stderr, "[decoder] Unable to initialize LDGM.\n");
                                        exit_uv(1);
                                        ret = FALSE;
                                        goto cleanup;
                                }
                        }
                }
#endif

                tile = vf_get_tile(frame, substream);

                struct line_decoder *line_decoder = 
                        &pbuf_data->line_decoder;

                if(line_decoder->decode_line) {
                        /* End of critical section */

                        /* MAGIC, don't touch it, you definitely break it 
                         *  *source* is data from network, *destination* is frame buffer
                         */

                        /* compute Y pos in source frame and convert it to 
                         * byte offset in the destination frame
                         */
                        int y = (data_pos / line_decoder->src_linesize) * line_decoder->dst_pitch;

                        /* compute X pos in source frame */
                        int s_x = data_pos % line_decoder->src_linesize;

                        /* convert X pos from source frame into the destination frame.
                         * it is byte offset from the beginning of a line. 
                         */
                        int d_x = ((int)((s_x) / line_decoder->src_bpp)) *
                                line_decoder->dst_bpp;

                        /* pointer to data payload in packet */
                        source = (unsigned char*)(data);

                        /* copy whole packet that can span several lines. 
                         * we need to clip data (v210 case) or center data (RGBA, R10k cases)
                         */
                        while (len > 0) {
                                /* len id payload length in source BPP
                                 * decoder needs len in destination BPP, so convert it 
                                 */                        
                                int l = ((int)(len / line_decoder->src_bpp)) * line_decoder->dst_bpp;

                                /* do not copy multiple lines, we need to 
                                 * copy (& clip, center) line by line 
                                 */
                                if (l + d_x > (int) line_decoder->dst_linesize) {
                                        l = line_decoder->dst_linesize - d_x;
                                }

                                /* compute byte offset in destination frame */
                                offset = y + d_x;

                                /* watch the SEGV */
                                if (l + line_decoder->base_offset + offset <= tile->data_len) {
                                        /*decode frame:
                                         * we have offset for destination
                                         * we update source contiguously
                                         * we pass {r,g,b}shifts */
                                        line_decoder->decode_line((unsigned char*)tile->data + line_decoder->base_offset + offset, source, l,
                                                        line_decoder->rshift, line_decoder->gshift,
                                                        line_decoder->bshift);
                                        /* we decoded one line (or a part of one line) to the end of the line
                                         * so decrease *source* len by 1 line (or that part of the line */
                                        len -= line_decoder->src_linesize - s_x;
                                        /* jump in source by the same amount */
                                        source += line_decoder->src_linesize - s_x;
                                } else {
                                        /* this should not ever happen as we call reconfigure before each packet
                                         * iff reconfigure is needed. But if it still happens, something is terribly wrong
                                         * say it loudly 
                                         */
                                        if((prints % 100) == 0) {
                                                fprintf(stderr, "WARNING!! Discarding input data as frame buffer is too small.\n"
                                                                "Well this should not happened. Expect troubles pretty soon.\n");
                                        }
                                        prints++;
                                        len = 0;
                                }
                                /* each new line continues from the beginning */
                                d_x = 0;        /* next line from beginning */
                                s_x = 0;
                                y += line_decoder->dst_pitch;  /* next line */
                        }
                } else {
                        memcpy(tile->data + data_pos, (unsigned char*)(pckt->data + sizeof(ldgm_video_payload_hdr_t)),
                                        len);
                }

                cdata = cdata->nxt;
        }

        if(!pckt) {
                ret = FALSE;
                goto cleanup;
        }

        for(i = 0; i < (int) decoder->max_substreams; ++i) {
                bool corrupted_frame_counted = false;
                if(buffer_len[i] != ll_count_bytes(pckt_list[i])) {
                        debug_msg("Frame incomplete - substream %d, buffer %d: expected %u bytes, got %u.\n", i,
                                        (unsigned int) buffer_num[i], buffer_len[i], (unsigned int) ll_count_bytes(pckt_list[i]));
                        ret = FALSE;
                }
        }

cleanup:
        ;
        unsigned int frame_size = 0;

        for(i = 0; i < (int) (sizeof(pckt_list) / sizeof(struct linked_list *)); ++i) {
                ll_destroy(pckt_list[i]);

                frame_size += buffer_len[i];
        }

        pbuf_data->max_frame_size = max(pbuf_data->max_frame_size, frame_size);

        if(ret) {
                decoder->displayed++;
                pbuf_data->decoded++;
        } else {
                decoder->corrupted++;
        }

        if(decoder->displayed % 600 == 599) {
                fprintf(stderr, "Decoder statistics: %lu displayed frames / %lu frames corrupted\n",
                                decoder->displayed, decoder->corrupted);
        }

        return ret;
}

