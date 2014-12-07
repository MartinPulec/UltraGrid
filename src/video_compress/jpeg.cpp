/**
 * @file   src/video_compress/jpeg.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2014 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "compat/platform_spin.h"
#include "debug.h"
#include "host.h"
#include "video_compress.h"
#include "module.h"
#include "video_compress/jpeg.h"
#include "libgpujpeg/gpujpeg_encoder.h"
#include "utils/video_frame_pool.h"
#include "video.h"
#include <memory>
#include <pthread.h>
#include <stdlib.h>

using namespace std;

namespace {

struct state_video_compress_jpeg {
        struct module module_data;

        struct gpujpeg_encoder *encoder;
        struct gpujpeg_parameters encoder_param;

        decoder_t decoder;
        unique_ptr<char []> decoded;
        unsigned int rgb:1;
        codec_t color_spec;

        struct video_desc saved_desc;

        int restart_interval;
        platform_spin_t spin;

        int encoder_input_linesize;

        video_frame_pool<default_data_allocator> pool;
};

static bool configure_with(struct state_video_compress_jpeg *s, struct video_frame *frame);
static void cleanup_state(struct state_video_compress_jpeg *s);
static struct response *compress_change_callback(struct module *mod, struct message *msg);
static bool parse_fmt(struct state_video_compress_jpeg *s, char *fmt);
static void jpeg_compress_done(struct module *mod);

static bool configure_with(struct state_video_compress_jpeg *s, struct video_frame *frame)
{
        unsigned int x;

        s->saved_desc.width = frame->tiles[0].width;
        s->saved_desc.height = frame->tiles[0].height;
        s->saved_desc.color_spec = frame->color_spec;
        s->saved_desc.fps = frame->fps;
        s->saved_desc.interlacing = frame->interlacing;
        s->saved_desc.tile_count = frame->tile_count;

        for (x = 0; x < frame->tile_count; ++x) {
                if (vf_get_tile(frame, x)->width != vf_get_tile(frame, 0)->width ||
                                vf_get_tile(frame, x)->width != vf_get_tile(frame, 0)->width) {
                        fprintf(stderr,"[JPEG] Requested to compress tiles of different size!");
                        return false;
                }
        }

        struct video_desc compressed_desc;
        compressed_desc = video_desc_from_frame(frame);
        compressed_desc.color_spec = JPEG;

        switch (frame->color_spec) {
                case RGB:
                        s->decoder = (decoder_t) memcpy;
                        s->rgb = TRUE;
                        break;
                case RGBA:
                        s->decoder = (decoder_t) vc_copylineRGBAtoRGB;
                        s->rgb = TRUE;
                        break;
                case BGR:
                        s->decoder = (decoder_t) vc_copylineBGRtoRGB;
                        s->rgb = TRUE;
                        break;
                /* TODO: enable (we need R10k -> RGB)
                 * case R10k:
                        s->decoder = (decoder_t) vc_copyliner10k;
                        s->rgb = TRUE;
                        break;*/
                case YUYV:
                        s->decoder = (decoder_t) vc_copylineYUYV;
                        s->rgb = FALSE;
                        break;
                case UYVY:
                        s->decoder = (decoder_t) memcpy;
                        s->rgb = FALSE;
                        break;
                case v210:
                        s->decoder = (decoder_t) vc_copylinev210;
                        s->rgb = FALSE;
                        break;
                case DVS10:
                        s->decoder = (decoder_t) vc_copylineDVS10;
                        s->rgb = FALSE;
                        break;
                case DPX10:
                        s->decoder = (decoder_t) vc_copylineDPX10toRGB;
                        s->rgb = TRUE;
                        break;
                default:
                        fprintf(stderr, "[JPEG] Unknown codec: %d\n", frame->color_spec);
                        return false;
        }

	s->encoder_param.verbose = 0;
	s->encoder_param.segment_info = 1;

        if(s->rgb) {
                s->encoder_param.interleaved = 0;
                s->encoder_param.restart_interval = s->restart_interval == -1 ? 8
                        : s->restart_interval;
                /* LUMA */
                s->encoder_param.sampling_factor[0].horizontal = 1;
                s->encoder_param.sampling_factor[0].vertical = 1;
                /* Cb and Cr */
                s->encoder_param.sampling_factor[1].horizontal = 1;
                s->encoder_param.sampling_factor[1].vertical = 1;
                s->encoder_param.sampling_factor[2].horizontal = 1;
                s->encoder_param.sampling_factor[2].vertical = 1;
        } else {
                s->encoder_param.interleaved = 1;
                s->encoder_param.restart_interval = s->restart_interval == -1 ? 2
                        : s->restart_interval;
                /* LUMA */
                s->encoder_param.sampling_factor[0].horizontal = 2;
                s->encoder_param.sampling_factor[0].vertical = 1;
                /* Cb and Cr */
                s->encoder_param.sampling_factor[1].horizontal = 1;
                s->encoder_param.sampling_factor[1].vertical = 1;
                s->encoder_param.sampling_factor[2].horizontal = 1;
                s->encoder_param.sampling_factor[2].vertical = 1;
        }


        struct gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);

        param_image.width = frame->tiles[0].width;
        param_image.height = frame->tiles[0].height;

        param_image.comp_count = 3;
        if(s->rgb) {
                param_image.color_space = GPUJPEG_RGB;
                param_image.sampling_factor = GPUJPEG_4_4_4;
        } else {
                param_image.color_space = GPUJPEG_YCBCR_BT709;
                param_image.sampling_factor = GPUJPEG_4_2_2;
        }

        s->encoder = gpujpeg_encoder_create(&s->encoder_param, &param_image);

        int data_len = frame->tiles[0].width * frame->tiles[0].height * 3;
        s->pool.reconfigure(compressed_desc, data_len);

        s->encoder_input_linesize = frame->tiles[0].width *
                (param_image.color_space == GPUJPEG_RGB ? 3 : 2);

        if(!s->encoder) {
                fprintf(stderr, "[JPEG] Failed to create encoder.\n");
                exit_uv(128);
                return false;
        }

        s->decoded = unique_ptr<char []>(new char[4 * frame->tiles[0].width * frame->tiles[0].height]);
        return true;
}

static struct response *compress_change_callback(struct module *mod, struct message *msg)
{
        struct state_video_compress_jpeg *s = (struct state_video_compress_jpeg *) mod->priv_data;

        static struct response *ret;

        struct msg_change_compress_data *data =
                (struct msg_change_compress_data *) msg;

        platform_spin_lock(&s->spin);
        parse_fmt(s, data->config_string);
        ret = new_response(RESPONSE_OK, NULL);
        memset(&s->saved_desc, 0, sizeof(s->saved_desc));
        platform_spin_unlock(&s->spin);

        free_message(msg);

        return ret;
}

static bool parse_fmt(struct state_video_compress_jpeg *s, char *fmt)
{
        if(fmt && fmt[0] != '\0') {
                char *tok, *save_ptr = NULL;
                gpujpeg_set_default_parameters(&s->encoder_param);
                tok = strtok_r(fmt, ":", &save_ptr);
                s->encoder_param.quality = atoi(tok);
                if (s->encoder_param.quality <= 0 || s->encoder_param.quality > 100) {
                        fprintf(stderr, "[JPEG] Error: Quality should be in interval (0-100]!\n");
                        return false;
                }

                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        s->restart_interval = atoi(tok);
                        if (s->restart_interval < 0) {
                                fprintf(stderr, "[JPEG] Error: Restart interval should be non-negative!\n");
                                return false;
                        }
                }
                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        fprintf(stderr, "[JPEG] WARNING: Trailing configuration parameters.\n");
                }
        }

        return true;
}

bool jpeg_is_supported() {
        return gpujpeg_init_device(cuda_devices[0], TRUE) == 0;
}

struct module * jpeg_compress_init(struct module *parent, const struct video_compress_params *params)
{
        struct state_video_compress_jpeg *s;
        const char *opts = params->cfg;

        if(opts && strcmp(opts, "help") == 0) {
                printf("JPEG comperssion usage:\n");
                printf("\t-c JPEG[:<quality>[:<restart_interval>]]\n");
                return &compress_init_noerr;
        } else if(opts && strcmp(opts, "list_devices") == 0) {
                printf("CUDA devices:\n");
                gpujpeg_print_devices_info();
                return &compress_init_noerr;
        }

        s = new state_video_compress_jpeg();

        s->restart_interval = -1;

        gpujpeg_set_default_parameters(&s->encoder_param);

        if(opts && opts[0] != '\0') {
                char *fmt = strdup(opts);
                if (!parse_fmt(s, fmt)) {
                        free(fmt);
                        delete s;
                        return NULL;
                }
                free(fmt);
        } else {
                printf("[JPEG] setting default encode parameters (quality: %d)\n",
                                s->encoder_param.quality
                );
        }

        s->encoder = NULL; /* not yet configured */

        platform_spin_init(&s->spin);

        module_init_default(&s->module_data);
        s->module_data.cls = MODULE_CLASS_DATA;
        s->module_data.priv_data = s;
        s->module_data.deleter = jpeg_compress_done;
        s->module_data.msg_callback = compress_change_callback;
        module_register(&s->module_data, parent);

        return &s->module_data;
}

shared_ptr<video_frame> jpeg_compress(struct module *mod, shared_ptr<video_frame> tx)
{

if (tx == NULL) return NULL;
        struct state_video_compress_jpeg *s = (struct state_video_compress_jpeg *) mod->priv_data;
        int i;
        unsigned char *line1, *line2;

        unsigned int x;

        gpujpeg_set_device(cuda_devices[0]);

        if(!s->encoder) {
                int ret;
                printf("Initializing CUDA device %d...\n", cuda_devices[0]);
                ret = gpujpeg_init_device(cuda_devices[0], TRUE);

                if(ret != 0) {
                        fprintf(stderr, "[JPEG] initializing CUDA device %d failed.\n", cuda_devices[0]);
                        exit_uv(127);
                        return {};
                }
                ret = configure_with(s, tx.get());
                if (!ret) {
                        exit_uv(127);
                        return {};
                }
        }

        struct video_desc desc;
        desc = video_desc_from_frame(tx.get());

        // if format changed, reconfigure
        if(!video_desc_eq_excl_param(s->saved_desc, desc, PARAM_INTERLACING)) {
                cleanup_state(s);
                int ret;
                ret = configure_with(s, tx.get());
                if(!ret) {
                        exit_uv(127);
                        return NULL;
                }
        }

        shared_ptr<video_frame> out = s->pool.get_frame();

        for (x = 0; x < tx->tile_count;  ++x) {
                struct tile *in_tile = vf_get_tile(tx.get(), x);
                struct tile *out_tile = vf_get_tile(out.get(), x);

                line1 = (unsigned char *) in_tile->data;
                line2 = (unsigned char *) s->decoded.get();

                for (i = 0; i < (int) in_tile->height; ++i) {
                        s->decoder(line2, line1, s->encoder_input_linesize,
                                        0, 8, 16);
                        line1 += vc_get_linesize(in_tile->width, tx->color_spec);
                        line2 += s->encoder_input_linesize;
                }

                line1 = (unsigned char *) out_tile->data + (in_tile->height - 1) * s->encoder_input_linesize;
                for( ; i < (int) out->tiles[0].height; ++i) {
                        memcpy(line2, line1, s->encoder_input_linesize);
                        line2 += s->encoder_input_linesize;
                }

                /*if(s->interlaced_input)
                        vc_deinterlace((unsigned char *) s->decoded, s->encoder_input_linesize,
                                        s->out->tiles[0].height);*/

                uint8_t *compressed;
                int size;
                int ret;


                struct gpujpeg_encoder_input encoder_input;
                gpujpeg_encoder_input_set_image(&encoder_input, (uint8_t *) s->decoded.get());
                ret = gpujpeg_encoder_encode(s->encoder, &encoder_input, &compressed, &size);

                if(ret != 0) {
                        return {};
                }

                out_tile->data_len = size;
                memcpy(out_tile->data, compressed, size);
        }

        return out;
}

static void jpeg_compress_done(struct module *mod)
{
        struct state_video_compress_jpeg *s = (struct state_video_compress_jpeg *) mod->priv_data;

        cleanup_state(s);

        platform_spin_destroy(&s->spin);

        delete s;
}

static void cleanup_state(struct state_video_compress_jpeg *s)
{
        if (s->encoder)
                gpujpeg_encoder_destroy(s->encoder);
        s->encoder = NULL;
}

} // end of anonymous namespace

struct compress_info_t jpeg_info = {
        "JPEG",
        jpeg_compress_init,
        jpeg_compress,
        NULL,
        jpeg_is_supported,
        {
                { "60", 60, 30*1000*1000, {10, 0.6, 75}, {10, 0.6, 75} },
                { "80", 70, 36*1000*1000, {12, 0.6, 90}, {15, 0.6, 100} },
                { "90", 80, 44*1000*1000, {15, 0.6, 100}, {20, 0.6, 150} },
        },
};

