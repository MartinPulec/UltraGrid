/*
 * FILE:    jpeg.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2011 CESNET z.s.p.o.
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
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "host.h"
#include "video_compress/jpeg.h"
#include "libgpujpeg/gpujpeg_encoder.h"
#include "libgpujpeg/gpujpeg_common.h"
#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <queue>
#include <pthread.h>
#include <stdlib.h>

#define MAX_IN_QUEUE_LEN 5

using namespace std;

struct compress_jpeg_state {
        struct gpujpeg_encoder *encoder;
        struct gpujpeg_parameters encoder_param;
        
        decoder_t decoder;
        char *decoded;
        unsigned int interlaced_input:1;
        unsigned int rgb:1;
        codec_t color_spec;

        storage_t storage;
        struct gpujpeg_opengl_texture* texture;
        struct gpujpeg_encoder_input encoder_input;

        int device_id;
        
        queue<struct video_frame *> in;
        queue<struct video_frame *> out;

        pthread_mutex_t lock;
        pthread_cond_t worker_in_cv;
        pthread_cond_t boss_in_cv;
        pthread_cond_t boss_out_cv;

        pthread_t thread_id;
};

static void *worker(void *args);
static struct video_frame * jpeg_compress(void *arg, struct video_frame * tx);

static void *worker(void *args) {
        struct compress_jpeg_state *s = (struct compress_jpeg_state *) args;

        while(1) {
                struct video_frame *frame;
                pthread_mutex_lock(&s->lock);
                while(s->in.empty()) {
                        pthread_cond_wait(&s->worker_in_cv, &s->lock);
                }

                frame = s->in.front();
                s->in.pop();

                pthread_cond_signal(&s->boss_in_cv);

                if(!frame) {
                        // pass poisoned pill to consumer and exit
                        s->out.push(NULL);
                        pthread_cond_signal(&s->boss_out_cv);
                        pthread_mutex_unlock(&s->lock);
                        break;
                }
                pthread_mutex_unlock(&s->lock);

                struct video_frame *out_frame;

                out_frame = jpeg_compress(s, frame);

                pthread_mutex_lock(&s->lock);
                s->out.push(out_frame);
                pthread_cond_signal(&s->boss_out_cv);
                pthread_mutex_unlock(&s->lock);
        }

        return NULL;
}

static int configure_with(struct compress_jpeg_state *s, struct video_frame *frame);

static int configure_with(struct compress_jpeg_state *s, struct video_frame *frame)
{
        unsigned int x;
        
        switch (frame->color_spec) {
                case RGB:
                        s->decoder = (decoder_t) memcpy;
                        s->rgb = TRUE;
                        break;
                case RGBA:
                        s->decoder = (decoder_t) vc_copylineRGBAtoRGB;
                        s->rgb = TRUE;
                        break;
                /* TODO: enable (we need R10k -> RGB)
                 * case R10k:
                        s->decoder = (decoder_t) vc_copyliner10k;
                        s->rgb = TRUE;
                        break;*/
                case UYVY:
                case Vuy2:
                case DVS8:
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
                        exit_uv(128);
                        return FALSE;
        }

        s->storage = frame->tiles[0].storage;

        /* We will deinterlace the output frame */
        if(frame->interlacing == INTERLACED_MERGED)
                s->interlaced_input = TRUE;
        else if(frame->interlacing == PROGRESSIVE)
                s->interlaced_input = FALSE;
        else {
                fprintf(stderr, "Unsupported interlacing option: %s.\n", get_interlacing_description(frame->interlacing));
                exit_uv(128);
                return FALSE;
        }

	s->encoder_param.verbose = 0;

        if(s->rgb) {
                s->encoder_param.interleaved = 0;
                s->encoder_param.restart_interval = 8;
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
                s->encoder_param.restart_interval = 2;
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

        
        if(s->storage == OPENGL_TEXTURE) {
                s->texture = gpujpeg_opengl_texture_register(vf_get_tile(frame, 0)->texture, GPUJPEG_OPENGL_TEXTURE_READ);

                gpujpeg_encoder_input_set_texture(&s->encoder_input, s->texture);
        }
        
        s->encoder = gpujpeg_encoder_create(&s->encoder_param, &param_image);
        
        if(!s->encoder) {
                fprintf(stderr, "[JPEG] Failed to create encoder.\n");
                exit_uv(128);
                return FALSE;
        }
        
        s->decoded = (char *) malloc(4 * frame->tiles[0].width * frame->tiles[0].height);

        return TRUE;
}

void jpeg_push(void *args, struct video_frame * tx)
{
        struct compress_jpeg_state *s = (struct compress_jpeg_state *) args;

        pthread_mutex_lock(&s->lock);

        while(s->in.size() > MAX_IN_QUEUE_LEN) {
                pthread_cond_wait(&s->boss_in_cv, &s->lock);
        }

        s->in.push(tx);
        pthread_cond_signal(&s->worker_in_cv);
        pthread_mutex_unlock(&s->lock);
}

struct video_frame *jpeg_pop(void *args)
{
        struct compress_jpeg_state *s = (struct compress_jpeg_state *) args;
        struct video_frame *res;

        pthread_mutex_lock(&s->lock);
        while(s->out.empty()) {
                pthread_cond_wait(&s->boss_out_cv, &s->lock);
        }

        res = s->out.front();
        s->out.pop();

        pthread_mutex_unlock(&s->lock);

        return res;
}

void * jpeg_compress_init(char * opts)
{
        struct compress_jpeg_state *s;
        
        s = new compress_jpeg_state;

        s->decoded = NULL;
        s->device_id = 0;

        pthread_cond_init(&s->worker_in_cv, NULL);
        pthread_cond_init(&s->boss_in_cv, NULL);
        pthread_cond_init(&s->boss_out_cv, NULL);
        pthread_mutex_init(&s->lock, NULL);
                
        if(opts && strcmp(opts, "help") == 0) {
                printf("JPEG comperssion usage:\n");
                printf("\t-c JPEG[:<quality>][:<cuda_device>]]\n");
                printf("\nCUDA devices:\n");
                gpujpeg_print_devices_info();
                return NULL;
        }

        if(opts) {
                char *tok, *save_ptr = NULL;
                gpujpeg_set_default_parameters(&s->encoder_param);
                tok = strtok_r(opts, ":", &save_ptr);
                s->encoder_param.quality = atoi(tok);
                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        int ret;
                        s->device_id = atoi(tok);
                        ret = gpujpeg_init_device(s->device_id, 0);

                        if(ret != 0) {
                                fprintf(stderr, "[JPEG] initializing CUDA device %d failed.\n", atoi(tok));
                                return NULL;
                        }
                } else {
                        printf("Initializing CUDA device 0...\n");
                        int ret = gpujpeg_init_device(0, 0);
                        if(ret != 0) {
                                fprintf(stderr, "[JPEG] initializing default CUDA device failed.\n");
                                return NULL;
                        }
                }
                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        fprintf(stderr, "[JPEG] WARNING: Trailing configuration parameters.\n");
                }
                        
        } else {
                gpujpeg_set_default_parameters(&s->encoder_param);
                printf("[JPEG] setting default encode parameters (quality: %d)\n", 
                                s->encoder_param.quality
                );
        }
                
        s->encoder = NULL; /* not yet configured */

        if(pthread_create(&s->thread_id, NULL, worker, (void *) s) != 0) {
                perror("Unable to initialzize thread");
                return NULL;
        }

        return s;
}

static struct video_frame * jpeg_compress(void *arg, struct video_frame * tx)
{
        struct compress_jpeg_state *s = (struct compress_jpeg_state *) arg;
        int i;
        unsigned char *line1, *line2;

        unsigned int x;
        struct video_frame *res;

        cudaSetDevice(s->device_id);
        
        if(!s->encoder) {
                int ret;
                ret = configure_with(s, tx);
                if(!ret)
                        return NULL;
        }

        if(s->storage == CPU_POINTER) {

                for (x = 0; x < tx->tile_count;  ++x) {
                        struct tile *in_tile = vf_get_tile(tx, x);
                        
                        line1 = (unsigned char *) in_tile->data;
                        line2 = (unsigned char *) s->decoded;

                        int dst_linesize = vc_get_linesize(in_tile->width, s->rgb ? RGB : UYVY);
                        
                        for (i = 0; i < (int) in_tile->height; ++i) {
                                s->decoder(line2, line1, dst_linesize,
                                                0, 8, 16);
                                line1 += vc_get_linesize(in_tile->width, tx->color_spec);
                                line2 += dst_linesize;
                        }
                        
#if 0
                        line1 = (unsigned char *) out_tile->data + (in_tile->height - 1) * out_tile->linesize;
                        for( ; i < (int) s->out->tiles[0].height; ++i) {
                                memcpy(line2, line1, out_tile->linesize);
                                line2 += out_tile->linesize;
                        }
#endif
                        
                        gpujpeg_encoder_input_set_image(&s->encoder_input, (uint8_t *) s->decoded);
                        uint8_t *compressed;
                        int size;
                        int ret;
                        ret = gpujpeg_encoder_encode(s->encoder, &s->encoder_input, &compressed, &size);
                        
                        if(ret != 0)
                                return NULL;

                        res = tx;
                        free(tx->tiles[0].data);
                        tx->tiles[0].data_len = size;
                        tx->tiles[0].data = (char *) malloc(size);
                        memcpy(tx->tiles[0].data, compressed, size);
                }
        } else {
#if 0
                uint8_t *compressed;
                int size;
                int ret;

                ret = gpujpeg_encoder_encode(s->encoder, &s->encoder_input, &compressed, &size);
                
                if(ret != 0)
                        return NULL;
                struct tile *out_tile = vf_get_tile(s->out, 0);
                out_tile->data_len = size;
                memcpy(out_tile->data, compressed, size);
#endif
        }

        res->color_spec = JPEG;

        return res;
}

void jpeg_compress_done(void *arg)
{
        struct compress_jpeg_state *s = (struct compress_jpeg_state *) arg;
        cudaSetDevice(s->device_id);

        pthread_cond_destroy(&s->worker_in_cv);
        pthread_cond_destroy(&s->boss_in_cv);
        pthread_cond_destroy(&s->boss_out_cv);
        pthread_mutex_destroy(&s->lock);
        
        if(s->encoder)
                gpujpeg_encoder_destroy(s->encoder);
        
        delete s;
}
