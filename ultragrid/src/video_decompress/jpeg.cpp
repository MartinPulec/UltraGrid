/*
 * FILE:    video_decompress/dxt_glsl.c
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

#include "libgpujpeg/gpujpeg_decoder.h"
//#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#include "video_decompress/jpeg.h"

#include <queue>

using namespace std;
using namespace std::tr1;

struct jpeg_decompress_frame {
        std::tr1::shared_ptr<char> data;
        size_t                     len;
};

struct state_decompress_jpeg {
        struct gpujpeg_decoder *decoder;

        struct video_desc desc;
        int compressed_len;
        int rshift, gshift, bshift;
        int pitch;
        codec_t out_codec;

        queue<struct video_frame *> in;
        queue<struct video_frame *> out;

        pthread_mutex_t lock;
        pthread_cond_t in_cv;
        pthread_cond_t out_cv;

        pthread_t thread_id;
};

static int configure_with(struct state_decompress_jpeg *s, struct video_desc desc);

static int configure_with(struct state_decompress_jpeg *s, struct video_desc desc)
{
        s->desc = desc;

        s->decoder = gpujpeg_decoder_create();
        if(!s->decoder) {
                return FALSE;
        }
        if(s->out_codec == RGB || s->out_codec == RGBA) {
                s->decoder->coder.param_image.color_space = GPUJPEG_RGB;
                s->decoder->coder.param_image.sampling_factor = GPUJPEG_4_4_4;
                s->compressed_len = desc.width * desc.height * 2;
        } else {
                s->decoder->coder.param_image.color_space = GPUJPEG_YCBCR_BT709;
                s->decoder->coder.param_image.sampling_factor = GPUJPEG_4_2_2;
                s->compressed_len = desc.width * desc.height * 3;
        }

        return TRUE;
}

void * jpeg_decompress_init(void)
{
        struct state_decompress_jpeg *s;

        s = new state_decompress_jpeg; 
        pthread_cond_init(&s->in_cv, NULL);
        pthread_cond_init(&s->out_cv, NULL);
        pthread_mutex_init(&s->lock, NULL);

        return s;
}

int jpeg_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_jpeg *s = (struct state_decompress_jpeg *) state;
        int ret;

        assert(out_codec == RGB || out_codec == RGBA || out_codec == UYVY);

        s->out_codec = out_codec;
        s->pitch = pitch;
        s->rshift = rshift;
        s->gshift = gshift;
        s->bshift = bshift;
        if(!s->decoder) {
                ret = configure_with(s, desc);
        } else {
                gpujpeg_decoder_destroy(s->decoder);
                ret = configure_with(s, desc);
        }

        if(ret)
                return s->compressed_len;
        else
                return 0;
}

void jpeg_decompress(void *state, unsigned char *dst, unsigned char *buffer, unsigned int src_len)
{
        struct state_decompress_jpeg *s = (struct state_decompress_jpeg *) state;
        int ret;
        struct gpujpeg_decoder_output decoder_output;


        if(s->out_codec == UYVY || (s->out_codec == RGB && s->rshift == 0 && s->gshift == 8 && s->bshift == 16)) {
                gpujpeg_decoder_output_set_default(&decoder_output);
                decoder_output.type = GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER;
                decoder_output.data = dst;
                //int data_decompressed_size = decoder_output.data_size;

                ret = gpujpeg_decoder_decode(s->decoder, (uint8_t*) buffer, src_len, &decoder_output);
                if (ret != 0) return;
        } else {
                unsigned int i;
                int linesize;
                unsigned char *line_src, *line_dst;

                gpujpeg_decoder_output_set_default(&decoder_output);
                decoder_output.type = GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER;
                //int data_decompressed_size = decoder_output.data_size;

                ret = gpujpeg_decoder_decode(s->decoder, (uint8_t*) buffer, src_len, &decoder_output);

                if (ret != 0) return;
                if(s->out_codec == RGB) {
                        linesize = s->desc.width * 3;
                } else if(s->out_codec == RGBA) {
                        linesize = s->desc.width * 4;
                } else if(s->out_codec == UYVY) {
                        linesize = s->desc.width * 2;
                }

                line_dst = dst;
                line_src = decoder_output.data;
                for(i = 0u; i < s->desc.height; i++) {
                        if(s->out_codec == RGB) {
                                vc_copylineRGB(line_dst, line_src, linesize,
                                                s->rshift, s->gshift, s->bshift);
                        } else if(s->out_codec == RGBA) {
                                vc_copylineRGBtoRGBA(line_dst, line_src, linesize,
                                                s->rshift, s->gshift, s->bshift);
                        } else {
                                memcpy(line_dst, line_src, linesize);
                        }

                        line_dst += s->pitch;
                        line_src += linesize;

                }
        }
}

void jpeg_push(void *state, std::tr1::shared_ptr<Frame> src)
{

}

std::tr1::shared_ptr<Frame> jpeg_pop(void *state)
{

}

void jpeg_decompress_done(void *state)
{
        struct state_decompress_jpeg *s = (struct state_decompress_jpeg *) state;

        if(s->decoder) {
                gpujpeg_decoder_destroy(s->decoder);
        }

        delete s;
}
