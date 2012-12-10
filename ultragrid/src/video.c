/*
 * FILE:    video_codec.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
#include "config_win32.h"
#include "debug.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "cuda_memory_pool.h"
#include "video.h"

#include "video_codec.h"

static void default_free(void *ptr, size_t size);

static void default_free(void *ptr, size_t size) {
        UNUSED(size);
        free(ptr);
}

struct video_frame * vf_alloc(int count)
{
        struct video_frame *buf;
        
        buf = (struct video_frame *) calloc(1, sizeof(struct video_frame));
        
        buf->tiles = (struct tile *) 
                        calloc(1, sizeof(struct tile) * count);
        buf->tile_count = count;

        buf->luts_to_apply = NULL;
        buf->deleter = default_free;

        return buf;
}

struct video_frame * vf_alloc_desc(struct video_desc desc)
{
        struct video_frame *buf;
        assert(desc.tile_count > 0);

        buf = vf_alloc(desc.tile_count);
        if(!buf) return NULL;

        buf->color_spec = desc.color_spec;
        buf->interlacing = desc.interlacing;
        buf->fps = desc.fps;
        buf->deleter = default_free;
        // tile_count already filled
        for(unsigned int i = 0u; i < desc.tile_count; ++i) {
                buf->tiles[i].width = desc.width;
                buf->tiles[i].height = desc.height;
                buf->tiles[i].data = NULL;
                buf->tiles[i].data_len = 0;
                buf->tiles[i].linesize = 0;
        }

        return buf;
}

struct video_frame * vf_alloc_desc_data(struct video_desc desc)
{
        struct video_frame *buf;

        buf = vf_alloc_desc(desc);

        if(buf) {
                for(unsigned int i = 0; i < desc.tile_count; ++i) {
                        buf->tiles[i].linesize = vc_get_linesize(desc.width,
                                        desc.color_spec);
                        buf->tiles[i].data_len = buf->tiles[i].linesize *
                                desc.height;
                        buf->tiles[i].data = (char *) malloc(buf->tiles[i].data_len);
                }
        }

        return buf;
}

struct video_frame * vf_alloc_desc_data_cuda(struct video_desc desc)
{
        struct video_frame *buf;

        buf = vf_alloc_desc(desc);
        buf->deleter = cuda_free;

        if(buf) {
                for(unsigned int i = 0; i < desc.tile_count; ++i) {
                        buf->tiles[i].linesize = vc_get_linesize(desc.width,
                                        desc.color_spec);
                        buf->tiles[i].data_len = buf->tiles[i].linesize *
                                desc.height;
                        buf->tiles[i].data = (char *) cuda_alloc(buf->tiles[i].data_len);
                }
        }

        return buf;
}

void vf_free(struct video_frame *buf)
{
        if(!buf)
                return;
        free(buf->tiles);
        free(buf);
}

void vf_free_data(struct video_frame *buf)
{
        if(!buf)
                return;

        for(unsigned int i = 0u; i < buf->tile_count; ++i) {
                buf->deleter(buf->tiles[i].data, buf->tiles[i].data_len);
        }
        vf_free(buf);
}

struct tile * vf_get_tile(struct video_frame *buf, int pos)
{
        assert ((unsigned int) pos < buf->tile_count);

        return &buf->tiles[pos];
}

int video_desc_eq(struct video_desc a, struct video_desc b)
{
        return a.width == b.width &&
               a.height == b.height &&
               a.color_spec == b.color_spec &&
               fabs(a.fps - b.fps) < 0.01;// &&
               // TODO: remove these obsolete constants
               //(a.aux & (~AUX_RGB & ~AUX_YUV & ~AUX_10Bit)) == (b.aux & (~AUX_RGB & ~AUX_YUV & ~AUX_10Bit));
}

int get_video_mode_tiles_x(int video_type)
{
        int ret = 0;
        switch(video_type) {
                case VIDEO_NORMAL:
                case VIDEO_DUAL:
                        ret = 1;
                        break;
                case VIDEO_4K:
                case VIDEO_STEREO:
                        ret = 2;
                        break;
        }
        return ret;
}

int get_video_mode_tiles_y(int video_type)
{
        int ret = 0;
        switch(video_type) {
                case VIDEO_NORMAL:
                case VIDEO_STEREO:
                        ret = 1;
                        break;
                case VIDEO_4K:
                case VIDEO_DUAL:
                        ret = 2;
                        break;
        }
        return ret;
}

const char *get_interlacing_description(enum interlacing_t interlacing)
{
        switch (interlacing) {
                case PROGRESSIVE:
                        return "progressive";
                case UPPER_FIELD_FIRST:
                        return "interlaced (upper field first)";
                case LOWER_FIELD_FIRST:
                        return "interlaced (lower field first)";
                case INTERLACED_MERGED:
                        return "interlaced merged";
                case SEGMENTED_FRAME:
                        return "progressive segmented";
        }

        return NULL;
}

const char *get_interlacing_flag(enum interlacing_t interlacing)
{
        switch (interlacing) {
                case PROGRESSIVE:
                        return "p";
                case UPPER_FIELD_FIRST:
                        return "uff";
                case LOWER_FIELD_FIRST:
                        return "lff";
                case INTERLACED_MERGED:
                        return "i";
                case SEGMENTED_FRAME:
                        return "psf";
        }

        return NULL;
}

const char *get_video_mode_description(int video_mode)
{
        switch (video_mode) {
                case VIDEO_NORMAL:
                        return "normal";
                case VIDEO_STEREO:
                        return "3D";
                case VIDEO_4K:
                        return "tiled 4K";
                case VIDEO_DUAL:
                        return "dual-link";
        }
        return NULL;
}

/* TODO: rewrite following 2 functions in more efficient way */
void il_upper_to_merged(char *dst, char *src, int linesize, int height)
{
        int y;
        char *tmp = (char *) malloc(linesize * height);
        char *line1, *line2;

        line1 = tmp;
        line2 = src;
        for(y = 0; y < (height + 1) / 2; y ++) {
                memcpy(line1, line2, linesize);
                line1 += linesize * 2;
                line2 += linesize;
        }

        line1 = tmp + linesize;
        line2 = src + linesize * ((height + 1) / 2);
        for(y = 0; y < height / 2; y ++) {
                memcpy(line1, line2, linesize);
                line1 += linesize * 2;
                line2 += linesize;
        }
        memcpy(dst, tmp, linesize * height);
        free(tmp);
}

void il_merged_to_upper(char *dst, char *src, int linesize, int height)
{
        int y;
        char *tmp = (char *) malloc(linesize * height);
        char *line1, *line2;

        line1 = tmp;
        line2 = src;
        for(y = 0; y < (height + 1) / 2; y ++) {
                memcpy(line1, line2, linesize);
                line1 += linesize;
                line2 += linesize * 2;
        }

        line1 = tmp + linesize * ((height + 1) / 2);
        line2 = src + linesize;
        for(y = 0; y < height / 2; y ++) {
                memcpy(line1, line2, linesize);
                line1 += linesize;
                line2 += linesize * 2;
        }
        memcpy(dst, tmp, linesize * height);
        free(tmp);
}

double compute_fps(int fps, int fpsd, int fd, int fi)
{
        double res; 

        res = fps;
        if(fd)
                res /= 1.001;
        res /= fpsd;

        if(fi) {
                res = 1.0 / res;
        }

        return res;
}

void dump_video_desc(struct video_desc *desc)
{
        fprintf(stderr, "Video Desc: %dx%d@%f %s\n", 
                        desc->width,
                        desc->height,
                        desc->fps,
                        get_codec_name(desc->color_spec));
}

