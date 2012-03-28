/*
 * FILE:    video_codec.h
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
#ifndef __video_h

#define __video_h

#ifdef _HAVE_CONFIG_H
#include "config.h"
#endif

#include <GL/glew.h>

enum color_space {
        RGB_709_D65,
        XYZ
};

typedef enum {
        RGBA,
        UYVY,
        Vuy2,
        DVS8,
        R10k,
        v210,
        DVS10,
        DXT1,
        DXT1_YUV,
        DXT5,
        RGB,
        RGB16,
        DPX10,
        JPEG,
        RAW
} codec_t;

enum interlacing_t {
        PROGRESSIVE = 0,
        UPPER_FIELD_FIRST = 1,
        LOWER_FIELD_FIRST = 2,
        INTERLACED_MERGED = 3,
        SEGMENTED_FRAME = 4
};

typedef enum {
        CPU_POINTER,
        OPENGL_TEXTURE
} storage_t;

typedef union {
        char *cpu_pointer;
        GLuint texture;
} data_ptr_t;

enum lut_type {
        LUT_NONE,
        LUT_1D_TABLE,
        LUT_3D_MATRIX
};

struct lut_list;

struct lut_list {
        enum lut_type type;
        void *lut;
        struct lut_list *next;
};

#define VIDEO_NORMAL                    0u
#define VIDEO_DUAL                      1u
#define VIDEO_STEREO                    2u
#define VIDEO_4K                        3u

/* please note that tiles have also its own widths and heights */
struct video_desc {
        /* in case of tiled video - width and height represent widht and height
         * of each tile, eg. for tiled superHD 1920x1080 */
        unsigned int         width;
        unsigned int         height;

        codec_t              color_spec;
        double               fps;
        enum interlacing_t   interlacing;
        unsigned int         tile_count;

        enum color_space     colorspace;
};

struct video_frame 
{
        /* these variables are unset by vf_alloc */
        codec_t              color_spec;
        enum interlacing_t   interlacing;
        double               fps;

        /* number of frames (sequential number) */
        int                  frames;

        /* linked list of luts that should be applied in a color transform step */
        struct lut_list     *luts_to_apply;

        /* private values - should not be modified */
        /* set by vf_alloc */
        struct tile         *tiles;
        unsigned int         tile_count;
        //enum color_space     colorspace; 

};

struct tile {
        unsigned int         width;
        unsigned int         height;
        
        storage_t           storage;
        char                *data; /* this is not beginning of the frame buffer actually but beginning of displayed data,
                                     * it is the case display is centered in larger window, 
                                     * i.e., data = pixmap start + x_start + y_start*linesize
                                     */
        GLuint              texture;
        unsigned int         data_len; /* relative to data pos, not framebuffer size! */      
        unsigned int         linesize;
};

struct video_frame * vf_alloc(int count);
void vf_free(struct video_frame *buf);
struct tile * vf_get_tile(struct video_frame *buf, int pos);
int video_desc_eq(struct video_desc, struct video_desc);
int get_video_mode_tiles_x(int video_mode);
int get_video_mode_tiles_y(int video_mode);
const char *get_interlacing_description(enum interlacing_t);
const char *get_video_mode_description(int video_mode);


/* these functions transcode one interlacing format to another */
void il_upper_to_merged(char *dst, char *src, int linesize, int height);
void il_merged_to_upper(char *dst, char *src, int linesize, int height);

double compute_fps(int fps, int fpsd, int fd, int fi);

#define AUX_INTERLACED  (1<<0)
#define AUX_PROGRESSIVE (1<<1)
#define AUX_SF          (1<<2)
#define AUX_RGB         (1<<3) /* if device supports both, set both */
#define AUX_YUV         (1<<4) 
#define AUX_10Bit       (1<<5)


#endif

