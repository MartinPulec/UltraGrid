/**
 * @file   video_frame.c
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * @brief This file contains video frame manipulation functions.
 */
/*
 * Copyright (c) 2005-2025 CESNET
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

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "pixfmt_conv.h"
#include "utils/pam.h"
#include "utils/y4m.h"
#include "video_codec.h"
#include "video_frame.h"

#define MOD_NAME "[video frame] "

struct video_frame * vf_alloc(int count)
{
        struct video_frame *buf;
        assert(count > 0);
        
        buf = (struct video_frame *) calloc(1, offsetof (struct video_frame, tiles[count]));
        assert(buf != NULL);
        
        buf->tile_count = count;

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
        // tile_count already filled
        for(unsigned int i = 0u; i < desc.tile_count; ++i) {
                buf->tiles[i].width = desc.width;
                buf->tiles[i].height = desc.height;
                if(codec_is_const_size(desc.color_spec)){
                        buf->tiles[i].data_len = get_pf_block_bytes(desc.color_spec);
                } else {
                        buf->tiles[i].data_len = vc_get_datalen(desc.width, desc.height, desc.color_spec);
                }
        }

        return buf;
}

static void vf_aligned_data_deleter(struct video_frame *buf)
{
        for (unsigned int i = 0u; i < buf->tile_count; ++i) {
                aligned_free(buf->tiles[i].data);
        }
}

/**
 * @brief allocates struct video_frame including data pointers in RAM
 * @note
 * Try to use hugepages in Linux, which may improve performance. See:
 * - https://kernel.org/doc/html//v5.15/admin-guide/mm/transhuge.html
 * - https://rigtorp.se/hugepages/
 */
struct video_frame * vf_alloc_desc_data(struct video_desc desc)
{
        struct video_frame *buf;

        buf = vf_alloc_desc(desc);

        if (!buf) {
                return NULL;
        }
        for(unsigned int i = 0; i < desc.tile_count; ++i) {
                if(codec_is_const_size(desc.color_spec)){
                        buf->tiles[i].data_len = get_pf_block_bytes(desc.color_spec);
                } else {
                        buf->tiles[i].data_len = vc_get_linesize(desc.width,
                                        desc.color_spec) *
                                desc.height;
                }
                buf->tiles[i].data = (char *) aligned_malloc(buf->tiles[i].data_len + MAX_PADDING, 1U<<21U /* 2 MiB */);
                assert(buf->tiles[i].data != NULL);
#ifdef __linux__
                madvise(buf->tiles[0].data, buf->tiles[0].data_len, MADV_HUGEPAGE);
#endif
        }

        buf->callbacks.data_deleter = vf_aligned_data_deleter;
        buf->callbacks.recycle = NULL;

        return buf;
}

void vf_free(struct video_frame *buf)
{
        if(!buf)
                return;

        vf_recycle(buf);

        if (buf->callbacks.data_deleter) {
                buf->callbacks.data_deleter(buf);
        }
        free(buf);
}

void vf_recycle(struct video_frame *buf)
{
        if(!buf)
                return;

        if(buf->callbacks.recycle)
                buf->callbacks.recycle(buf);
}

void vf_data_deleter(struct video_frame *buf)
{
        if(!buf)
                return;

        for(unsigned int i = 0u; i < buf->tile_count; ++i) {
                free(buf->tiles[i].data);
        }
}

struct tile * vf_get_tile(struct video_frame *buf, int pos)
{
        assert ((unsigned int) pos < buf->tile_count);

        return &buf->tiles[pos];
}

bool video_desc_eq(struct video_desc a, struct video_desc b)
{
        return video_desc_eq_excl_param(a, b, 0);
}

bool video_desc_eq_excl_param(struct video_desc a, struct video_desc b, unsigned int excluded_params)
{
        return ((excluded_params & PARAM_WIDTH) || a.width == b.width) &&
                ((excluded_params & PARAM_HEIGHT) || a.height == b.height) &&
                ((excluded_params & PARAM_CODEC) || a.color_spec == b.color_spec) &&
                ((excluded_params & PARAM_INTERLACING) || a.interlacing == b.interlacing) &&
                ((excluded_params & PARAM_TILE_COUNT) || a.tile_count == b.tile_count) &&
                ((excluded_params & PARAM_FPS) || fabs(a.fps - b.fps) < 0.01);// &&
}

struct video_desc
video_desc_from_frame(const struct video_frame *frame)
{
        struct video_desc desc;

        assert(frame != NULL);

        desc.width = frame->tiles[0].width;
        desc.height = frame->tiles[0].height;
        desc.color_spec = frame->color_spec;
        desc.fps = frame->fps;
        desc.interlacing = frame->interlacing;
        desc.tile_count = frame->tile_count;

        return desc;
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

static const char *interlacing_suffixes[] = {
                [PROGRESSIVE] = "p",
                [UPPER_FIELD_FIRST] = "tff",
                [LOWER_FIELD_FIRST] = "bff",
                [INTERLACED_MERGED] = "i",
                [SEGMENTED_FRAME] = "psf",
};

const char *get_interlacing_suffix(enum interlacing_t interlacing)
{
        if (interlacing < sizeof interlacing_suffixes / sizeof interlacing_suffixes[0])
                return interlacing_suffixes[interlacing];
        return NULL;
}

/**
 * @returns interlacing_t member
 * @retval INTERLACING_MAX+1 on error
 */
enum interlacing_t get_interlacing_from_suffix(const char *suffix)
{
        for (size_t i = 0; i < sizeof interlacing_suffixes / sizeof interlacing_suffixes[0]; ++i) {
                if (interlacing_suffixes[i] && strcmp(suffix, interlacing_suffixes[i]) == 0) {
                        return i;
                }
        }

        return INTERLACING_MAX + 1;
}

/**
 * @todo
 * Needs to be more efficient
 */
void il_lower_to_merged(char *dst, char *src, int linesize, int height, void **stored_state)
{
        struct il_lower_to_merged_state {
                size_t field_len;
                char field[];
        };
        struct il_lower_to_merged_state *last_field = (struct il_lower_to_merged_state *) *stored_state;

        char *tmp = malloc(linesize * height);
        char *line1, *line2;

        // upper field
        line1 = tmp;
        int upper_field_len = linesize * ((height + 1) / 2);
        // first check if we have field from last frame
        if (last_field == NULL) {
                last_field = (struct il_lower_to_merged_state *)
                        malloc(sizeof(struct il_lower_to_merged_state) + upper_field_len);
                last_field->field_len = upper_field_len;
                *stored_state = last_field;
                // if no, use current one
                line2 = src + linesize * (height / 2);
        } else {
                // otherwise use field from last "frame"
                line2 = last_field->field;
        }
        for (int y = 0; y < (height + 1) / 2; y++) {
                memcpy(line1, line2, linesize);
                line1 += linesize * 2;
                line2 += linesize;
        }
        // store
        assert ((int) last_field->field_len == upper_field_len);
        memcpy(last_field->field, src + linesize * (height / 2), upper_field_len);

        // lower field
        line1 = tmp + linesize;
        line2 = src;
        for (int y = 0; y < height / 2; y++) {
                memcpy(line1, line2, linesize);
                line1 += linesize * 2;
                line2 += linesize;
        }
        memcpy(dst, tmp, linesize * height);
        free(tmp);
}

/* TODO: rewrite following 2 functions in more efficient way */
void il_upper_to_merged(char *dst, char *src, int linesize, int height, void **state)
{
        UNUSED(state);
        int y;
        char *tmp = malloc(linesize * height);
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

void il_merged_to_upper(char *dst, char *src, int linesize, int height, void **state)
{
        UNUSED(state);
        int y;
        char *tmp = malloc(linesize * height);
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

/**
 * Computes FPS from packet format values:
 * https://www.cesnet.cz/wp-content/uploads/2013/01/ultragrid-4k.pdf
 *
 * @note
 * Current implementation in UG differs from the above in a sense that fps and fpsd
 * are not offset by one, so 30 means really 30, not 31. As a consequence a value of
 * fpsd == 0 is invalid but it can represent some special value in future (undefined,
 * infinite).
 *
 * @retval computed fps on success or -1 on error (fpsd == 0)
 */
double compute_fps(int fps, int fpsd, int fd, int fi)
{
        if (fpsd == 0) {
                return -1;
        }

        double res = fps;
        if (fd) {
                res /= 1.001;
        }
        res /= fpsd;

        if (fi) {
                res = 1.0 / res;
        }

        return res;
}

struct video_frame *vf_get_copy(struct video_frame *original_frame) {
        struct video_frame *frame_copy = vf_alloc_desc(video_desc_from_frame(original_frame));

        for(int i = 0; i < (int) frame_copy->tile_count; ++i) {
                frame_copy->tiles[i].data = (char *) malloc(frame_copy->tiles[i].data_len);
                memcpy(frame_copy->tiles[i].data, original_frame->tiles[i].data,
                                frame_copy->tiles[i].data_len);
        }

        if(frame_copy->callbacks.copy){
                frame_copy->callbacks.copy(frame_copy);
        }

        frame_copy->callbacks.data_deleter = vf_data_deleter;

        return frame_copy;
}

/**
 * returns RGB >8-bit data eligible to be written to PNM (big endian in 16-bit container)
 */
static unsigned char *get_16_bit_pnm_data(struct video_frame *frame) {
        decoder_t dec = get_decoder_from_to(frame->color_spec, RG48);
        if (!dec) {
                log_msg(LOG_LEVEL_WARNING, "Unable to find decoder from %s to RG48\n",
                                get_codec_name(frame->color_spec));
                return NULL;
        }
        unsigned char *tmp = malloc(vc_get_datalen(frame->tiles[0].width, frame->tiles[0].height, RG48));
        assert (tmp);
        int src_linesize = vc_get_linesize(frame->tiles[0].width, frame->color_spec);
        int dst_linesize = vc_get_linesize(frame->tiles[0].width, RG48);
        unsigned depth = get_bits_per_component(frame->color_spec);
        for (unsigned i = 0; i < frame->tiles[0].height; ++i) {
                uint16_t *dstline = (uint16_t *) (void *) (tmp + (size_t) i * dst_linesize);
                dec((unsigned char *) dstline, (unsigned char *) frame->tiles[0].data + (size_t) i * src_linesize,
                                dst_linesize, DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
                for (unsigned i = 0; i < frame->tiles[0].width * 3; ++i) {
                        uint16_t tmp = *dstline;
                        tmp >>= 16U - depth;
                        *dstline++ = htons(tmp);
                }
        }
        return tmp;
}

bool save_video_frame_as_pnm(struct video_frame *frame, const char *name)
{
        unsigned char *data = NULL, *tmp_data = NULL;
        struct tile *tile = &frame->tiles[0];
        if (frame->color_spec == RGB) {
                data = (unsigned char *) tile->data;
        } else if (get_bits_per_component(frame->color_spec) <= 8 ||
                        get_decoder_from_to(frame->color_spec, RG48) == NULL) {
                decoder_t dec = get_decoder_from_to(frame->color_spec, RGB);
                if (!dec) {
                        log_msg(LOG_LEVEL_WARNING, "Unable to find decoder from %s to RGB\n",
                                        get_codec_name(frame->color_spec));
                        return false;
                }
                int len = tile->width * tile->height * 3;
                data = tmp_data = (unsigned char *) malloc(len);
                dec (data, (const unsigned char *) tile->data, len, 0, 0, 0);
        } else {
                data = tmp_data = get_16_bit_pnm_data(frame);
                if (!data) {
                        return false;
                }
        }

        if (!data) {
                return false;
        }

        pam_write(name, tile->width, tile->width, tile->height, 3,
                  (1 << get_bits_per_component(frame->color_spec)) - 1, data,
                  true);
        free(tmp_data);

        return true;
}

static bool save_video_frame_as_y4m(struct video_frame *frame, const char *name)
{
        struct tile *tile = &frame->tiles[0];
        if (get_bits_per_component(frame->color_spec) <= 8 && (frame->color_spec == UYVY || get_decoder_from_to(frame->color_spec, UYVY))) {
                unsigned char *uyvy          = (unsigned char *) tile->data;
                unsigned char *tmp_data_uyvy = NULL;
                if (frame->color_spec != UYVY) {
                        decoder_t dec = get_decoder_from_to(frame->color_spec, UYVY);
                        int len = vc_get_datalen(tile->width, tile->height, UYVY);
                        uyvy = tmp_data_uyvy = malloc(len);
                        dec (uyvy, (const unsigned char *) tile->data, len, 0, 0, 0);
                }
                unsigned char *i422 = malloc(tile->width * tile->height + 2 * ((tile->width + 1) / 2) * tile->height);
                uyvy_to_i422(tile->width, tile->height, uyvy, i422);

                struct y4m_metadata info = { .width = tile->width, .height = tile->height, .bitdepth = 8, .subsampling = Y4M_SUBS_422, .limited = true };
                bool ret = y4m_write(name, &info, i422);
                free(tmp_data_uyvy);
                free(i422);
                return ret;
        } else if (get_decoder_from_to(frame->color_spec, Y416)) {
                unsigned char *y416          = (unsigned char *) tile->data;
                unsigned char *tmp_data_y416 = NULL;
                if (frame->color_spec != Y416) {
                        decoder_t dec = get_decoder_from_to(frame->color_spec, Y416);
                        int len = vc_get_datalen(tile->width, tile->height, Y416);
                        y416 = tmp_data_y416 = malloc(len);
                        dec (y416, (const unsigned char *) tile->data, len, 0, 0, 0);
                }
                unsigned char *i444 = malloc(tile->width * tile->height * 6);
                int depth = get_bits_per_component(frame->color_spec);
                y416_to_i444(tile->width, tile->height, y416, i444, depth);

                struct y4m_metadata info = { .width = tile->width, .height = tile->height, .bitdepth = depth, .subsampling = Y4M_SUBS_444, .limited = true };
                bool ret = y4m_write(name, &info, (unsigned char *) i444);
                free(tmp_data_y416);
                free(i444);
                return ret;
        }

        log_msg(LOG_LEVEL_WARNING, "Unable to find decoder from %s to UYVY or Y416\n",
                        get_codec_name(frame->color_spec));
        return false;
}

/**
 * Saves video_frame to file name.<ext>.
 */
const char *save_video_frame(struct video_frame *frame, const char *name, bool raw) {
        _Thread_local static char filename[FILENAME_MAX];
        if (!raw && !is_codec_opaque(frame->color_spec)) {
                if (codec_is_a_rgb(frame->color_spec) &&
                                ((get_bits_per_component(frame->color_spec) <= 8 && get_decoder_from_to(frame->color_spec, RGB))
                                 || (get_bits_per_component(frame->color_spec) > 8 && get_decoder_from_to(frame->color_spec, RG48)))) {
                        snprintf(filename, sizeof filename, "%s.pnm", name);
                        bool ret = save_video_frame_as_pnm(frame, filename);
                        return ret ? filename : NULL;
                } else if (!codec_is_a_rgb(frame->color_spec) &&
                                ((get_bits_per_component(frame->color_spec) <= 8 && get_decoder_from_to(frame->color_spec, UYVY))
                                 || (get_bits_per_component(frame->color_spec) > 8 && get_decoder_from_to(frame->color_spec, Y416)))) {
                        snprintf(filename, sizeof filename, "%s.y4m", name);
                        bool ret = save_video_frame_as_y4m(frame, filename);
                        return ret ? filename : NULL;
                }
        }
        snprintf(filename, sizeof filename, "%s.%s", name, get_codec_file_extension(frame->color_spec));
        errno = 0;
        FILE *out = fopen(filename, "wb");
        if (out == NULL) {
                perror("save_video_frame fopen");
                return NULL;
        }
        fwrite(frame->tiles[0].data, frame->tiles[0].data_len, 1, out);
        if (ferror(out)) {
                perror("save_video_frame fwrite");
                fclose(out);
                return NULL;
        }
        fclose(out);
        return filename;
}

struct video_frame *
load_video_frame(const char *name, codec_t codec, int width, int height)
{
        FILE *in = fopen(name, "rb");
        if (!in) {
                perror("fopen");
                return NULL;
        }
        struct video_frame *out = vf_alloc_desc_data(
            (struct video_desc){ width, height, codec, 1.0, PROGRESSIVE, 1 });
        size_t bytes = fread(out->tiles[0].data, 1, out->tiles[0].data_len, in);
        fclose(in);
        if (bytes == out->tiles[0].data_len) {
                return out;
        }
        MSG(ERROR, "Cannot read %u B from %s, got only %zu B!\n",
                out->tiles[0].data_len, name, bytes);
        vf_free(out);
        return NULL;
}

void
vf_copy_metadata(struct video_frame *dest, const struct video_frame *src)
{
        memcpy((char *) dest + offsetof(struct video_frame, VF_METADATA_START),
               (const char *) src +
                   offsetof(struct video_frame, VF_METADATA_START),
               VF_METADATA_SIZE);
}

void
vf_store_metadata(const struct video_frame *f, void *s)
{
        memcpy(s,
               (const char *) f +
                   offsetof(struct video_frame, VF_METADATA_START),
               VF_METADATA_SIZE);
}

void vf_restore_metadata(struct video_frame *f, void *s)
{
        memcpy((char *) f + offsetof(struct video_frame, VF_METADATA_START), s, VF_METADATA_SIZE);
}

unsigned int vf_get_data_len(struct video_frame *f)
{
        unsigned int ret = 0;
        for (unsigned int i = 0u; i < f->tile_count; ++i) {
                ret += f->tiles[i].data_len;
        }

        return ret;
}

void buf_get_planes(int width, int height, codec_t color_spec, char *data, char **planes)
{
        char *tmp = data;
        int sub[8];
        codec_get_planes_subsampling(color_spec, sub);
        for (int i = 0; i < 4; ++i) {
                if (sub[i * 2] == 0) { // less than 4 planes
                        break;
                }
                planes[i] = tmp;
                tmp += ((width + sub[i * 2] - 1) / sub[i * 2])
                        * ((height + sub[i * 2 + 1] - 1) / sub[i * 2 + 1]);
        }
}

void buf_get_linesizes(int width, codec_t color_spec, int *linesize)
{
        int sub[8];
        codec_get_planes_subsampling(color_spec, sub);
        for (int i = 0; i < 4; ++i) {
                if (sub[2 * i] == 0) {
                        break;
                }
                linesize[i] = (width + sub[2 * i] - 1) / sub[2 * i];
        }
}

void vf_clear(struct video_frame *f)
{
        clear_video_buffer((unsigned char *)f->tiles[0].data,
                        vc_get_linesize(f->tiles[0].width, f->color_spec),
                        vc_get_linesize(f->tiles[0].width, f->color_spec),
                        f->tiles[0].height,
                        f->color_spec);
}

bool parse_fps(const char *fps, struct video_desc *desc)
{
        char *endptr = NULL;
        desc->fps = strtod(fps, &endptr);
        if (desc->fps <= 0.0) {
                log_msg(LOG_LEVEL_ERROR, "FPS must be positive, got: %s!\n",
                        fps);
                return false;
        }
        desc->interlacing = PROGRESSIVE;
        if (strlen(endptr) != 0) { // optional interlacing suffix
                desc->interlacing = get_interlacing_from_suffix(endptr);
                if (desc->interlacing != PROGRESSIVE &&
                    desc->interlacing != SEGMENTED_FRAME &&
                    desc->interlacing != INTERLACED_MERGED) { // tff or bff
                        log_msg(LOG_LEVEL_ERROR,
                                "Unsupported interlacing format: %s!\n",
                                endptr);
                        return false;
                }
                if (desc->interlacing == INTERLACED_MERGED) {
                        desc->fps /= 2;
                }
        }
        return true;
}
