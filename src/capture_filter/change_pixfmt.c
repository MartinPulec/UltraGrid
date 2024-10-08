/**
 * @file   capture_filter/change_pixfmt.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020-2024 CESNET
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

#include <stdbool.h>                                // for false, bool, true
#include <stdint.h>                                 // for uint32_t
#include <stdio.h>                                  // for printf
#include <stdlib.h>                                 // for free, NULL, calloc
#include <string.h>                                 // for memcpy, strcmp

#include "capture_filter.h"                         // for CAPTURE_FILTER_AB...
#include "debug.h"                                  // for log_msg, LOG_LEVE...
#include "lib_common.h"                             // for REGISTER_MODULE
#include "pixfmt_conv.h"                            // for get_decoder_from_to
#include "types.h"                                  // for tile, video_frame
#include "utils/color_out.h"                        // for color_printf, TER...
#include "utils/macros.h"                           // for to_fourcc
#include "video_codec.h"                            // for vc_get_linesize
#include "video_frame.h"                            // for vf_alloc_desc
#include "vo_postprocess.h"                         // for VO_PP_PROPERTY_CO...
#include "vo_postprocess/capture_filter_wrapper.h"  // for ADD_VO_PP_CAPTURE...
struct module;

#define MAGIC to_fourcc('C', 'F', 'C', 'P')
#define MOD_NAME "[change pixfmt cap. f.] "

struct state_capture_filter_change_pixfmt {
        uint32_t magic;
        codec_t to_codec;
        void *vo_pp_out_buffer; ///< buffer to write to if we use vo_pp wrapper (otherwise unused)
};

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        if (strlen(cfg) == 0 || strcmp(cfg, "help") == 0) {
                printf("Performs pixel format change transformation.\n\n"
                       "usage:\n");
                color_printf(TERM_FG_RED "\t--capture-filter change_pixfmt:<name>\n" TERM_FG_RESET);
                return 1;
        }

        struct state_capture_filter_change_pixfmt *s = calloc(1, sizeof(struct state_capture_filter_change_pixfmt));
        s->magic = MAGIC;
        s->to_codec = get_codec_from_name(cfg);
        if (!s->to_codec) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong codec: %s\n", cfg);
                free(s);
                return -1;
        }

        *state = s;
        return 0;
}

static void done(void *state)
{
        free(state);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_capture_filter_change_pixfmt *s = state;
        struct video_desc desc = video_desc_from_frame(in);
        desc.color_spec = s->to_codec;
        decoder_t decoder = get_decoder_from_to(in->color_spec, s->to_codec);

        if (!decoder) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to find decoder!\n");
                return NULL;
        }
        struct video_frame *out = vf_alloc_desc(desc);
        if (s->vo_pp_out_buffer) {
                out->tiles[0].data = s->vo_pp_out_buffer;
        } else {
                out->tiles[0].data = malloc(out->tiles[0].data_len + MAX_PADDING);
                out->callbacks.data_deleter = vf_data_deleter;
        }
        out->callbacks.dispose = vf_free;

        unsigned char *in_data = (unsigned char *) in->tiles[0].data;
        unsigned char *out_data = (unsigned char *) out->tiles[0].data;
        int src_linesize = vc_get_linesize(in->tiles[0].width, in->color_spec);
        int dst_linesize = vc_get_linesize(in->tiles[0].width, s->to_codec);

        for (unsigned int i = 0; i < in->tiles[0].height; i += 1) {
                decoder(out_data, in_data, dst_linesize, DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
                in_data += src_linesize;
                out_data += dst_linesize;
        }

        VIDEO_FRAME_DISPOSE(in);

        return out;
}


static void vo_pp_set_out_buffer(void *state, char *buffer)
{
        struct state_capture_filter_change_pixfmt *s = state;
        s->vo_pp_out_buffer = buffer;
}

static const struct capture_filter_info capture_filter_change_pixfmt = {
        .init = init,
        .done = done,
        .filter = filter,
};

static bool
change_pixfmt_vo_pp_get_property(void *state, int property, void *val, size_t *len)
{
        if (property != VO_PP_PROPERTY_CODECS) {
                return false;
        }

        struct state_capture_filter_change_pixfmt *s = state;
        const size_t alloc_len = *len;
        *len = 0;
        for (codec_t c = VC_FIRST; c < VC_END; ++c)  {
                decoder_t decoder = get_decoder_from_to(c, s->to_codec);
                if (decoder == NULL) {
                        continue;
                }
                if (*len + sizeof c > alloc_len) {
                        MSG(ERROR, "Insufficient size %zd for get_property!\n",
                            alloc_len);
                        return false;
                }
                memcpy((char*) val + *len, &c, sizeof c);
                *len += sizeof c;
        }
        return true;
}

// coverity[leaked_storage:SUPPRESS]
ADD_VO_PP_CAPTURE_FILTER_WRAPPER(change_pixfmt, init, filter, done,
                                 vo_pp_set_out_buffer,
                                 change_pixfmt_vo_pp_get_property)

REGISTER_MODULE(change_pixfmt, &capture_filter_change_pixfmt, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);


/* vim: set expandtab sw=8: */
