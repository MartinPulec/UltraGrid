// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2016-2026 CESNET, zájmové sdružení právických osob

/**
 * @file
 * mostly adapted from rtp/video_decoders
 */

#include <stdlib.h> // for free, calloc, malloc
#include <string.h> // for strcmp, strlen

#include "capture_filter.h"  // for CAPTURE_FILTER_ABI_VERSION, capture_fil...
#include "compat/c23.h"      // for countof
#include "debug.h"           // for LOG_LEVEL_ERROR, MSG, UNUSED
#include "lib_common.h"      // for REGISTER_MODULE, library_class
#include "types.h"           // for interlacing, tile, video_frame, video_desc
#include "utils/color_out.h" // for color_printf, TBOLD, TRED
#include "video_codec.h"     // for vc_get_linesize
#include "video_frame.h"     // for il_lower_to_merged, il_merged_to_upper
struct module;

#define MOD_NAME "[vcf/change_il] "

static int  init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

typedef void (*change_il_fn)(char *dst, char *src, int linesize, int height,
                             void **state);

struct state_change_il {
        enum interlacing dst_il_mode;

        void            *change_il_state;
        enum interlacing saved_il_mode;
        change_il_fn     change_il;
};

struct transcode_fn {
        enum interlacing in;
        enum interlacing out;
        change_il_fn     func;
};

static void
usage()
{
        color_printf("Changes interlacing mode of incoming frame.\n");
        color_printf("Note: just subset of conversion is supported.\n");
        color_printf("\n");

        color_printf("Usage:\n");
        color_printf("\t" TBOLD(TRED("-F change_il") ":<il_suffix>")
                     " -t <vidcap> <receiver>\n");
        color_printf("\n");

        color_printf("Available suffixes:\n");
        for (unsigned i = 0; i < INTERLACING_COUNT; ++i) {
                color_printf("\t- " TBOLD("%s")
                             " - %s\n",
                             get_interlacing_suffix(i),
                             get_interlacing_description(i));
        }
        color_printf("\n");

        color_printf("Available conversions:\n");
        for (unsigned from = 0; from < INTERLACING_COUNT; ++from) {
                for (unsigned to = 0; to < INTERLACING_COUNT; ++to) {
                        if (get_change_il_fn(from, to)) {
                                color_printf("\t- " TBOLD("%s->%s")
                                             "\n",
                                             get_interlacing_suffix(from),
                                             get_interlacing_suffix(to));
                        }
                }
        }
        color_printf("\n");
}

static int
init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);
        if (strlen(cfg) == 0 || !strcmp(cfg, "help")) {
                usage();
                return strcmp(cfg, "help") == 0 ? 1 : -1;
        }
        enum interlacing il = get_interlacing_from_suffix(cfg);
        if (il == INTERLACING_COUNT) {
                MSG(ERROR, "Unknown interlacing suffix: %s\n", cfg);
                return -1;
        }

        struct state_change_il *s = calloc(1, sizeof(struct state_change_il));
        s->dst_il_mode            = il;
        *state                    = s;
        return 0;
}

static void
done(void *state)
{
        struct state_change_il *s = state;
        free(s->change_il_state);
        free(state);
}

static struct video_frame *
filter(void *state, struct video_frame *in)
{
        if (in == nullptr) {
                return nullptr;
        }
        struct state_change_il *s = state;

        if (in->interlacing == s->dst_il_mode) {
                return in;
        }

        if (s->saved_il_mode != in->interlacing) {
                free(s->change_il_state);
                s->change_il_state = nullptr;
                s->change_il =
                    get_change_il_fn(in->interlacing, s->dst_il_mode);
                if (!s->change_il) {
                        MSG(ERROR,
                            "Cannot find interlacing conversion fn from %s to "
                            "%s\n",
                            get_interlacing_description(in->interlacing),
                            get_interlacing_description(s->dst_il_mode));
                        VIDEO_FRAME_DISPOSE(in);
                        return nullptr;
                }
                s->saved_il_mode = in->interlacing;
        }

        struct video_desc desc  = video_desc_from_frame(in);
        desc.interlacing        = s->dst_il_mode;
        struct video_frame *out = vf_alloc_desc_data(desc);
        out->callbacks.dispose  = vf_free;

        for (unsigned i = 0; i < in->tile_count; ++i) {
                s->change_il(
                    out->tiles[i].data, in->tiles[i].data,
                    vc_get_linesize(in->tiles[i].width, in->color_spec),
                    (int) in->tiles[i].height, &s->change_il_state);
        }

        VIDEO_FRAME_DISPOSE(in);

        return out;
}

static const struct capture_filter_info capture_filter_change_il = {
        .init   = init,
        .done   = done,
        .filter = filter,
};

REGISTER_MODULE(change_il, &capture_filter_change_il,
                LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
