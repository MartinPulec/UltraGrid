// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#include "video.h"

#include <assert.h>    // for assert
#include <pthread.h>   // for pthread_create, pthread_join, pthread_t
#include <stdatomic.h> // for atomic_bool
#include <stdint.h>    // for uint32_t
#include <stdio.h>     // for snprintf
#include <stdlib.h>    // for size_t, calloc, free

#include "compat/c23.h"     // IWYU pragma: keep
#include "compat/qsort_s.h" // for qsort_s
#include "debug.h"          // for LOG_LEVEL_ERROR, MSG, debug_msg
#include "host.h"           // for register_should_exit_callback, unregister...
#include "pixfmt_conv.h"
#include "rxtx.h"           // for rxtx_medium_params, rxtx_have_receive_vid...
#include "tv.h"
#include "types.h"          // for rxtx_mode, tx_media_type, video_desc
#include "utils/macros.h"   // for to_fourcc
#include "utils/pthread.h"  // for PTHREAD_NULL
#include "utils/thread.h"   // for set_thread_name
#include "video_codec.h"    // for get_codec_name
#include "video_decompress.h"
#include "video_display.h"  // for PITCH_DEFAULT, PUTF_NONBLOCK, display_get...
#include "video_frame.h"    // for vf_copy_data_pitch, vf_copy_metadata, vf_...

#define MOD_NAME "[video] "
#define MAGIC to_fourcc('v', 'd', 'e', 'o')

struct state_video {
        uint32_t        magic;
        pthread_t       receiver_thread;
        struct rxtx    *rxtx;
        struct module  *parent;

        struct display   *display;
        codec_t           display_codecs[VIDEO_CODEC_COUNT];
        int               display_rgb_shift[3];
        size_t            display_pitch;
        long long         putf_timeout;
        struct video_desc saved_network_desc;
        codec_t           configured_display_codec;
        video_decompress *decompress;

        char  *scratchpad;
        size_t scratchpad_allocated;
};

typedef struct {
        struct pixfmt_desc desc;
        codec_t            codec;
} desc_codec_pair;

static QSORT_S_COMP_DEFINE(compare, a, b, context)
{
        const pixfmt_desc *desc   = context;
        struct pixfmt_desc desc_a = get_pixfmt_desc(*(const codec_t *) a);
        struct pixfmt_desc desc_b = get_pixfmt_desc(*(const codec_t *) b);
        return compare_pixdesc(&desc_a, &desc_b, desc);
}

static unsigned
video_decoder_order_output_codecs(pixfmt_desc     comp_int_prop,
                                  codec_t        *display_codecs,
                                  desc_codec_pair ret[static VC_COUNT])
{
        unsigned count = 0;
        bool used[VC_COUNT] = { 0 };
        // first add hw-accelerated codecs
        for (codec_t *c = display_codecs; *c; c++) {
                if (codec_is_hw_accelerated(*c)) {
                        ret[count++] = (desc_codec_pair){ comp_int_prop, *c };
                        if (comp_int_prop.depth != 0) {
                                ret[count++] =
                                    (desc_codec_pair){ (pixfmt_desc){ 0 }, *c };
                        }
                        used[*c] = true;
                }
        }
        // then codecs matching exactly internal codec
        for (codec_t *c = display_codecs; *c; c++) {
                if (used[*c]) {
                        continue;
                };
                if (pixdesc_equals(get_pixfmt_desc(*c), comp_int_prop)) {
                        ret[count++] = (desc_codec_pair){ comp_int_prop, *c };
                        if (comp_int_prop.depth != 0) {
                                ret[count++] =
                                    (desc_codec_pair){ (pixfmt_desc){ 0 }, *c };
                        }
                        used[*c] = true;
                }
        }
        // then add also all other codecs
        codec_t remaining[VC_COUNT];
        unsigned rem_count = 0;
        for (codec_t *c = display_codecs; *c; c++) {
                if (used[*c]) {
                        continue;
                }
                remaining[rem_count++] = *c;
        }
        remaining[rem_count] = VC_NONE;
        if (comp_int_prop.depth != 0) {
                qsort_s(remaining, rem_count, sizeof remaining[0], compare, &comp_int_prop);
        }
        for (codec_t *c = remaining; *c; c++) {
                ret[count++] = (desc_codec_pair){ comp_int_prop, *c };
                if (comp_int_prop.depth != 0) {
                        ret[count++] = (desc_codec_pair){ (pixfmt_desc){0 }, *c };
                }
        }
        ret[count] = (desc_codec_pair){ (pixfmt_desc){ 0 }, VC_NONE };

        if (log_level >= LOG_LEVEL_VERBOSE) {
                MSG(VERBOSE, "Trying codecs in this order:\n");
                for (unsigned i = 0; i < count; ++i) {
                        MSG(VERBOSE, "\t%s, internal: %s\n",
                            get_codec_name(ret[i].codec),
                            get_pixdesc_desc(ret[i].desc));
                }
        }

        return count;
}

#define codec_list_to_string(cl) codec_list_to_str(cl, STR_LEN, (char[1024]){})

static bool
init_decompress(struct state_video *s, struct video_desc desc,
                struct pixfmt_desc comp_int_prop, codec_t *out_codec)
{
        decompress_done(s->decompress);
        s->decompress = nullptr;

        bool probe = comp_int_prop.depth == 0;
        if (probe) {
                video_decompress *d = decompress_init(
                    desc.color_spec, (struct pixfmt_desc){ 0 }, VIDEO_CODEC_NONE,
                    desc.tile_count);
                if (d) {
                        int buf_size  = decompress_reconfigure(
                            d, desc, s->display_rgb_shift[0],
                            s->display_rgb_shift[1], s->display_rgb_shift[2],
                            VC_NONE);
                        if (!buf_size) {
                                MSG(ERROR, "Cannot reconfigure for probe!\n");
                                decompress_done(d);
                                return false;
                        }
                        s->decompress = d;
                        *out_codec    = VC_NONE;
                        return true;
                }
                MSG(VERBOSE, "Auto-detection not supported!\n");
        }
        desc_codec_pair formats_to_try[VC_COUNT];
        const unsigned  count = video_decoder_order_output_codecs(
            comp_int_prop, s->display_codecs, formats_to_try);

        for (unsigned i = 0; i < count; i++) {
                video_decompress *d =
                    decompress_init(desc.color_spec, formats_to_try[i].desc,
                                    formats_to_try[i].codec, desc.tile_count);
                if (d) {
                        int buf_size = decompress_reconfigure(
                            d, desc, s->display_rgb_shift[0],
                            s->display_rgb_shift[1], s->display_rgb_shift[2],
                            formats_to_try[i].codec);
                        if (!buf_size) {
                                MSG(ERROR, "Cannot reconfigure for probe!\n");
                                decompress_done(d);
                                return false;
                        }
                        s->decompress = d;
                        *out_codec = formats_to_try[i].codec;
                        return true;
                }
        }

        MSG(ERROR, "Unable to find decoder for input codec \"%s\"!!!\n",
            get_codec_name(desc.color_spec));
        MSG(ERROR,
            "Display doesn't support video codec %s! Supported codecs: %s\n",
            get_codec_name(desc.color_spec),
            codec_list_to_string(s->display_codecs));
        MSG(INFO,
            "Compression internal codec is \"%s\". Native codecs are: "
            "%s\n",
            get_pixdesc_desc(comp_int_prop),
            codec_list_to_string(s->display_codecs));
        MSG(ERROR,
            "Could not find neither line conversion nor decompress "
            "from %s to display supported formats (%s).\n",
            get_codec_name(desc.color_spec),
            codec_list_to_string(s->display_codecs));
        return false;
}

static bool
my_display_reconfigure(struct state_video *s, struct video_desc network_desc,
                       codec_t display_codec)
{
        struct video_desc display_desc = network_desc;
        display_desc.color_spec        = display_codec;
        if (!display_reconfigure(s->display, display_desc)) {
                MSG(ERROR, "Cannot reconfigure display!\n");
                return false;
        }
        int    display_requested_pitch = PITCH_DEFAULT;
        size_t len                     = sizeof display_requested_pitch;
        bool ret = display_ctl_property(s->display, DISPLAY_PROPERTY_BUF_PITCH,
                                        &display_requested_pitch, &len);
        if (!ret) {
                debug_msg("Failed to get pitch from video driver.\n");
                display_requested_pitch = PITCH_DEFAULT;
        }
        if (display_requested_pitch == PITCH_DEFAULT) {
                s->display_pitch = vc_get_linesize(display_desc.width,
                                                   display_desc.color_spec);
        } else {
                s->display_pitch = display_requested_pitch;
        }
        s->saved_network_desc = network_desc;
        return true;
}

static bool
recv_reconfigure(struct state_video *s, struct video_desc desc)
{
        decompress_done(s->decompress);
        s->decompress = nullptr;

        if (codec_is_in_set(desc.color_spec, s->display_codecs)) {
                s->configured_display_codec = desc.color_spec;
        } else {
                codec_t dec_codec = VC_NONE;
                if (!init_decompress(s, desc, (struct pixfmt_desc){ 0 },
                                     &dec_codec)) {
                        return false;
                }
                if (!dec_codec) { // probing now
                        s->saved_network_desc = desc;
                        return true;
                }
                s->configured_display_codec = dec_codec;
        }

        return my_display_reconfigure(s, desc, s->configured_display_codec);
}

static bool
decompress(struct state_video *s, const struct video_frame *recv_frame,
           struct video_frame **display_frame_inout)
{
        struct video_frame *display_frame = *display_frame_inout;
        for (unsigned i = 0; i < recv_frame->tile_count; ++i) {
                size_t max_len = ((size_t) recv_frame->tiles[i].width *
                                  recv_frame->tiles[i].height * MAX_BPS) +
                                 MAX_PADDING;
                char *buf = nullptr;
                struct video_frame_callbacks *clbcks = nullptr;
                if (!display_frame) {
                        if (s->scratchpad_allocated < max_len) {
                                free(s->scratchpad);
                                s->scratchpad_allocated = max_len;
                        }
                        buf = s->scratchpad = malloc(s->scratchpad_allocated);
                } else {
                        /// @todo implementovat dekodovani tiled-4k do merged
                        assert(display_frame->tile_count ==
                               recv_frame->tile_count);
                        buf = display_frame->tiles[i].data;
                        clbcks = &display_frame->callbacks;
                }

                struct pixfmt_desc comp_desc = { 0 };
                decompress_status  ret  = decompress_frame(
                    s->decompress->state[i], (unsigned char *) buf,
                    (unsigned char *) recv_frame->tiles[i].data,
                    recv_frame->tiles[i].data_len, recv_frame->seq, clbcks,
                    &comp_desc, s->display_pitch);
                switch (ret) {
                case DECODER_NO_FRAME: 
                        return false;
                case DECODER_UNSUPP_PIXFMT:
                        codec_list_erase(s->display_codecs,
                                         s->configured_display_codec);
                        s->saved_network_desc = (struct video_desc){ 0 };
                        return false;
                case DECODER_GOT_CODEC: {
                        MSG(NOTICE, "Detected compression properties: %s\n",
                            get_pixdesc_desc(comp_desc));
                        codec_t dec_codec = VC_NONE;
                        if (!init_decompress(s, s->saved_network_desc,
                                             comp_desc, &dec_codec)) {
                                s->saved_network_desc =
                                    (struct video_desc){ 0 };
                                return false;
                        }
                        if (!dec_codec) {
                                MSG(FATAL, "Decompress didn't return output codec!\n");
                                abort();
                        }
                        if (!my_display_reconfigure(s, s->saved_network_desc,
                                                    dec_codec)) {
                                s->saved_network_desc =
                                    (struct video_desc){ 0 };
                                return false;
                        }
                        *display_frame_inout = display_frame =
                            display_get_frame(s->display);
                        // do the actual decode - restart loop
                        i = -1;
                        continue;
                }
                case DECODER_GOT_FRAME:
                        continue;
                }
                abort(); // ret not part of enum
        }
        return true;
}

static void *
video_receiver_thread(void *arg)
{
        set_thread_name(__func__);
        struct state_video *s = arg;

        struct video_frame *display_frame = nullptr;

        while (true) {
                struct video_frame *recv_frame = s->decompress ? nullptr : display_frame;
                struct video_frame *ret =
                    rxtx_recv_video_frame(s->rxtx, recv_frame, s->display_pitch);
                if (!ret) {
                        break;
                }
                if (ret == rxtx_retry) {
                        continue;
                }
                if (!video_desc_eq(video_desc_from_frame(ret),
                                   s->saved_network_desc)) {
                        if (display_frame) {
                                display_put_frame(s->display, display_frame,
                                                  PUTF_DISCARD);
                                display_frame = nullptr;
                        }
                        if (!recv_reconfigure(s, video_desc_from_frame(ret))) {
                                VIDEO_FRAME_DISPOSE(ret);
                                continue;
                        }
                }

                if (s->decompress) {
                        if (!decompress(s, ret, &display_frame)) {
                                VIDEO_FRAME_DISPOSE(ret);
                                continue;
                        }
                } else {
                        if (!display_frame) {
                                display_frame = display_get_frame(s->display);
                                assert(display_frame);
                        }

                        if (ret != display_frame) {
                                vf_copy_metadata(display_frame, ret);
                                vf_copy_data_pitch(display_frame,
                                                   s->display_pitch, ret);
                        }
                }
                if (ret != display_frame) {
                        VIDEO_FRAME_DISPOSE(ret);
                }

                display_put_frame(s->display, display_frame, s->putf_timeout);
                display_frame = display_get_frame(s->display);
        }

        if (display_frame) {
                display_put_frame(s->display, display_frame, PUTF_DISCARD);
        }

        // pass poisoned pill to display
        display_put_frame(s->display, nullptr, PUTF_BLOCKING);
        decompress_done(s->decompress);

        return nullptr;
}

ADD_TO_PARAM("decoder-drop-policy",
                "* decoder-drop-policy=blocking|nonblock|<sec>\n"
                "  Force specified blocking policy (default nonblock).\n"
                "  <sec> - specifies frame timeout in seconds (can have suffixes, eg. \"20ms\")\n");
static bool
recv_config(struct state_video *s)
{
        size_t len = sizeof s->display_codecs;
        if (!display_ctl_property(s->display, DISPLAY_PROPERTY_CODECS,
                                  s->display_codecs, &len)) {
                MSG(ERROR, "Failed to query codecs from video display.\n");
                return false;
        }
        len = sizeof s->display_rgb_shift;
        if (!display_ctl_property(s->display, DISPLAY_PROPERTY_RGB_SHIFT,
                                   &s->display_rgb_shift, &len)) {
                debug_msg(
                    "Failed to get r,g,b shift property from video driver.\n");
                int rgb_shift[] = DEFAULT_RGB_SHIFT_INIT;
                memcpy(&s->display_rgb_shift, rgb_shift,
                       sizeof rgb_shift);
        }

        const char *drop_policy = get_commandline_param("decoder-drop-policy");
        if (drop_policy == nullptr) {
                drop_policy = "nonblock";
        }
        if (strcmp(drop_policy, "nonblock") == 0) {
                s->putf_timeout = PUTF_NONBLOCK;
        } else if (strcmp(drop_policy, "blocking") == 0) {
                s->putf_timeout = PUTF_BLOCKING;
        } else {
                s->putf_timeout =
                    SEC_TO_NS(unit_evaluate_dbl(drop_policy, true, nullptr));
        }
        return true;
}

struct state_video *
video_start(struct rxtx *rxtx, const struct rxtx_params *params,
            struct module *parent, struct display *d)
{
        struct state_video *s = calloc(1, sizeof *s);
        s->magic              = MAGIC;
        s->receiver_thread    = PTHREAD_NULL;
        s->rxtx               = rxtx;
        s->display            = d;
        s->parent             = parent;

        if (params->medium[TX_MEDIA_VIDEO].rxtx_mode & MODE_RECEIVER &&
            rxtx_have_receive_video_frame(rxtx)) {
                if (!recv_config(s)) {
                        video_done(s);
                        return nullptr;
                }
                pthread_create(&s->receiver_thread, nullptr, video_receiver_thread, s);
        }

#if 0
        /// @todo set
        size_t len = sizeof s->rtp_common->display_supp_for_mult_sources;
        display_ctl_property(
            s->display_device, DISPLAY_PROPERTY_SUPPORTS_MULTI_SOURCES,
            &s->rtp_common->display_supp_for_mult_sources, &len);
#endif


        return s;
}

void
video_join(struct state_video *s)
{
        if (s == nullptr) {
                return;
        }
        if (s->receiver_thread != PTHREAD_NULL) {
                pthread_join(s->receiver_thread, nullptr);
        }
}

void
video_done(struct state_video *s)
{
        if (!s) {
                return;
        }
        assert(s->magic == MAGIC);
        free(s->scratchpad);
        free(s);
}
