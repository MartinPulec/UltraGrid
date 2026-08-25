// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#include "video_recv.h"

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
#include "rxtx.h" // for rxtx_medium_params, rxtx_have_receive_vid...
#include "tv.h"
#include "types.h"         // for rxtx_mode, tx_media_type, video_desc
#include "utils/macros.h"  // for to_fourcc
#include "utils/pthread.h" // for PTHREAD_NULL
#include "utils/thread.h"  // for set_thread_name
#include "utils/video.h"
#include "utils/worker.h"
#include "video_codec.h" // for get_codec_name
#include "video_decompress.h"
#include "video_display.h" // for PITCH_DEFAULT, PUTF_NONBLOCK, display_get...
#include "video_frame.h"   // for vf_copy_data_pitch, vf_copy_metadata, vf_...

#define MOD_NAME "[vrecv] "
#define MAGIC    to_fourcc('v', 'd', 'r', 'x')

static const struct video_frame thread_poison_pill;

struct state_video_recv {
        uint32_t       magic;
        pthread_t      vid_recv_thread_id;
        struct rxtx   *rxtx;
        struct module *parent;

        struct display       *display;
        struct display_params display_params;
        size_t                display_pitch;
        long long             putf_timeout;
        struct video_desc     saved_network_desc;
        enum video_mode       video_mode;
        bool                  merged_fb;

        // either line decoder or decompress
        pthread_t           decode_thread_id;
        pthread_cond_t      decode_thread_new_frame_ready;
        pthread_cond_t      decode_thread_frame_consumed;
        pthread_mutex_t     decode_thread_lock;
        struct video_frame *decode_thread_frame;
        unsigned long long  decode_dropped_frames;
        bool                decompress_accepts_corrupted;
        decoder_t           decode_line;
};

typedef struct {
        struct state_video_recv *s;
        video_decompress        *decompress;
        codec_t                  dec_codec;
        codec_t                  configured_display_codec;
        unsigned char           *scratchpad;
        size_t                   scratchpad_allocated;
} decompress_thread_data;

typedef struct {
        struct pixfmt_desc desc;
        codec_t            codec;
} desc_codec_pair;

static bool decompress(decompress_thread_data   *d,
                       const struct video_frame *recv_frame,
                       struct video_frame       *display_frame,
                       struct pixfmt_desc       *comp_desc);
static bool vrcv_display_reconfigure(struct state_video_recv *s,
                                     struct video_desc        desc);

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
        unsigned count          = 0;
        bool     used[VC_COUNT] = { false };
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
        codec_t  remaining[VC_COUNT];
        unsigned rem_count = 0;
        for (codec_t *c = display_codecs; *c; c++) {
                if (used[*c]) {
                        continue;
                }
                remaining[rem_count++] = *c;
        }
        remaining[rem_count] = VC_NONE;
        if (comp_int_prop.depth != 0) {
                qsort_s(remaining, rem_count, sizeof remaining[0], compare,
                        &comp_int_prop);
        }
        for (codec_t *c = remaining; *c; c++) {
                ret[count++] = (desc_codec_pair){ comp_int_prop, *c };
                if (comp_int_prop.depth != 0) {
                        ret[count++] =
                            (desc_codec_pair){ (pixfmt_desc){ 0 }, *c };
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

/// @returns whether writting to merged fb
static bool
adjust_desc_video_mode_for_tiles(const struct state_video_recv *s,
                                 struct video_desc             *desc,
                                 enum video_mode               *video_mode)
{
        *video_mode      = guess_video_mode(desc->tile_count);
        bool stereo_recv = desc->tile_count == 2;
        bool merged_fb =
            s->display_params.display_mode == DISPLAY_PROPERTY_VIDEO_MERGED ||
            (s->display_params.display_mode ==
                 DISPLAY_PROPERTY_VIDEO_SEPARATE_3D &&
             !stereo_recv);
        if (!merged_fb) {
                return false;
        }
        desc->width *= get_video_mode_tiles_x(*video_mode);
        desc->height *= get_video_mode_tiles_y(*video_mode);
        desc->tile_count = 1;
        return true;
}

static void *
line_decoder_thread(void *arg)
{
        set_thread_name(__func__);
        struct state_video_recv *s = arg;
        struct video_frame      *f = nullptr;

        while (true) {
                VIDEO_FRAME_DISPOSE(f);

                CHK_PTHR(pthread_mutex_lock(&s->decode_thread_lock));
                {
                        while (!s->decode_thread_frame) {
                                CHK_PTHR(pthread_cond_wait(
                                    &s->decode_thread_new_frame_ready,
                                    &s->decode_thread_lock));
                        }
                        f                      = s->decode_thread_frame;
                        s->decode_thread_frame = nullptr;
                }
                CHK_PTHR(pthread_mutex_unlock(&s->decode_thread_lock));
                CHK_PTHR(pthread_cond_signal(&s->decode_thread_frame_consumed));

                if (f == &thread_poison_pill) {
                        f = nullptr;
                        break;
                }

                struct video_frame *display_frame =
                    display_get_frame(s->display);

                int *restrict shifts = s->display_params.rgb_shift;
                for (unsigned pos = 0; pos < f->tile_count; ++pos) {
                        struct tile *s_tile = vf_get_tile(f, pos);
                        int d_tile_idx      = s->merged_fb ? 0 : f->tile_count;
                        struct tile *d_tile =
                            vf_get_tile(display_frame, d_tile_idx);
                        size_t tile_pos_x =
                            pos % get_video_mode_tiles_x(s->video_mode);
                        size_t tile_pos_y =
                            pos / get_video_mode_tiles_x(s->video_mode);
                        size_t dst_buf_offset =
                            (tile_pos_y * s->saved_network_desc.height *
                             s->display_pitch) +
                            vc_get_linesize(tile_pos_x *
                                                s->saved_network_desc.width,
                                            display_frame->color_spec);
                        size_t src_linesize =
                            vc_get_linesize(s_tile->width, f->color_spec);
                        size_t dst_linesize = vc_get_linesize(
                            s_tile->width, display_frame->color_spec);

                        char       *src     = f->tiles[pos].data;
                        const char *src_end = src + s_tile->data_len;
                        char       *dst     = d_tile->data + dst_buf_offset;
                        while (src < src_end) {
                                s->decode_line((unsigned char *) dst,
                                               (unsigned char *) src,
                                               dst_linesize, shifts[0],
                                               shifts[1], shifts[2]);
                                src += src_linesize;
                                dst += s->display_pitch;
                        }
                }
                display_frame->ssrc      = f->ssrc;
                display_frame->timestamp = f->timestamp;

                display_put_frame(s->display, display_frame, s->putf_timeout);
        }

        return nullptr;
}

/**
 * mimics choose_codec_and_decoder() (rtp/video_decoders)
 * @note
 * The properties of DXTn do not exactly match - bpp is 0.5, but line (actually
 * 4 lines) is (2 * width) long, so it makes troubles when using line decoder
 * and tiles. So the fallback is external decoder. The DXT compression is
 * exceptional in that, that it can be both internally and externally
 * decompressed.
 */
static bool
setup_line_decoder(struct state_video_recv *s, struct video_desc desc)
{
        codec_t found_codec = VC_NONE;
        for (const codec_t *it = s->display_params.native_codecs;
             !found_codec && *it != VC_NONE; it++) {
                if (desc.color_spec != *it) {
                        continue;
                }
                if ((desc.color_spec == DXT1 || desc.color_spec == DXT1_YUV ||
                     desc.color_spec == DXT5) &&
                    desc.tile_count != 1) {
                        continue; /// DXT it is an exception, see note
                                  /// above
                }
                s->decode_line = vc_memcpy;
                /* another exception - we may change shifts */
                if (desc.color_spec == RGBA) {
                        s->decode_line = get_decoder_from_to(desc.color_spec,
                                                             desc.color_spec);
                }
                found_codec = *it;
        }
        // if codec doesen't match, try to find line decoder
        for (const codec_t *it = s->display_params.native_codecs;
             !found_codec && *it != VC_NONE; it++) {
                s->decode_line = get_decoder_from_to(desc.color_spec, *it);
                if (s->decode_line) {
                        found_codec = *it;
                }
        }
        if (!found_codec) { // no eligible line decoder was found
                return false;
        }
        desc.color_spec = found_codec;
        s->merged_fb =
            adjust_desc_video_mode_for_tiles(s, &desc, &s->video_mode);

        bool ret = vrcv_display_reconfigure(s, desc);
        if (!ret) {
                return false;
        }
        CHK_PTHR(pthread_create(&s->decode_thread_id, nullptr,
                                line_decoder_thread, s));
        assert(s->decode_thread_id != PTHREAD_NULL);
        return true;
}

static video_decompress *
decompress_init_reconfigure(struct state_video_recv *s, struct video_desc desc,
                            struct pixfmt_desc int_fmt, codec_t out_codec)
{
        video_decompress *d = decompress_init(desc.color_spec, int_fmt,
                                              out_codec, (int) desc.tile_count);
        if (!d) {
                return nullptr;
        }
        int *shift    = s->display_params.rgb_shift;
        int  buf_size = decompress_reconfigure(d, desc, shift[0], shift[1],
                                               shift[2], out_codec);
        if (!buf_size) {
                MSG(ERROR, "Cannot reconfigure decompress%s!\n",
                    out_codec == VC_NONE ? " for probe" : "");
                decompress_done(d);
                return nullptr;
        }

        int    accepts = 0;
        size_t size    = sizeof(accepts);
        int    ret     = decompress_get_property(
            d, DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME, &accepts, &size);
        s->decompress_accepts_corrupted = ret && accepts;
        MSG(VERBOSE, "Decoder accepts corrupted frames: %d\n",
            (int) s->decompress_accepts_corrupted);
        return d;
}

#define codec_list_to_string(cl) codec_list_to_str(cl, STR_LEN, (char[1024]){})

static video_decompress *
init_decompress(struct state_video_recv *s, struct video_desc desc,
                struct pixfmt_desc comp_int_prop, codec_t *out_codec,
                bool probe)
{
        if (probe) {
                video_decompress *d = decompress_init_reconfigure(
                    s, desc, (struct pixfmt_desc){ 0 }, VC_NONE);
                if (d) {
                        *out_codec = VC_NONE;
                        return d;
                }
                MSG(VERBOSE, "Auto-detection not supported!\n");
        }
        desc_codec_pair formats_to_try[VC_COUNT];
        const unsigned  codec_count = video_decoder_order_output_codecs(
            comp_int_prop, s->display_params.native_codecs, formats_to_try);

        for (unsigned i = 0; i < codec_count; i++) {
                video_decompress *d = decompress_init_reconfigure(
                    s, desc, formats_to_try[i].desc, formats_to_try[i].codec);
                if (!d) { // try next
                        continue;
                }
                *out_codec = formats_to_try[i].codec;
                return d;
        }

        MSG(INFO,
            "Compression internal codec is \"%s\". Native codecs are: "
            "%s\n",
            get_pixdesc_desc(comp_int_prop),
            codec_list_to_string(s->display_params.native_codecs));

        return nullptr;
}

/**
 * called after compression format is probed (or probe unsupported)
 * @returns true if dec reinit for given properties succeeds
 */
static bool
decompress_display_codec_format_probed(decompress_thread_data   *d,
                                       const struct pixfmt_desc *comp_desc)
{
        struct state_video_recv *s         = d->s;
        codec_t                  dec_codec = VC_NONE;
        video_decompress        *new_dec   = init_decompress(
            s, s->saved_network_desc, *comp_desc, &dec_codec, false);
        if (!new_dec) {
                return false;
        }
        decompress_done(d->decompress);
        d->decompress = new_dec;
        if (!dec_codec) {
                MSG(FATAL, "Decompress didn't return output codec!\n");
                abort();
        }
        struct video_desc desc = s->saved_network_desc;
        desc.color_spec        = dec_codec;

        s->merged_fb =
            adjust_desc_video_mode_for_tiles(s, &desc, &s->video_mode);

        if (!vrcv_display_reconfigure(s, desc)) {
                MSG(ERROR, "Cannot reconfigure display for decompress!\n");
                return false;
        }
        d->configured_display_codec = dec_codec;
        return true;
}

static void *
decompress_thread(void *arg)
{
        set_thread_name(__func__);
        decompress_thread_data  *d             = arg;
        struct state_video_recv *s             = d->s;
        struct video_frame      *f             = nullptr;
        struct video_frame      *display_frame = nullptr;

        while (true) {
                VIDEO_FRAME_DISPOSE(f);
                CHK_PTHR(pthread_mutex_lock(&s->decode_thread_lock));
                {
                        while (!s->decode_thread_frame) {
                                CHK_PTHR(pthread_cond_wait(
                                    &s->decode_thread_new_frame_ready,
                                    &s->decode_thread_lock));
                        }
                        f                      = s->decode_thread_frame;
                        s->decode_thread_frame = nullptr;
                }
                CHK_PTHR(pthread_mutex_unlock(&s->decode_thread_lock));
                CHK_PTHR(pthread_cond_signal(&s->decode_thread_frame_consumed));

                if (f == &thread_poison_pill) {
                        f = nullptr;
                        break;
                }
                if ((f->flags & FRM_FLG_CORRUPTED) &&
                    !s->decompress_accepts_corrupted) {
                        continue;
                }

                struct pixfmt_desc comp_desc = { 0 };
                bool ret = decompress(d, f, display_frame, &comp_desc);
                if (!ret) {
                        continue;
                }
                if (!display_frame) { // not yet configured until probed
                        if (!decompress_display_codec_format_probed(
                                d, &comp_desc)) {
                                continue;
                        }
                        display_frame = display_get_frame(s->display);
                        if (!decompress(d, f, display_frame, &comp_desc)) {
                                continue;
                        }
                }
                display_put_frame(s->display, display_frame, s->putf_timeout);
                display_frame = display_get_frame(s->display);
        }

        if (display_frame) {
                display_put_frame(s->display, display_frame, PUTF_DISCARD);
        }

        decompress_done(d->decompress);
        free(d->scratchpad);
        free(d);

        return nullptr;
}

static bool
vrcv_display_reconfigure(struct state_video_recv *s, struct video_desc desc)
{
        if (!display_reconfigure(s->display, desc)) {
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
                s->display_pitch = vc_get_linesize(desc.width, desc.color_spec);
        } else {
                s->display_pitch = display_requested_pitch;
        }
        return true;
}

static void
start_decompress_thread(struct state_video_recv *s, video_decompress *d,
                        codec_t dec_codec)
{
        decompress_thread_data *thr_data = malloc(sizeof *thr_data);
        *thr_data = (decompress_thread_data){ .decompress = d,
                                              .s          = s,
                                              .dec_codec  = dec_codec };
        CHK_PTHR(pthread_create(&s->decode_thread_id, nullptr,
                                decompress_thread, thr_data));
        assert(s->decode_thread_id != PTHREAD_NULL);
}

/// @returns whether display wants native rgb shifts or we need to use a line
/// decoder
static bool
disp_rgba_has_native_shifts(const int rgb_shift[const static 3])
{
        return rgb_shift[0] == DEFAULT_R_SHIFT &&
               rgb_shift[1] == DEFAULT_G_SHIFT &&
               rgb_shift[2] == DEFAULT_B_SHIFT;
}

static bool
recv_reconfigure(struct state_video_recv *s, struct video_desc desc)
{
        s->saved_network_desc = desc;
        // natively supported
        if (codec_is_in_set(desc.color_spec, s->display_params.native_codecs) &&
            (desc.color_spec != RGBA ||
             disp_rgba_has_native_shifts(s->display_params.rgb_shift))) {
                struct video_desc desc_copy = desc;
                bool merged_fb =
                    adjust_desc_video_mode_for_tiles(s, &desc_copy, &s->video_mode);
                if (!merged_fb || s->video_mode == VIDEO_NORMAL) {
                        return vrcv_display_reconfigure(s, desc);
                }
        }

        if (setup_line_decoder(s, desc)) {
                return true;
        }

        // finally try to find decompress
        codec_t           dec_codec = VC_NONE;
        video_decompress *d         = init_decompress(
            s, desc, (struct pixfmt_desc){ 0 }, &dec_codec, true);
        if (d) {
                start_decompress_thread(s, d, dec_codec);
                return true;
        }

        MSG(ERROR, "Unable to find decoder for input codec \"%s\"!!!\n",
            get_codec_name(desc.color_spec));
        MSG(ERROR,
            "Display doesn't support video codec %s! Supported codecs: %s\n",
            get_codec_name(desc.color_spec),
            codec_list_to_string(s->display_params.native_codecs));
        MSG(ERROR,
            "Could not find neither line conversion nor decompress "
            "from %s to display supported formats (%s).\n",
            get_codec_name(desc.color_spec),
            codec_list_to_string(s->display_params.native_codecs));

        s->saved_network_desc = (struct video_desc){ 0 };
        return false;
}

typedef struct {
        decompress_thread_data   *decoder;
        unsigned                  pos;
        const struct video_frame *compressed;
        decompress_status         ret;
        unsigned char            *out;
        /// @todo callback should not be ideally written in parallel
        struct video_frame_callbacks *callbacks;
        // set only if probing (ret == DECODER_GOT_CODEC)
        struct pixfmt_desc internal_prop;
        size_t             pitch;
} decompress_tile_data;
static void *
decompress_worker(void *data)
{
        decompress_tile_data   *d       = data;
        decompress_thread_data *decoder = d->decoder;

        if (!d->compressed->tiles[d->pos].data) {
                return nullptr;
        }
        d->ret = decompress_frame(
            decoder->decompress->state[d->pos], d->out,
            (unsigned char *) d->compressed->tiles[d->pos].data,
            d->compressed->tiles[d->pos].data_len, (int) d->compressed->seq,
            d->callbacks, &d->internal_prop, d->pitch);
        return d;
}

static bool
decompress(decompress_thread_data *d, const struct video_frame *recv_frame,
           struct video_frame *display_frame, struct pixfmt_desc *comp_desc)
{
        struct state_video_recv *s = d->s;
        unsigned                 tile_width =
            recv_frame->tiles[0]
                .width; // get_video_mode_tiles_x(decoder->video_mode);
        unsigned tile_height =
            recv_frame->tiles[0]
                .height; // get_video_mode_tiles_y(decoder->video_mode);

        if (!display_frame) {
                size_t max_len = ((size_t) recv_frame->tiles[0].width *
                                  recv_frame->tiles[0].height * MAX_BPS) +
                                 MAX_PADDING;
                if (d->scratchpad_allocated < max_len) {
                        free(d->scratchpad);
                        d->scratchpad_allocated = max_len;
                }
                d->scratchpad = malloc(d->scratchpad_allocated);
        }

        task_result_handle_t handle[recv_frame->tile_count];
        decompress_tile_data data[recv_frame->tile_count];
        for (unsigned i = 0; i < recv_frame->tile_count; ++i) {
                unsigned char                *buf    = nullptr;
                struct video_frame_callbacks *clbcks = nullptr;
                size_t                        pitch  = s->display_pitch;
                const struct tile            *tile   = &recv_frame->tiles[i];
                if (!display_frame) {
                        assert(tile->width == recv_frame->tiles[0].width &&
                               tile->height == recv_frame->tiles[0].height);
                        buf   = d->scratchpad;
                        pitch = vc_get_linesize(tile->width,
                                                recv_frame->color_spec);
                } else {
                        if (s->merged_fb) {
                                size_t x =
                                    i % get_video_mode_tiles_x(s->video_mode);
                                size_t y =
                                    i / get_video_mode_tiles_x(s->video_mode);
                                buf = (unsigned char *) vf_get_tile(
                                          display_frame, 0)
                                          ->data +
                                      (y * s->display_pitch * tile_height) +
                                      (vc_get_linesize(
                                           tile_width,
                                           display_frame->color_spec) *
                                       x);
                        } else {
                                buf = (unsigned char *) display_frame->tiles[i]
                                          .data;
                        }
                        clbcks = &display_frame->callbacks;
                }

                data[i].decoder    = d;
                data[i].pos        = i;
                data[i].compressed = recv_frame;
                data[i].ret        = DECODER_NO_FRAME;
                data[i].out        = buf;
                data[i].callbacks  = clbcks;
                // data[i].internal_prop;
                data[i].pitch = pitch;

                handle[i] = task_run_async(decompress_worker, &data[i]);
        }
        for (unsigned i = 0; i < recv_frame->tile_count; ++i) {
                wait_task(handle[i]);
        }

        for (unsigned i = 0; i < recv_frame->tile_count; ++i) {
                switch (data[i].ret) {
                case DECODER_NO_FRAME:
                        return false;
                case DECODER_UNSUPP_PIXFMT:
                        codec_list_erase(s->display_params.native_codecs,
                                         d->configured_display_codec);
                        s->saved_network_desc = (struct video_desc){ 0 };
                        return false;
                case DECODER_GOT_CODEC: {
                        *comp_desc = data[i].internal_prop;
                        MSG(NOTICE, "Detected compression properties: %s\n",
                            get_pixdesc_desc(*comp_desc));
                        return true;
                }
                case DECODER_GOT_FRAME:
                        continue;
                }
                abort(); // ret not part of enum
        }
        return true;
}

static void
submit_frame(struct state_video_recv *s, struct video_frame *f)
{
        CHK_PTHR(pthread_mutex_lock(&s->decode_thread_lock));
        {
                // poison pill must be enqueued
                if (f == &thread_poison_pill) {
                        while (s->decode_thread_frame) {
                                CHK_PTHR(pthread_cond_wait(
                                    &s->decode_thread_frame_consumed,
                                    &s->decode_thread_lock));
                        }
                } else {
                        if (s->decode_thread_frame) {
                                VIDEO_FRAME_DISPOSE(f);
                                if (s->decode_dropped_frames++ % 150 == 20) {
                                        MSG(WARNING, "Your computer may be too "
                                                     "SLOW to play this !!!\n");
                                }
                                goto unlock;
                        }
                }
                s->decode_thread_frame = f;
                CHK_PTHR(
                    pthread_cond_signal(&s->decode_thread_new_frame_ready));
        }
unlock:
        CHK_PTHR(pthread_mutex_unlock(&s->decode_thread_lock));
}

static void
stop_thread(struct state_video_recv *s)
{
        if (s->decode_thread_id == PTHREAD_NULL) {
                return;
        }
        const struct video_frame *cpill = &thread_poison_pill;
        struct video_frame       *pill =
            CONST_CAST(struct video_frame *, pill, cpill);
        submit_frame(s, pill);
        CHK_PTHR(pthread_join(s->decode_thread_id, nullptr));
        s->decode_thread_id = PTHREAD_NULL;
}

static void *
video_receiver_thread(void *arg)
{
        set_thread_name(__func__);
        struct state_video_recv *s = arg;

        struct video_frame *display_frame = nullptr;

        while (true) {
                struct video_frame *ret = rxtx_recv_video_frame(
                    s->rxtx, display_frame, s->display_pitch);
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
                        stop_thread(s);
                        if (!recv_reconfigure(s, video_desc_from_frame(ret))) {
                                VIDEO_FRAME_DISPOSE(ret);
                                continue;
                        }
                }

                const bool have_thread = s->decode_thread_id != PTHREAD_NULL;
                if (have_thread) {
                        submit_frame(s, ret);
                        continue;
                }

                if (!display_frame) {
                        display_frame = display_get_frame(s->display);
                        assert(display_frame);
                }

                if (ret != display_frame) {
                        vf_copy_metadata(display_frame, ret);
                        vf_copy_data_pitch(display_frame, s->display_pitch,
                                           ret);
                        VIDEO_FRAME_DISPOSE(ret);
                }

                display_put_frame(s->display, display_frame, s->putf_timeout);
                display_frame = display_get_frame(s->display);
        }

        if (display_frame) {
                display_put_frame(s->display, display_frame, PUTF_DISCARD);
        }

        stop_thread(s);
        // pass poisoned pill to display
        display_put_frame(s->display, nullptr, PUTF_BLOCKING);

        return nullptr;
}

ADD_TO_PARAM("decoder-drop-policy",
             "* decoder-drop-policy=blocking|nonblock|<sec>\n"
             "  Force specified blocking policy (default nonblock).\n"
             "  <sec> - specifies frame timeout in seconds (can have suffixes, "
             "eg. \"20ms\")\n");
struct state_video_recv *
video_recv_start(struct rxtx *rxtx, const struct rxtx_params *params,
                 struct module *parent, struct display *d)
{
        struct state_video_recv *s = calloc(1, sizeof *s);
        s->magic                   = MAGIC;
        s->vid_recv_thread_id      = PTHREAD_NULL;
        s->rxtx                    = rxtx;
        s->display                 = d;
        s->parent                  = parent;
        s->decode_thread_frame     = nullptr;
        s->decode_thread_id        = PTHREAD_NULL;
        s->display_params          = params->display_params;
        ug_pthread_mutex_init(&s->decode_thread_lock);
        CHK_PTHR(pthread_cond_init(&s->decode_thread_new_frame_ready, nullptr));
        CHK_PTHR(pthread_cond_init(&s->decode_thread_frame_consumed, nullptr));

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

        if (!(params->medium[TX_MEDIA_VIDEO].rxtx_mode & MODE_RECEIVER) ||
            !rxtx_have_receive_video_frame(rxtx)) {
                return s;
        }

        pthread_create(&s->vid_recv_thread_id, nullptr, video_receiver_thread,
                       s);

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
video_recv_join(struct state_video_recv *s)
{
        if (s == nullptr) {
                return;
        }
        if (s->vid_recv_thread_id != PTHREAD_NULL) {
                pthread_join(s->vid_recv_thread_id, nullptr);
        }
}

void
video_recv_done(struct state_video_recv *s)
{
        if (!s) {
                return;
        }
        assert(s->magic == MAGIC);
        CHK_PTHR(pthread_mutex_destroy(&s->decode_thread_lock));
        CHK_PTHR(pthread_cond_destroy(&s->decode_thread_new_frame_ready));
        CHK_PTHR(pthread_cond_destroy(&s->decode_thread_frame_consumed));
        free(s);
}
