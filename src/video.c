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
#include "debug.h"          // for LOG_LEVEL_ERROR, MSG, debug_msg
#include "host.h"           // for register_should_exit_callback, unregister...
#include "rxtx.h"           // for rxtx_medium_params, rxtx_have_receive_vid...
#include "types.h"          // for rxtx_mode, tx_media_type, video_desc
#include "utils/macros.h"   // for to_fourcc
#include "utils/pthread.h"  // for PTHREAD_NULL
#include "utils/thread.h"   // for set_thread_name
#include "video_codec.h"    // for get_codec_name
#include "video_display.h"  // for PITCH_DEFAULT, PUTF_NONBLOCK, display_get...
#include "video_frame.h"    // for vf_copy_data_pitch, vf_copy_metadata, vf_...

#define MOD_NAME "[video] "
#define MAGIC to_fourcc('v', 'd', 'e', 'o')

struct state_video {
        uint32_t        magic;
        pthread_t       receiver_thread;
        struct rxtx    *rxtx;
        struct module  *parent;

        struct display *display;
        size_t          display_pitch;
};

static bool
check_display_supports_codec(struct display *display_device, codec_t codec)
{
        codec_t display_codecs[VIDEO_CODEC_COUNT];
        size_t  len = sizeof display_codecs;
        if (!display_ctl_property(display_device, DISPLAY_PROPERTY_CODECS,
                                  display_codecs, &len)) {
                MSG(ERROR, "Failed to query codecs from video display.\n");
                return false;
        }
        for (unsigned i = 0; i < len / sizeof(codec_t); ++i) {
                if (display_codecs[i] == codec) {
                        return true;
                }
        }

        char  buf[STR_LEN] = "";
        char *codec_list   = buf;
        for (unsigned i = 0; i < len / sizeof(codec_t); ++i) {
                if (display_codecs[i] == codec) {
                        return true;
                }
                codec_list += snprintf(
                    codec_list, buf + sizeof buf - codec_list, "%s%s",
                    i != 0 ? ", " : "", get_codec_name(display_codecs[i]));
        }
        MSG(ERROR,
            "Display doesn't support video codec %s! Supported codecs: %s\n",
            get_codec_name(codec), buf);
        return false;
}


static struct video_frame *
recv_reconfigure(struct state_video *s, struct video_desc desc)
{
        if (!check_display_supports_codec(s->display, desc.color_spec)) {
                return nullptr;
        }

        if (!display_reconfigure(s->display, desc)) {
                MSG(ERROR, "Cannot reconfigure display!\n");
                return nullptr;
        }
        int display_requested_pitch = PITCH_DEFAULT;
        size_t len = sizeof display_requested_pitch;
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
        struct video_frame *f = display_get_frame(s->display);
        assert(f);
        return f;
}

static void *
video_receiver_thread(void *arg)
{
        set_thread_name(__func__);
        struct state_video *s = arg;

        struct video_frame *f = nullptr;

        while (true) {
                struct video_frame *ret =
                    rxtx_recv_video_frame(s->rxtx, f, s->display_pitch);
                if (!ret) {
                        break;
                }
                if (ret == rxtx_retry) {
                        continue;
                }
                if (ret != f) {
                        if (!f || !video_desc_eq(video_desc_from_frame(f),
                                                video_desc_from_frame(ret))) {
                                if (f) {
                                        display_put_frame(s->display, f,
                                                          PUTF_DISCARD);
                                }
                                f = recv_reconfigure(
                                    s, video_desc_from_frame(ret));
                                if (f == nullptr) {
                                        VIDEO_FRAME_DISPOSE(ret);
                                        continue;
                                }
                                assert(
                                    video_desc_eq(video_desc_from_frame(f),
                                                  video_desc_from_frame(ret)));
                        }

                        vf_copy_metadata(f, ret);
                        vf_copy_data_pitch(f, s->display_pitch, ret);
                        VIDEO_FRAME_DISPOSE(ret);
                }
                display_put_frame(s->display, f, PUTF_NONBLOCK);
                f = display_get_frame(s->display);
        }

        if (f) {
                display_put_frame(s->display, f, PUTF_DISCARD);
        }

        // pass poisoned pill to display
        display_put_frame(s->display, nullptr, PUTF_BLOCKING);

        return nullptr;
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
        free(s);
}
