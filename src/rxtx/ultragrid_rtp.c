/*
 * FILE:    rxtx/ultragrid_rtp.c
 * AUTHORS: Colin Perkins    <csp@csperkins.org>
 *          Ladan Gharai     <ladan@isi.edu>
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *          David Cassany    <david.cassany@i2cat.net>
 *          Ignacio Contreras <ignacio.contreras@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
 * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2005-2026 CESNET, zájmové sdružení právnických osob
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET, zájmové sdružení právnických osob.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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

#include "rxtx/ultragrid_rtp.h"

#include <assert.h>  // for assert
#include <stdatomic.h>
#include <pthread.h> // for pthread_mutex_lock, pthread_mutex...
#include <stdint.h>  // for uint32_t
#include <stdio.h>   // for fprintf, stderr
#include <stdlib.h>  // for free, calloc
#include <string.h>  // for strcmp
// IWYU pragma: no_include <sys/time.h> # via tv.h

#include "audio/types.h"       // for audio_frame2_delete
#include "compat/c23.h"        // IWYU pragma: keep
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "pdb.h"
#include "rtp/audio_decoders.h" // for decode_audio_frame
#include "rtp/fec.h"            // for fec
#include "rtp/pbuf.h"
#include "rtp/rtp.h"
#include "rtp/video_decoders.h"
#include "rxtx.h"
#include "rxtx/rtp_common.h"  // for rtp_common
#include "tfrc.h"
#include "transmit.h"
#include "tv.h"
#include "types.h"            // for video_frame (ptr only), video_mode
#include "utils/color_out.h"  // for TBOLD, color_printf
#include "utils/macros.h"     // for to_fourcc
#include "utils/misc.h"       // for format_in_si_units
#include "utils/pthread.h"    // for CHK_PTHR, ug_pthread_mutex_ini
#include "utils/text.h"       // for wrap_paragraph
#include "utils/thread.h"
#include "utils/worker.h"
#include "video_display.h"

struct audio_frame2;
struct display;

#define MAGIC    to_fourcc('R', 'T', 'u', 'r')
#define MOD_NAME "[rxtx/ultragrid_rtp] "

struct ultragrid_rtp_rxtx {
        uint32_t magic;

        struct rtp_rxtx_common *rtp_common;

        struct display  *display_device;

        struct display  **display_copies; ///< some displays can be "forked"
                                         ///< and used simultaneously from
                                         ///< multiple decoders, here are
                                         ///< saved forked states
        unsigned display_copies_count;

        /**
         * This variables serve as a notification when asynchronous sending exits
         * @{ */
        bool            async_sending;
        pthread_cond_t  async_sending_cv;
        pthread_mutex_t async_sending_lock;
        /// @}

        long long int         send_bytes_total;
        struct module        *parent;

        time_ns_t start_time;

        struct module *receiver_mod;

        atomic_bool should_exit;
};

// protoypes
static void usage();

static void done(void *state)
{
        struct ultragrid_rtp_rxtx *s = state;
        for (unsigned i = 0; i < s->display_copies_count; ++i) {
                display_done(s->display_copies[i]);
        }
        rtp_rxtx_common_done(s->rtp_common);
        CHK_PTHR(pthread_cond_destroy(&s->async_sending_cv));
        CHK_PTHR(pthread_mutex_destroy(&s->async_sending_lock));
        free(s);
}

static void *
init(struct rxtx_params *params)
{
        if (strlen(params->protocol_opts) > 0) {
                usage();
                return strcmp(params->protocol_opts, "help") == 0 ? INIT_NOERR
                                                                  : nullptr;
        }

        struct ultragrid_rtp_rxtx *s = calloc(1, sizeof *s);

        s->magic          = MAGIC;
        s->display_device = params->display_device;
        s->parent         = params->parent;
        s->start_time     = params->start_time;
        s->receiver_mod   = params->receiver_mod;
        ug_pthread_mutex_init(&s->async_sending_lock);
        pthread_cond_init(&s->async_sending_cv, nullptr);
        int rc = rtp_rxtx_common_init(&s->rtp_common, params);
        if (rc != 0) {
                done(s);
                return rc < 0 ? nullptr : INIT_NOERR;
        }

        if (strlen(params->video_compression) == 0) {
                snprintf_ch(params->video_compression, "none");
        }
        return s;
}


static void join(void *state) {
        struct ultragrid_rtp_rxtx *s = state;
        CHK_PTHR(pthread_mutex_lock(&s->async_sending_lock));
        while (s->async_sending) {
                pthread_cond_wait(&s->async_sending_cv, &s->async_sending_lock);
        }
        CHK_PTHR(pthread_mutex_unlock(&s->async_sending_lock));
}

struct async_data {
        struct ultragrid_rtp_rxtx *s;
        struct video_frame              *f;
};

static void *send_video_frame_async_callback(void *arg);

static void
send_video_frame(void *state, struct video_frame *tx_frame)
{
        struct ultragrid_rtp_rxtx *s = state;
        struct rtp_rxtx_medium *video =
            &s->rtp_common->medium[TX_MEDIA_VIDEO];

        if (video->fec_state != nullptr) {
                struct video_frame *f = fec_encode_video_frame(
                    video->fec_state, tx_frame);
                tx_frame->dispose(tx_frame);
                tx_frame = f;
        }

        struct async_data *data = malloc(sizeof *data);
        data->s = s;
        data->f = tx_frame;

        CHK_PTHR(pthread_mutex_lock(&s->async_sending_lock));
        while (s->async_sending) {
                pthread_cond_wait(&s->async_sending_cv, &s->async_sending_lock);
        }
        rtp_rxtx_sender_do_housekeeping(s->rtp_common, TX_MEDIA_VIDEO);
        s->async_sending = true;
        task_run_async_detached(send_video_frame_async_callback, (void *) data);
        CHK_PTHR(pthread_mutex_unlock(&s->async_sending_lock));
}

static void *send_video_frame_async_callback(void *arg) {
        struct async_data *data = arg;
        struct ultragrid_rtp_rxtx *s    = data->s;
        struct rtp_rxtx_medium *video = &s->rtp_common->medium[TX_MEDIA_VIDEO];
        struct video_frame *tx_frame = data->f;
        free(data);

        CHK_PTHR(pthread_mutex_lock(&video->lock));
        tx_send(video->tx, tx_frame, video->network_device);
        CHK_PTHR(pthread_mutex_unlock(&video->lock));

        tx_frame->dispose(tx_frame);

        CHK_PTHR(pthread_mutex_lock(&s->async_sending_lock));
        s->async_sending = false;
        CHK_PTHR(pthread_mutex_unlock(&s->async_sending_lock));
        CHK_PTHR(pthread_cond_signal(&s->async_sending_cv));

        return nullptr;
}

/**
 * @todo implement decoding to video buffer
 */
static struct video_frame *
recv_vid_frame(void *arg, struct video_frame *display_buffer,
               size_t display_pitch)
{
        struct ultragrid_rtp_rxtx *s = arg;
        return rtp_recv_video_frame(s->rtp_common, decode_video_frame,
                                    display_buffer, display_pitch);
}

static void usage() {
        color_printf("Transport " TBOLD("ultragrid_rtp")
                     " doesn't take any options.\n\n");
        color_printf("Usage:\n\t" TBOLD("-x ultragrid_rtp")
                     "\n");
}

void
ultragrid_rtp_server_mode_help()
{
        color_printf(TBOLD("server mode")
                     " is one of " TBOLD("NAT traversal")
                     " techniques in UG.\n\n");
        char desc[] =
            "It is useful in cases when at least one end is " TBOLD("outside")
            " NAT. "
            "This end will become the \"server\" while the one behind "
            "NAT the client.\n\n";
        color_printf("%s", wrap_paragraph(desc));
        color_printf("Usage:\n");
        color_printf("\t" TBOLD("uv [uv_args] -S")
                     "\n\t\t the server\n");
        color_printf("\t" TBOLD("uv [uv_args] -C <server_address>")
                     "\n\t\t the client\n");
        color_printf("\nSee "
                     "also: <https://github.com/CESNET/UltraGrid/wiki/"
                     "NAT-traversal#server-mode>\nfor more details.\n");
}

static void
send_audio_frame(void *state, const struct audio_frame2 *frame)
{
        struct ultragrid_rtp_rxtx *s = state;
        struct rtp_rxtx_medium *audio = &s->rtp_common->medium[TX_MEDIA_AUDIO];

        rtp_rxtx_sender_do_housekeeping(s->rtp_common, TX_MEDIA_AUDIO);

        struct audio_frame2 *fec_frame = nullptr;
        if (audio->fec_state != nullptr) {
                frame = fec_frame =
                    fec_encode_audio_frame(audio->fec_state, frame);
        }

        audio_tx_send(
            s->rtp_common->medium[TX_MEDIA_AUDIO].tx,
            s->rtp_common->medium[TX_MEDIA_AUDIO].network_device, frame);
        audio_frame2_delete(fec_frame);
}

static bool
ctl_property(void *state, enum rxtx_property p,
                           void *val, size_t *len)
{
        struct ultragrid_rtp_rxtx *s = state;
        assert(s->magic == MAGIC);
        switch (p) {
        case GET_RTP_COMMON_STATE: {
                // NOLINTBEGIN(bugprone-sizeof-expression)
                assert(*len >= sizeof s->rtp_common);
                *len = sizeof s->rtp_common;
                // NOLINTEND(bugprone-sizeof-expression)
                memcpy(val, (void *) &s->rtp_common, *len);
                return true;
        }
        case SET_ULTRAGRID_RTP_MUTLI_OUT_AUDIO:
        case SET_ULTRAGRID_RTP_MUTLI_OUT_VIDEO: {
                bool *var =
                    SET_ULTRAGRID_RTP_MUTLI_OUT_AUDIO
                        ? &s->rtp_common->aplayback_supports_multiple_streams
                        : &s->rtp_common->vplayback_supports_multiple_streams;
                assert(*len >= sizeof *var);
                memcpy(var, val, sizeof *var);
                return true;
        }
        case SET_RTP_AUD_FRM_SZ: {
                int sz = 0;
                assert(*len >= sizeof sz);
                memcpy(&sz, val, sizeof sz);
                rtp_set_recv_buf(
                    s->rtp_common->medium[TX_MEDIA_AUDIO].network_device, sz);
                return true;
        }
        }
        MSG(WARNING, "Unexpected property %d queried!\n", (int) p);
        return false;
}

static struct rx_audio_frames *
recv_audio_frame(void *state)
{
        struct ultragrid_rtp_rxtx *s = state;
        return rtp_recv_audio_frame(s->rtp_common, decode_audio_frame);
}

static const struct rxtx_info ultragrid_rtp_rxtx_info = {
        .long_name    = "UltraGrid RTP",
        .create       = init,
        .done         = done,
        .ctl_property = ctl_property,

        .send_audio_frame = send_audio_frame,
        .recv_audio_frame = recv_audio_frame,

        .send_video_frame_c = send_video_frame,
        .recv_video_frame   = recv_vid_frame,
        .join_video_sender  = join,
};

REGISTER_MODULE(ultragrid_rtp, &ultragrid_rtp_rxtx_info, LIBRARY_CLASS_RXTX,
                RXTX_ABI_VERSION);
