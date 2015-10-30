/**
 * @file   video_rxtx/mvtp.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2014 CESNET z.s.p.o.
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


#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <memory>
#include <string>
#include <sstream>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "video_display.h"
#include "video_rxtx/mvtp.h"
#include "video_rxtx.h"
#include "video.h"

#include <libMVTP/MVTPConvert.h>

using namespace std;
using namespace MVTP;

mvtp_video_rxtx::UGMVTPStreamer::UGMVTPStreamer(MVTPSocket &socket, MVTPTimeSource &time_source)
        : MVTPStreamer(socket, time_source), should_exit(false)
{
        thread_id = thread(worker, this);
}

mvtp_video_rxtx::UGMVTPReceiver::UGMVTPReceiver(MVTPSocket &socket, struct display *d)
        : MVTPBasicReceiver(socket), display_device(d)
{
        codec_t native_codecs[20];
        size_t len = sizeof(native_codecs);
        int ret = display_get_property(d, DISPLAY_PROPERTY_CODECS, native_codecs, &len);
        if (ret) {
                bool found = false;
                for (size_t i = 0; i < len / sizeof(codec_t); ++i) {
                        if (native_codecs[i] == UYVY) {
                                found = true;
                        }
                }
                if (!found) {
                        throw string("UYVY is not supported within display!\n");
                }
        }
}

mvtp_video_rxtx::UGMVTPStreamer::~UGMVTPStreamer()
{
        should_exit = true;
        thread_id.join();
}

bool mvtp_video_rxtx::UGMVTPReceiver::channel_format_changed(uint8_t channel)
{
        const MVTP::MVTPFrameParams& params = parameters[channel];
        cerr << "Format changed for channel " << static_cast<int>(channel) << endl;
        cerr << params.width << "x" << params.height <<" @" << params.get_fps() << "Hz"<< endl;

        if (channel == 0) {
                struct video_desc display_desc{params.width, params.height, UYVY,
                        static_cast<double>(params.get_fps()),
                        PROGRESSIVE, 1};
                if (!display_reconfigure(display_device, display_desc)) {
                        log_msg(LOG_LEVEL_FATAL, "Unable to reconfigure display!\n");
                        abort();
                }
        }

        return MVTPBasicReceiver::channel_format_changed(channel);

        return true;
}


bool mvtp_video_rxtx::UGMVTPReceiver::frame_complete(uint8_t channel)
{
        if (channel != 0) {
                return false;
        }

        struct video_frame *f = display_get_frame(display_device);

        uint8_t * const src = &frame_data[channel][0];
        const MVTP::MVTPDataFormat *src_data = reinterpret_cast<const MVTP::MVTPDataFormat*>(src);

        for (size_t row = 0; row < parameters[0].height; ++row) {
                uint8_t * dest = (uint8_t *) f->tiles[0].data + row * vc_get_linesize(parameters[0].width, UYVY);
                for (size_t col = 0; col < static_cast<size_t>(parameters[0].width>>1);++col) {
                        src_data++->get_as_uyvy(dest);
                        dest+=4;
                }
        }

        display_put_frame(display_device, f, PUTF_BLOCKING);

        return true;
}

mvtp_video_rxtx::mvtp_video_rxtx(map<string, param_u> const &params) :
        video_rxtx(params),
        socket(params.at("rx_port").i),
        streamer(socket, timer),
        receiver(socket, (struct display *) params.at("display_device").ptr)
{
        auto address = static_cast<const char *>(params.at("receiver").ptr);
        int tx_port = params.at("tx_port").i;

        socket.set_target(address, tx_port);
}

void mvtp_video_rxtx::UGMVTPStreamer::worker(class mvtp_video_rxtx::UGMVTPStreamer *s)
{
        while (!s->should_exit) {
                unique_lock<mutex> lk(s->lock);
                auto video_frame = s->frame_to_send;
                lk.unlock();
                if (video_frame) {
                        s->stream_frames();
                }
        }
}

static void uyvy_to_mvtp(unsigned char *dst, unsigned char *src, size_t data_len)
{
        MVTP::MVTPDataFormat *dest = reinterpret_cast<MVTP::MVTPDataFormat*>(dst);

        for (size_t pix=0;pix<data_len;pix+=4) {
                dest++->set_from_components(conv_Y_8_10(src[1]), conv_Y_8_10(src[3]),
                                conv_C_8_10(src[0]), conv_C_8_10(src[2]));
                //dest++->set_from_components(src[1]<<2,src[3]<<2,src[0]<<2,src[2]<<2);
                //dest++->set_from_components(src[0],src[2],src[1],src[3]);
                src+=4;
        }
}

void mvtp_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame)
{
        streamer.send_frame(tx_frame);
}

void mvtp_video_rxtx::UGMVTPStreamer::send_frame(shared_ptr<video_frame> tx_frame)
{
        if (tx_frame->color_spec != UYVY) {
                log_msg(LOG_LEVEL_ERROR, "Unsupported codec format - currently only UYVY is supported!\n");
                return;
        }
        struct video_desc desc = video_desc_from_frame(tx_frame.get());
        desc.color_spec = VIDEO_CODEC_NONE;
        shared_ptr<video_frame> copy(vf_alloc_desc_data(desc), vf_free);
        copy->data_deleter = vf_data_deleter;
        copy->tiles[0].data_len = 5 * desc.width * desc.height;
        copy->tiles[0].data = (char *) malloc(copy->tiles[0].data_len);
        uyvy_to_mvtp((unsigned char *) copy->tiles[0].data, (unsigned char *) tx_frame->tiles[0].data, tx_frame->tiles[0].data_len);
        unique_lock<mutex> lk(lock);
        frame_to_send = copy;
}

mvtp_video_rxtx::~mvtp_video_rxtx()
{
}

static bool verify_format(struct video_desc desc) {
        return ((desc.width == 1920 && desc.height == 1080) ||
                        (desc.width == 2048 && desc.height == 1080) ||
                        (desc.width == 1280 && desc.height == 720)) &&
        (desc.interlacing == PROGRESSIVE || desc.interlacing == INTERLACED_MERGED || desc.interlacing == SEGMENTED_FRAME);
}

static string get_format(struct video_desc desc) {
        if (!verify_format(desc)) {
                return {};
        } else {
                ostringstream oss;
                if (desc.height == 2048) {
                        oss << "2k";
                } else {
                        oss << desc.height;
                }

                oss << get_interlacing_suffix(desc.interlacing) << setiosflags(ios_base::fixed) << setprecision(0)
                        << (desc.interlacing == PROGRESSIVE || desc.interlacing == SEGMENTED_FRAME ? 1 : 2)
                        * (desc.fps - floor(desc.fps) == 0 ? desc.fps : desc.fps * 100.0);

                return oss.str();
        }
}

const uint8_t *mvtp_video_rxtx::UGMVTPStreamer::next_frame(uint8_t channel)
{
        unique_lock<mutex> lk(lock);
        auto frame = frame_to_send;
        lk.unlock();
        if (channel != 0 || !frame) {
                return 0;
        }

        MVTP::MVTPFrameParams &params = parameters[0];
        const string & format = get_format(video_desc_from_frame(frame_to_send.get()));
        if (format.empty() || !MVTP::MVTPFormats.count(format)) {
                log_msg(LOG_LEVEL_ERROR, "Unsupported MVTP format!\n");
                return 0;
        }
        params = MVTP::MVTPFormats[format];
        params.set_full_frame(false);
        params.set_channel(0);
        params.compute();

        old_frame = frame;
        return reinterpret_cast<uint8_t*>(frame->tiles[0].data);
}

void *mvtp_video_rxtx::UGMVTPReceiver::receiver_loop()
{
        while (!should_exit_receiver) {
                step();
        }

        // pass posioned pill to display
        display_put_frame(display_device, NULL, PUTF_BLOCKING);

        return nullptr;
}

static video_rxtx *create_video_rxtx_mvtp(std::map<std::string, param_u> const &params)
{
        return new mvtp_video_rxtx(params);
}

static const struct video_rxtx_info mvtp_video_rxtx_info = {
        "MVTP",
        create_video_rxtx_mvtp
};

REGISTER_MODULE(mvtp, &mvtp_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

