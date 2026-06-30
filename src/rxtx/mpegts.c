// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#include <assert.h> // for assert
#include <stdint.h> // for uint8_t, int64_t, uint32_t
#include <stdio.h>  // for FILE, fclose, fopen, fwrite
#include <stdlib.h> // for abort, free, calloc, malloc
#include <string.h> // for memcpy, strcmp

#include "../../ext-deps/libmpegts/common.h"    // for TIMESTAMP_CLOCK, TS_CLOCK, TS_PACKE...
#include "../../ext-deps/libmpegts/libmpegts.h" // for LIBMPEGTS_MPEG2_AAC_1_CHANNEL, LIBM...

#include "audio/types.h" // for AC_OPUS, AC_AAC, audio_frame2_get_c...
#include "compat/c23.h"  // for countof
#include "debug.h"       // for LOG_LEVEL_DEBUG, MSG
#include "lib_common.h"  // for REGISTER_MODULE, library_class
#include "rtp/net_udp.h" // for socket_udp, udp_init, udp_send
#include "rxtx.h"        // for rxtx_params, RXTX_ABI_VERSION, rxtx...
#include "types.h"       // for video_frame, tile, kHz48, video_fra...
struct audio_frame2;

#define MOD_NAME "[rxtx/mpegts] "

#include "types.h"

struct rxtx_mpegts {
        uint32_t     magic;
        ts_writer_t *writer;
        bool         init;
        long long    frames;
        socket_udp  *sock;
        int          mtu;

        double audio_duration;

        FILE *dump_f;
};

enum {
        PCR_PID   = 0x100,
        VIDEO_PID = 0x100,
        AUDIO_PID = 0x101,
        PMT_PID   = 0x1000
};

static void *
init(const struct rxtx_params *params)
{
        struct rxtx_mpegts *s = calloc(1, sizeof *s);
        s->mtu                = params->mtu / TS_PACKET_SIZE * TS_PACKET_SIZE;
        s->writer             = ts_create_writer();
        s->sock               = udp_init(params->receiver, 0, 1234, params->ttl,
                                         params->force_ip_version, false);

        if (strcmp(params->protocol_opts, "dump") == 0) {
                s->dump_f = fopen("out.ts", "wb");
        }

        return s;
}

static bool
init_video(struct rxtx_mpegts *s)
{
        ts_stream_t stream   = { 0 };
        stream.pid           = VIDEO_PID;
        stream.stream_format = LIBMPEGTS_VIDEO_AVC;           // = 2 [1]
        stream.stream_id     = LIBMPEGTS_STREAM_ID_MPEGVIDEO; // = 0xe0 [1]

        ts_program_t prog = { 0 }; // = &main_params.programs[0];
        prog.program_num  = 1;
        prog.pmt_pid      = PMT_PID;
        prog.pcr_pid      = PCR_PID;
        prog.num_streams  = 1;
        prog.streams      = &stream;

        ts_main_t main_params  = { 0 };
        main_params.lowlatency = 1;
        main_params.ts_id      = 1;
        main_params.muxrate    = 5000000; // 5 Mbps
        // Constant bitrate - if set to 1, it will fill to match bitrate
        main_params.cbr          = 0;
        main_params.ts_type      = TS_TYPE_GENERIC;
        main_params.num_programs = 1;
        main_params.programs     = &prog;

        int rc = ts_setup_transport_stream(s->writer, &main_params);
        if (rc != 0) {
                return false;
        }

        // Setup AVC stream parameters
        rc = ts_setup_mpegvideo_stream(
            s->writer,
            VIDEO_PID, // PID
            52,        // level
            AVC_HIGH,  // profile (from avc_profile_t enum) [1]
            5000000,   // vbv_maxrate (bits/s)
            1000000,   // vbv_bufsize
            0          // frame_rate (not used for AVC) [1]
        );
        return rc == 0;
}

static void
udp_send_packets(struct rxtx_mpegts *s, uint8_t *output, int output_len)
{
        if (s->dump_f != nullptr) {
                fwrite(output, output_len, 1, s->dump_f);
        }

        int len = s->mtu;
        while (output_len > 0) {
                if (output_len < len) {
                        len = output_len;
                }
                udp_send(s->sock, (char *) output, len);
                output_len -= len;
                output += len;
        }
}

static void
send_video_frame(void *state, struct video_frame *f)
{
        struct rxtx_mpegts *s = state;

        if (!s->init) {
                if (!init_video(s)) {
                        abort();
                }
                s->init = true;
        }

        ts_frame_t ts_frame = { 0 };
        ts_frame.pid        = VIDEO_PID;
        ts_frame.data       = (uint8_t *) f->tiles[0].data;
        ts_frame.size       = (int) f->tiles[0].data_len;
        // ts_frame.random_access = 1; // is keyframe
        // ts_frame.frame_type    = LIBMPEGTS_CODING_TYPE_SLICE_IDR;
        // int nal_ref_idc        = 3; // @todo
        // ts_frame.ref_pic_idc   = nal_ref_idc;

        // 90kHz clock ticks [1]
        ts_frame.dts = ts_frame.pts =
            (s->frames + 1) * (TIMESTAMP_CLOCK / f->fps);

        ts_frame.cpb_initial_arrival_time = s->frames * (TS_CLOCK / f->fps);
        ts_frame.cpb_final_arrival_time = (s->frames + 1) * (TS_CLOCK / f->fps);

        uint8_t *output     = nullptr;
        int      output_len = 0;
        int64_t *pcr_list   = nullptr;

        ts_write_frames(s->writer, &ts_frame, 1, &output, &output_len,
                        &pcr_list);

        MSG(DEBUG, "ts_write_frames: %d B\n", output_len);
        udp_send_packets(s, output, output_len);
        f->callbacks.dispose(f);

        s->frames += 1;
}

static bool
init_audio(struct rxtx_mpegts *s, audio_codec_t ac, double duration)
{
        assert(ac == AC_OPUS || ac == AC_AAC);
        ts_stream_t stream   = { 0 };
        stream.pid           = AUDIO_PID;
        stream.stream_format = ac == AC_OPUS ? LIBMPEGTS_AUDIO_OPUS
                                             : LIBMPEGTS_AUDIO_ADTS; // = 2 [1]
        stream.stream_id     = LIBMPEGTS_STREAM_ID_MPEGAUDIO; // = 0xe0 [1]
        stream.audio_frame_size = TIMESTAMP_CLOCK * duration;

        ts_program_t prog = { 0 }; // = &main_params.programs[0];
        prog.program_num  = 1;
        prog.pmt_pid      = PMT_PID;
        prog.pcr_pid      = PCR_PID;
        prog.num_streams  = 1;
        prog.streams      = &stream;

        ts_main_t main_params  = { 0 };
        main_params.lowlatency = 1;
        main_params.ts_id      = 1;
        main_params.muxrate    = 5000000; // must be a larger number - see
                                       // libmpegts/libmpegts.c:1795, if set to
                                       // eg 200 kbps check_pcr returns always 1
                                       // and no actual audio data get sent
        // Constant bitrate - if set to 1, it will fill to match bitrate
        main_params.cbr          = 0;
        main_params.ts_type      = TS_TYPE_GENERIC;
        main_params.num_programs = 1;
        main_params.programs     = &prog;

        int rc = ts_setup_transport_stream(s->writer, &main_params);
        if (rc != 0) {
                return false;
        }

        // Setup Opus stream parameters
        rc = ac == AC_OPUS
                 ? ts_setup_opus_stream(s->writer, AUDIO_PID,
                                        LIBMPEGTS_CHANNEL_CONFIG_MONO)
                 : ts_setup_mpeg2_aac_stream(
                       s->writer, AUDIO_PID,           // ADTS -> MPEG2
                       LIBMPEGTS_MPEG2_AAC_LC_PROFILE, // default by FFmpeg
                       LIBMPEGTS_MPEG2_AAC_1_CHANNEL);
        return rc == 0;
}

enum { ADTS_HDR_SZ = 7 };
static void
write_adts_header(uint8_t *header, size_t frame_size, unsigned sample_rate,
                  int channel_cfg, int profile)
{
        unsigned       sample_rate_idx = 3; // default - 48 kHz
        const unsigned rates[] = { 96000, 88200, 64000, 48000, 44100, 32000,
                                   24000, 22050, 16000, 12000, 11025, 8000 };
        for (unsigned i = 0; i < countof(rates); i++) {
                if (rates[i] == sample_rate) {
                        sample_rate_idx = i;
                        break;
                }
        }

        // Byte 0-1: Syncword (0xFFF)
        header[0] = 0xFF;
        header[1] = 0xF9; // F1 = no CRC, MPEG-4
        // Alternative: 0xF9 for MPEG-2, 0xF0 if CRC present

        // Byte 2: Profile + sample_rate_idx (upper bits)
        header[2] = ((profile & 0x03) << 6) | ((sample_rate_idx & 0x0F) << 2) |
                    ((channel_cfg >> 2) & 0x01);

        // Byte 3: Channel config (lower bits) + frame length (upper bits)
        header[3] = ((channel_cfg & 0x03) << 6) | ((frame_size >> 11) & 0x03);

        // Byte 4: Frame length (middle bits)
        header[4] = (frame_size >> 3) & 0xFF;

        // Byte 5: Frame length (lower bits) + buffer fullness (upper bits)
        header[5] =
            ((frame_size & 0x07) << 5) | 0x1F; // 0x1F = variable bitrate

        // Byte 6: Number of AAC frames (0 = 1 frame - 1)
        header[6] = 0xFC; // 0xFC = 1 frame, no CRC
}

static uint8_t *
add_adts_header(const uint8_t *data, int len)
{
        size_t new_sz  = len + ADTS_HDR_SZ;
        uint8_t *ret = malloc(new_sz);
        write_adts_header((unsigned char *) ret, new_sz, kHz48,
                          LIBMPEGTS_MPEG2_AAC_1_CHANNEL,
                          LIBMPEGTS_MPEG2_AAC_LC_PROFILE);
        memcpy(ret + ADTS_HDR_SZ, data, len);
        return ret;
}

static void
send_audio_frame(void *state, const struct audio_frame2 *f)
{
        struct rxtx_mpegts *s = state;

        double duration = audio_frame2_get_duration(f);
        if (!s->init) {
                if (!init_audio(s, audio_frame2_get_codec(f), duration)) {
                        abort();
                }
                s->init = true;
        }

        const uint8_t *ad = (const uint8_t *) audio_frame2_get_data(f, 0);

        ts_frame_t ts_frame = { 0 };
        ts_frame.pid        = AUDIO_PID;
        // equals to `ts_frame.data = (const void *) ad;`:
        memcpy((void *) &ts_frame.data, (const void *) &ad, sizeof ad);
        ts_frame.size = (int) audio_frame2_get_data_len(f, 0);
        // ts_frame.random_access = 1; // is keyframe
        // ts_frame.frame_type    = LIBMPEGTS_CODING_TYPE_SLICE_IDR;
        // int nal_ref_idc        = 3; // @todo
        // ts_frame.ref_pic_idc   = nal_ref_idc;

        if (audio_frame2_get_codec(f) == AC_AAC) { // we need to add ADTS hdr
                ts_frame.data = add_adts_header(ad, ts_frame.size);
                ts_frame.size += ADTS_HDR_SZ;
        }

        // fwrite(ts_frame.data, ts_frame.size, 1, out); fclose(out); abort();

        // 90kHz clock ticks [1]
        ts_frame.dts = ts_frame.pts =
            (int64_t) ((s->audio_duration + duration) * TIMESTAMP_CLOCK);

        // ts_frame.cpb_initial_arrival_time = s->audio_duration * TS_CLOCK;
        // ts_frame.cpb_final_arrival_time =
        //     ts_frame.cpb_initial_arrival_time + (TS_CLOCK * duration);

        uint8_t *output     = nullptr;
        int      output_len = 0;
        int64_t *pcr_list   = nullptr;

        ts_write_frames(s->writer, &ts_frame, 1, &output, &output_len,
                        &pcr_list);
        MSG(DEBUG, "ts_write_frames audio: %d B (in %d B)\n", output_len,
            ts_frame.size);

        udp_send_packets(s, output, output_len);

        // ts_write_frames(s->writer, &ts_frame, 0, &output, &output_len,
        // &pcr_list); MSG(DEBUG, " 2 ts_write_frames audio: %d B (in %d B)\n",
        // output_len, ts_frame.size);

        if ((const uint8_t *) ts_frame.data != ad) {
                free(ts_frame.data);
        }

        s->audio_duration += duration;
}

static void
done(void *state)
{
        struct rxtx_mpegts *s = state;
        if (s->dump_f) {
                fclose(s->dump_f);
        }
        free(s);
}

static const struct rxtx_info mpegts_rxtx_info = {
        .long_name    = "MPEG transport stream",
        .create       = init,
        .done         = done,
        .ctl_property = nullptr,

        .send_audio_frame = send_audio_frame,
        .recv_audio_frame = nullptr,

        .send_video_frame   = nullptr,
        .send_video_frame_c = send_video_frame,
        .video_recv_routine = nullptr,
        .join_video_sender  = nullptr,
};

REGISTER_MODULE(mpegts, &mpegts_rxtx_info, LIBRARY_CLASS_RXTX,
                RXTX_ABI_VERSION);
