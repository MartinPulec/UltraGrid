/*
 * AUTHOR:   David Cassany   <david.cassany@i2cat.net>,
 *           Ignacio Contreras <ignacio.contreras@i2cat.net>,
 *           Gerard Castillo <gerard.castillo@i2cat.net>
 *
 * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2015-2024 CESNET
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
 *      California Information Sciences Institute.
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

#ifndef _RTP_DEC_H264_H
#define _RTP_DEC_H264_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
        NAL_H264_MIN = 1,
        NAL_H264_MAX = 23,
};

enum nal_type {
        // H.264
        NAL_H264_NON_IDR = 1,
        NAL_H264_IDR = 5,
        NAL_H264_SEI = 6,
        NAL_H264_SPS = 7,
        NAL_H264_PPS = 8,
        NAL_H264_AUD = 9,
        // HEVC
        NAL_HEVC_VPS = 32,
        NAL_HEVC_SPS = 33,
        NAL_HEVC_PPS = 34,
        NAL_HEVC_AUD = 35,
};

/// NAL values >23 are invalid in H.264 codestream but used by RTP
enum aux_nal_types {
        RTP_STAP_A = 24,
        RTP_STAP_B = 25,
        RTP_MTAP16 = 26,
        RTP_MTAP24 = 27,
        RTP_FU_A   = 28,
        RTP_FU_B   = 29,
};

struct coded_data;
struct decode_data_rtsp;
struct video_desc;

#define H264_NALU_HDR_GET_TYPE(nal) ((nal) & 0x1F)
#define H264_NALU_HDR_GET_NRI(nal) (((nal) & 0x60) >> 5)
#define NALU_HDR_GET_TYPE(nal, is_hevc) \
        ((is_hevc) ? (nal) >> 1 : H264_NALU_HDR_GET_TYPE((nal)))

int decode_frame_h264(struct coded_data *cdata, void *decode_data);
struct video_frame *get_sps_pps_frame(const struct video_desc *desc,
                                      struct decode_data_rtsp *decode_data);
int width_height_from_SDP(int *widthOut, int *heightOut , unsigned char *data, int data_len);

#ifdef __cplusplus
}
#endif

#endif
