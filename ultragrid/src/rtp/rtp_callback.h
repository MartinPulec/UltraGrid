/*
 * FILE:   rtp_callback.h
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *
 * Copyright (c) 2001-2003 University of Southern California
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
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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

#include "host.h"

/*
 * Packet formats:
 * http://www.cesnet.cz/doc/techzpravy/2010/4k-packet-format/
 */

#if 0
typedef struct {
    uint16_t    width;      /* pixels */
    uint16_t    height;     /* pixels */
    uint32_t    offset;     /* in bytes */
    uint16_t    length;     /* octets */
    uint8_t     colorspc;
    uint8_t     flags;
    uint32_t    fps;        /* fixed point fps. take care! */
    uint32_t    aux;        /* auxiliary data */
    uint32_t    tileinfo;   /* info about tile position (if tiled) */
} payload_hdr_t;
#endif

/* VIDEO PART */
#define PCKT_LENGTH             0
#define PCKT_HRES_VRES          1       /* bits 0 - 15 - horizontal resolution
                                           bits 15 - 31 - vertical resolution */
#define PCKT_FOURCC             2
#define PCKT_IL_FPS             3       /* bits 0 - 2 interlace flag
                                           bits 3 - 12 FPS
                                           bits 13 - 16 FPSd
                                           bit 17 Fd
                                           bit 18 Fi */
#define PCKT_SEQ_NEXT_HDR       4       /* bits 0 - 30 packet seq
                                           bit 31 next header */

#define PCKT_HDR_BASE_LEN       5

/* EXTENSIONS */
#define PCKT_EXT_INFO           0       /* bits 0 - 4 type
                                           bits 5 - 20 ext hdr length
                                           bit 31 next header */
#define PCKT_EXT_INFO_LEN       1

/* AUDIO */
#define PCKT_EXT_AUDIO_TYPE     0x1

#define PCKT_EXT_AUDIO_LENGTH   0
#define PCKT_EXT_AUDIO_QUANT_SAMPLE_RATE   1 /* bits 0 - 5 audio quant.
                                                 bits 6 - 31 audio sample rate */
#define PCKT_EXT_AUDIO_CHANNEL_COUNT  2
#define PCKT_EXT_AUDIO_TAG      3

#define PCKT_HDR_AUDIO_LEN      4
 

#define PCKT_HDR_MAX_LEN        (PCKT_HDR_BASE_LEN + PCKT_EXT_INFO_LEN + PCKT_HDR_AUDIO_LEN)


typedef struct {
        /* first word */
        uint32_t substream_bufnum; /* bits 0 - 9 substream
                                      bits 10 - 31 buffer  */

        /* second word */
        uint32_t offset;

        /* third word */
        uint32_t length;

        /* fourth word */
        uint16_t hres;
        uint16_t vres;

        /* fifth word */
        uint32_t fourcc;
        
        /* temporary */
        uint32_t il_fps; /* bits 0 - 2 interlace flag
                            bits 3 - 12 FPS
                            bits 13 - 16 FPSd
                            bit 17 Fd
                            bit 18 Fi */
        uint32_t frame;

} __attribute__((__packed__)) video_payload_hdr_t;

typedef struct {
        /* first word */
        uint32_t substream_bufnum; /* bits 0 - 9 substream
                                      bits 10 - 31 buffer */

        /* second word */
        uint32_t offset;

        /* third word */
        uint32_t length;

        /* fourth word */
        uint32_t quant_sample_rate; /* bits 0 - 5 audio quant.
                                       bits 6 - 31 audio sample rate */

        /* fifth word */
        uint32_t audio_tag;
} __attribute__((__packed__)) audio_payload_hdr_t;


void rtp_recv_callback(struct rtp *session, rtp_event *e);
int handle_with_buffer(struct rtp *session,rtp_event *e);
int check_for_frame_completion(struct rtp *);
void process_packet_for_display(char *);
void call_display_frame(void);
