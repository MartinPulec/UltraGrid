/*
 * AUTHOR:   Ladan Gharai/Colin Perkins
 * 
 * Copyright (c) 2003-2004 University of Southern California
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
 * $Revision: 1.1.2.4 $
 * $Date: 2010/01/30 20:11:45 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/decoders.h"
#include "video_codec.h"

#define DXT_WIDTH 1920/4
#define DXT_DEPTH 8

static void 
copy_p2f (struct video_frame *frame, rtp_packet *pckt)
{
    char            *offset;
    payload_hdr_t   *hdr;
    uint32_t        len;
    uint32_t        data_pos;

    hdr = (payload_hdr_t *)pckt->data;
    offset = (char*)(pckt->data + sizeof(payload_hdr_t));

    len = ntohs(hdr->length);
    data_pos = ntohl(hdr->offset);
   
    if(frame->data_len > data_pos + len) {
        memcpy(frame->data + data_pos, offset, len);
    }

    frame->width = ntohs(hdr->width);
    frame->height = ntohs(hdr->height);
    frame->color_spec = hdr->colorspc;

//    fprintf(stdout, "Received: len %d, pos %d, width %d, height %d, color spc %d\n",
  //      ntohs(hdr->length), ntohl(hdr->offset), hd_size_x, hd_size_y, hd_color_spc);

}

void
decode_frame(struct coded_data *cdata, struct video_frame *frame)
{
	while (cdata != NULL) {
		copy_p2f(frame, cdata->data);
		cdata = cdata->nxt;
	}
}

