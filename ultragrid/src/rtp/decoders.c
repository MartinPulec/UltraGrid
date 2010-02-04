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
 * $Revision: 1.1.2.6 $
 * $Date: 2010/02/04 09:30:36 $
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

void
decode_frame(struct coded_data *cdata, struct video_frame *frame)
{
        uint32_t width;
        uint32_t height;
        uint32_t offset;
        uint32_t len;
        codec_t color_spec;
        rtp_packet *pckt;
        unsigned char *source;
        payload_hdr_t   *hdr;
        uint32_t data_pos;

	while (cdata != NULL) {

                pckt = cdata->data;
                hdr = (payload_hdr_t *)pckt->data;
                width = ntohs(hdr->width);
                height = ntohs(hdr->height);
                color_spec = hdr->colorspc;
                len = ntohs(hdr->length);
                data_pos = ntohl(hdr->offset);

                /* Critical section 
                 * each thread *MUST* wait here if this condition is true
                 */
                if(!(frame->width == width &&
                     frame->height == height &&
                     frame->color_spec == color_spec)) {
                        frame->reconfigure(frame->state, width, height, color_spec);
                        frame->src_linesize = vc_getsrc_linesize(width, color_spec);
                        if(frame->src_linesize < frame->dst_linesize - frame->dst_x_offset) {
                                frame->visiblesize = frame->src_linesize;
                        } else {
                                frame->visiblesize = frame->dst_linesize - frame->dst_x_offset;
                        }
                }
                /* End of critical section */
        
                /* MAGIC, don't touch it, you definitely break it */
                int y = (data_pos / frame->src_linesize)*frame->dst_linesize;
                int x = data_pos % frame->src_linesize;
                source = pckt->data + sizeof(payload_hdr_t);
                while(len > 0){
                        int l = len;
                        if(l + x > frame->visiblesize) {
                                l = frame->visiblesize - x;
                        }
                        offset = y + x;
                        if(l + offset < frame->data_len) {
                                frame->decoder(frame->data+offset, source, l, 
                                              frame->rshift, frame->gshift, frame->bshift);
                                len -= frame->src_linesize - x;
                                source += frame->src_linesize - x;
                        } else {
                                len = 0;
                        }
                        x = 0; /* next line from beginning */
                        y += frame->dst_linesize; /* next line */
                }

		cdata = cdata->nxt;
	}
}

