/*
 * FILE:     pbuf.h
 * AUTHOR:   N.Cihan Tas
 * MODIFIED: Ladan Gharai
 *           Colin Perkins
 *           Martin Benes     <martinbenesh@gmail.com>
 *           Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *           Petr Holub       <hopet@ics.muni.cz>
 *           Milos Liska      <xliska@fi.muni.cz>
 *           Jiri Matela      <matela@ics.muni.cz>
 *           Dalibor Matura   <255899@mail.muni.cz>
 *           Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 * 
 * Copyright (c) 2003-2004 University of Southern California
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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
 * $Revision: 1.5 $
 * $Date: 2010/01/28 10:06:59 $
 *
 */

/******************************************************************************/
/* The main playout buffer data structures. See "RTP: Audio and Video for the */
/* Internet" Figure 6.8 (page 167) for a diagram.                       [csp] */
/******************************************************************************/

#include "config.h"
#include "video_display.h"

#include "audio/audio.h"
#include "rtp/rtp.h"

/* The coded representation of a single frame */
struct coded_data {
        struct coded_data       *nxt;
        struct coded_data       *prv;
        uint16_t                 seqno;
        rtp_packet              *data;
};

/* The playout buffer */
struct pbuf;
struct state_decoder;
struct state_audio_decoder;

struct pbuf_video_data {
        struct video_frame *frame_buffer;
        struct state_decoder *decoder;
};

struct pbuf_audio_data {
        audio_frame *buffer;
        struct state_audio *audio_state;
        int saved_channels;
        int saved_bps;
        int saved_sample_rate;
};

/**
 * @param decode_data
 */
typedef int decode_frame_t(struct coded_data *cdata, void *decode_data);
/* 
 * External interface: 
 */
struct pbuf	*pbuf_init(void);
void		 pbuf_insert(struct pbuf *playout_buf, rtp_packet *r);
int 	 	 pbuf_decode(struct pbuf *playout_buf, struct timeval curr_time,
                             decode_frame_t decode_func, void *data, int wait_for_playout_time);
                             //struct video_frame *framebuffer, int i, struct state_decoder *decoder);
void		 pbuf_remove(struct pbuf *playout_buf, struct timeval curr_time);


