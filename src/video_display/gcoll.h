/*
 * FILE:   display_null.h
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
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 */

#define DISPLAY_GCOLL_ID       0x16e271fb
#include "video.h"
#include "video_display.h" // display_type_t
#include "host.h"
#include "rtp/rtp.h"

#define GCOLL_FRONT  0
#define GCOLL_SIDE   1
#define GCOLL_GROUP  2

struct gcoll_init_params {
        uint32_t front_ssrc;
        uint32_t side_ssrc;

        bool send_group_camera;
        uint32_t group_ssrc;

        bool send_audio;
        uint32_t audio_ssrc;

        uint32_t group_id;

        const char *reflector_addr;
	int port_number;

	struct rtp *rtp_session[CAP_DEV_COUNT];
};


struct audio_frame;

display_type_t		*display_gcoll_probe(void);
/**
 * @param udata          struct gcoll_init_params cast to (void *)
 */
void 			*display_gcoll_init(char *fmt, unsigned int flags, void *udata);
void 			 display_gcoll_run(void *state);
void 			 display_gcoll_finish(void *state);
void 			 display_gcoll_done(void *state);
/**
 * Puts video frame to driver.
 * Driver is responsible for freeing the video frame.
 */
int 			 display_gcoll_putf(void *state, struct video_frame *frame);
int                      display_gcoll_get_property(void *state, int property, void *val, size_t *len);

