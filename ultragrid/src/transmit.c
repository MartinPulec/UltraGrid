/*
 * FILE:     transmit.c
 * AUTHOR:  Colin Perkins <csp@csperkins.org>
 *          Ladan Gharai
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2004 University of Southern California
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
 */

#include "rtp_transmit.h"
#include "transmit.h"

struct audio_frame;

typedef struct {
        void * (* init)(void *state);
        void (* done)(void *state);
        void (* send_tile)(void *state, int connection_nr, struct video_frame *frame, int pos, unsigned int tile_id);
        void (* send)(void *state, int connection_nr, struct video_frame *frame);
        void (* audio_tx_send)(void *state, struct audio_frame *buffer);
} transmit_t;

transmit_t available_transmit_devices[] = {
        [RTP_TRANSMIT] = { rtp_tx_init, rtp_tx_done, rtp_tx_send_tile, rtp_tx_send, rtp_audio_tx_send }
};

struct tx {
        void *state;
        enum transmit_kind kind;
};


struct tx *tx_init(enum transmit_kind kind, void *state)
{
        struct tx * res;

        res = (struct tx *) malloc(sizeof(struct tx));
        res->kind = kind;
        res->state = state;

        return res;
}

void tx_done(struct tx *tx_session)
{
        available_transmit_devices[tx_session->kind].done(tx_session->state);
}

void tx_send_tile(struct tx *tx_session, int connection_nr, struct video_frame *frame, int tile_id, unsigned int total_frames)
{
        available_transmit_devices[tx_session->kind].send_tile(tx_session->state, connection_nr, frame, tile_id, total_frames);
}

void tx_send(struct tx *tx_session, int connection_nr, struct video_frame *frame)
{
        available_transmit_devices[tx_session->kind].send(tx_session->state, connection_nr, frame);
}

void audio_tx_send(struct tx *tx_session, struct audio_frame *buffer)
{
        available_transmit_devices[tx_session->kind].audio_tx_send(tx_session->state, buffer);
}

