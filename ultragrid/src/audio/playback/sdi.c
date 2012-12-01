/*
 * FILE:    audio/playback/sdi.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
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
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <stdlib.h>

#include "audio/audio.h" 
#include "audio/audio_playback.h" 
#include "audio/playback/sdi.h" 
#include "video_display.h" 
#include "debug.h"

#define MAGIC 0xa42bf933

struct state_sdi_playback {
        void *display_state;
        uint32_t magic;
};

struct audio_playback_type *sdi_probe(void) {
        struct audio_playback_type *probe = malloc(2 * sizeof(struct audio_playback_type));
        probe[0].name = "Embedded SDI audio";
        probe[0].driver_identifier = "embedded";
        probe[1].name = NULL;
        probe[1].driver_identifier = NULL;

        return probe;
}

void * sdi_playback_init(char *cfg)
{
        struct state_sdi_playback *s = malloc(sizeof(struct state_sdi_playback));
        UNUSED(cfg);
        s->display_state = NULL;
        s->magic = MAGIC;
        return s;
}

void sdi_register_display(void *state, void *display)
{
        struct state_sdi_playback *s = (struct state_sdi_playback *) state;
        
        assert(s->magic == MAGIC);

        s->display_state = display;
}

void sdi_put_frame(void *state, struct audio_frame *frame)
{
        struct state_sdi_playback *s;
        s = (struct state_sdi_playback *) state;

        assert(s->magic == MAGIC);

        if(s->display_state)
                display_put_audio_frame(s->display_state, frame);
}

struct audio_frame * sdi_get_frame(void *state)
{
        struct state_sdi_playback *s;
        s = (struct state_sdi_playback *) state;
        
        assert(s->magic == MAGIC);

        if(s->display_state) {
                return display_get_audio_frame(s->display_state);
        } else {
                return NULL;
        }
}

int sdi_reconfigure(void *state, int quant_samples, int channels,
                int sample_rate)
{
        struct state_sdi_playback *s;
        s = (struct state_sdi_playback *) state;

        assert(s->magic == MAGIC);

        if(s->display_state) {
                return display_reconfigure_audio(s->display_state, quant_samples, channels, sample_rate);
        } else {
                return FALSE;
        }
}


void sdi_playback_done(void *s)
{
        UNUSED(s);
}

void sdi_playback_reset(void *state)
{
        struct state_sdi_playback *s;
        s = (struct state_sdi_playback *) state;

        assert(s->magic == MAGIC);

        display_audio_reset(s->display_state);
}

/* vim: set expandtab: sw=8 */

