/*
 * FILE:    audio/audio_capture.h
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
 */

#ifdef __cplusplus
extern "C" {
#endif

struct audio_frame;
struct audio_params;
struct state_audio_capture;

struct audio_capture_params {
        const char *driver;
        const char *cfg;
        const struct audio_params *audio_params;
};


void                        audio_capture_init_devices(void);
void                        audio_capture_print_help(void);

/**
 * @see display_init
 */
int                         audio_capture_init(const struct audio_capture_params *,
                struct state_audio_capture **);
struct state_audio_capture *audio_capture_init_null_device(const struct audio_params *);
struct audio_frame         *audio_capture_read(struct state_audio_capture * state);
void                        audio_capture_finish(struct state_audio_capture * state);
void                        audio_capture_done(struct state_audio_capture * state);

unsigned int                audio_capture_get_vidcap_flags(const char *device_name);
unsigned int                audio_capture_get_vidcap_index(const char *device_name);
const char                 *audio_capture_get_driver_name(struct state_audio_capture * state);
/**
 * returns directly state of audio capture device. Little bit silly, but it is needed for
 * SDI (embedded sound).
 */
void                       *audio_capture_get_state_pointer(struct state_audio_capture *s);

void audio_capture_params_init(struct audio_capture_params *, const struct audio_params *);

#ifdef __cplusplus
}
#endif

/* vim: set expandtab: sw=4 */

