/*
 * FILE:    audio/playback/alsa.c
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


/*
 * Changes should use Safe ALSA API (http://0pointer.de/blog/projects/guide-to-sound-apis).
 *
 * Please, report all differencies from it here:
 * - used format SND_PCM_FORMAT_S24_LE
 * - used "default" device for arbitrary number of channels
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef HAVE_ALSA

#include <alsa/asoundlib.h>
#include <stdlib.h>
#include <string.h>
#include <tv.h>

#include "audio/audio.h"
#include "audio/utils.h"
#include "audio/audio_playback.h"
#include "audio/playback/alsa.h"
#include "debug.h"

#define BUFFER_MIN 101
#define BUFFER_MAX 200

struct state_alsa_playback {
        snd_pcm_t *handle;
        struct audio_desc settings;

        unsigned int min_device_channels;
        struct timeval t0;
        int frames;
        int total;
};

int audio_play_alsa_reconfigure(void *state, int quant_samples, int channels,
                                int sample_rate)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;
        snd_pcm_hw_params_t *params;
        snd_pcm_format_t format;
        unsigned int val;
        int dir;
        int rc;
        snd_pcm_uframes_t frames;

        s->settings.bps = quant_samples / 8;
        s->min_device_channels = s->settings.ch_count = channels;
        s->settings.sample_rate = sample_rate;

        s->frames = 0;
        s->total = 0;


        /* Allocate a hardware parameters object. */
        snd_pcm_hw_params_alloca(&params);

        /* Fill it in with default values. */
        rc = snd_pcm_hw_params_any(s->handle, params);
        if (rc < 0) {
                fprintf(stderr, "cannot obtain default hw parameters: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        /* Set the desired hardware parameters. */

        /* Interleaved mode */
        rc = snd_pcm_hw_params_set_access(s->handle, params,
                        SND_PCM_ACCESS_RW_INTERLEAVED);
        if (rc < 0) {
                fprintf(stderr, "cannot set interleaved hw access: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        switch(quant_samples) {
                case 8:
                        format = SND_PCM_FORMAT_U8;
                        break;
                case 16:
                        format = SND_PCM_FORMAT_S16_LE;
                        break;
                case 24:
                        format = SND_PCM_FORMAT_S24_LE;
                        break;
                case 32:
                        format = SND_PCM_FORMAT_S32_LE;
                        break;
                default:
                        fprintf(stderr, "[ALSA playback] Unsupported BPS for audio (%d).\n", quant_samples);
                        return FALSE;
        }
        /* Signed 16-bit little-endian format */
        rc = snd_pcm_hw_params_set_format(s->handle, params,
                        format);
        if (rc < 0) {
                fprintf(stderr, "cannot set format: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        /* Two channels (stereo) */
        rc = snd_pcm_hw_params_set_channels(s->handle, params, channels);
        if (rc < 0) {
                if(channels == 1) {
                        snd_pcm_hw_params_set_channels_first(s->handle, params, &s->min_device_channels);
                } else {
                        fprintf(stderr, "cannot set requested channel count: %s\n",
                                        snd_strerror(rc));
                        return FALSE;
                }
        }

        /* 44100 bits/second sampling rate (CD quality) */
        val = sample_rate;
        dir = 0;
        rc = snd_pcm_hw_params_set_rate_near(s->handle, params,
                        &val, &dir);
        if (rc < 0) {
                fprintf(stderr, "cannot set requested sample rate: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        /* Set period size to 1 frame. */
        frames = 1;
        dir = 1;
        rc = snd_pcm_hw_params_set_period_size_near(s->handle,
                        params, &frames, &dir);
        if (rc < 0) {
                fprintf(stderr, "cannot set period time: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }


        val = BUFFER_MIN * 1000;
        dir = 1;
        rc = snd_pcm_hw_params_set_buffer_time_min(s->handle, params,
                        &val, &dir); 
        if (rc < 0) {
                fprintf(stderr, "Warining - unable to set minimal buffer size: %s\n",
                        snd_strerror(rc));
        }

        val = BUFFER_MAX * 1000;
        dir = -1;
        rc = snd_pcm_hw_params_set_buffer_time_max(s->handle, params,
                        &val, &dir); 
        if (rc < 0) {
                fprintf(stderr, "Warining - unable to set maximal buffer size: %s\n",
                        snd_strerror(rc));
        }


        /* Write the parameters to the driver */
        rc = snd_pcm_hw_params(s->handle, params);
        if (rc < 0) {
                fprintf(stderr,
                        "unable to set hw parameters: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        return TRUE;
}

struct audio_playback_type *audio_play_alsa_probe(void)
{
        struct audio_playback_type *ret = malloc(sizeof(struct audio_playback_type));
        int count = 0;
        void **hints;

        //printf("\talsa %27s default ALSA device (same as \"alsa:default\")\n", ":");
        snd_device_name_hint(-1, "pcm", &hints); 
        while(*hints != NULL) {
                char *tmp = strdup(*(char **) hints);
                char *save_ptr = NULL;
                char *name_part;
                char *desc;
                char *desc_short;
                char *desc_long;
                char *name;


                name_part = strtok_r(tmp + 4, "|", &save_ptr);
                desc = strtok_r(NULL, "|", &save_ptr);
                desc_short = strtok_r(desc + 4, "\n", &save_ptr);
                desc_long = strtok_r(NULL, "\n", &save_ptr);

                name = malloc(strlen("alsa:") + strlen(name_part) + 1);
                strcpy(name, "alsa:");
                strcat(name, name_part);

                int cur_index = count;
                count += 1;
                ret = realloc(ret, (count + 1) * sizeof(struct audio_playback_type));

                ret[cur_index].name = strdup(desc_short);
                ret[cur_index].driver_identifier = name;

                /* if(desc_long) {
                        printf(" - %s", desc_long);
                } */
                UNUSED(desc_long);
                hints++;
                free(tmp);
        }

        ret[count].name = ret[count].driver_identifier = NULL;

        return ret;
}

void * audio_play_alsa_init(char *cfg)
{
        int rc;
        struct state_alsa_playback *s;
        char *name;

        s = calloc(1, sizeof(struct state_alsa_playback));
        if(cfg && strlen(cfg) > 0) {
                name = cfg;
        } else {
                name = "default";
        }
        rc = snd_pcm_open(&s->handle, name,
                                            SND_PCM_STREAM_PLAYBACK, 0);


        if (rc < 0) {
                    fprintf(stderr, "unable to open pcm device: %s\n",
                                    snd_strerror(rc));
                    goto error;
        }
        
        return s;

error:
        free(s);
        return NULL;
}

struct audio_frame *audio_play_alsa_get_frame(void *state)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;
        UNUSED(s);

        return NULL;
}

void audio_play_alsa_reset(void *state)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;

        int err;

        if ((err = snd_pcm_drop(s->handle)) < 0)
        {
                fprintf(stderr, "Alsa reset failed\n");
                return;
        }
        if ((err = snd_pcm_prepare(s->handle)) < 0)
        {
                fprintf(stderr, "Alsa prepare failed\n");
                return;
        }
        return;
}

void audio_play_alsa_put_frame(void *state, struct audio_frame *frame)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;
        int rc;
        char *data = frame->data;
        int frames = frame->data_len / (s->settings.bps * s->settings.ch_count);

#ifdef DEBUG
        snd_pcm_sframes_t delay;
        snd_pcm_delay(s->handle, &delay);
        printf("Alsa delay: %d samples (%u Hz)\n", (int)delay, (unsigned int) s->settings.sample_rate);
#endif

        if(s->settings.bps == 1) { // convert to unsigned
                signed2unsigned(frame->data, frame->data, frame->data_len);
        }

        if((int) s->min_device_channels > frame->ch_count && frame->ch_count == 1) {
                if(frame->data_len * s->min_device_channels < frame->max_size) {
                        audio_frame_multiply_channel(frame, s->min_device_channels);
                }
        }

        rc = snd_pcm_writei(s->handle, data, frames);
        if (rc == -EPIPE) {
                /* EPIPE means underrun */
                printf("underrun occurred\n");
                snd_pcm_prepare(s->handle);

                /* fill the stream with some sasmples */
                for (double sec = 0.0; sec < BUFFER_MIN / 1000.0; sec += (double) frames / s->settings.sample_rate) {
                        int frames_to_write = frames;
                        if(sec + (double) frames/s->settings.sample_rate > BUFFER_MIN / 1000.0) {
                                frames_to_write = (BUFFER_MIN / 1000.0 - sec) * s->settings.sample_rate;
                        }
                        assert(frames_to_write > 0);
                        int rc = snd_pcm_writei(s->handle, data, frames_to_write);
                        if(rc < 0) {
                                fprintf(stderr, "error from writei: %s\n",
                                                snd_strerror(rc));
                                break;
                        }
                }
        } else if (rc < 0) {
                fprintf(stderr, "error from writei: %s\n",
                        snd_strerror(rc));
        }  else if (rc != (int)frames) {
                fprintf(stderr, "short write, write %d frames\n", rc);
        }

        if(s->frames == 0) {
                gettimeofday(&s->t0, NULL);
                s->frames += frames;
        } else {
                struct timeval t;
                gettimeofday(&t, NULL);
                printf("%f seconds, %d frames\n", tv_diff(t,s->t0),s->frames);
                ///s->t0 = t;
                s->frames += frames;
        }

        s->total += 1;
}

void audio_play_alsa_done(void *state)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;

        snd_pcm_drain(s->handle);
        snd_pcm_close(s->handle);
        free(s);
}

#endif /* HAVE_ALSA */
