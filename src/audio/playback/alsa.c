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
#endif

#ifdef HAVE_ALSA

#include <alsa/asoundlib.h>
#include <stdlib.h>
#include <string.h>

#include "audio/audio.h"
#include "audio/utils.h"
#include "audio/playback/alsa.h" 
#include "audio/playout_buffer.h"
#include "debug.h"

#define BUFFER_MIN 10
#define BUFFER_MAX 100

struct state_alsa_playback {
        snd_pcm_t *handle;
        struct audio_desc audio_desc;

        pthread_t thread_id;
        bool thread_started;

        struct audio_playout_buffer *playout_buffer;
};

static void *worker(void *arg);

static void *worker(void *arg)
{
        struct state_alsa_playback *s = arg;
        snd_pcm_uframes_t buffer_size,
                          period_size;
        snd_pcm_get_params(s->handle,
                        &buffer_size,
                        &period_size);

        bool stopped = false;

        double no_data_sec = 0.0;

        while (1) {
                int frames = 128;
                int data_len = frames * s->audio_desc.bps * s->audio_desc.ch_count;
                char buffer[data_len];
                int rc;

                int ret = audio_playout_buffer_read(s->playout_buffer, buffer, frames,
                                s->audio_desc.ch_count, s->audio_desc.bps, false);
                if (ret == -1)
                        return NULL;

                snd_pcm_sframes_t avail_frms = snd_pcm_avail(s->handle);
                if (ret == 0) {
                        if (stopped) {
                                usleep(1000);
                                continue;
                        }
                        snd_pcm_sframes_t avail_frms = snd_pcm_avail(s->handle);
                        long int frms_ms = s->audio_desc.sample_rate / 1000;
                        const long int wait_ms = 3;
                        unsigned long int wait_frames = wait_ms * frms_ms;

                        if (avail_frms < 0 || // error
                                        (buffer_size - avail_frms) < wait_frames) {
                                fprintf(stderr, "ALSA: Warning: Playout buffer "
                                                "underrun, %ld frames remaining "
                                                "to play and no data in playout buffer.\n",
                                                buffer_size - avail_frms);
                                memset(buffer, 0, sizeof(buffer));
                                no_data_sec += (double) frames / s->audio_desc.sample_rate;
                        } else {
                                //snd_pcm_wait(s->handle,  wait_ms);
                                usleep(1000);
                                continue;
                        }
                        if (no_data_sec > 1.0) {
                                snd_pcm_drain(s->handle);
                                stopped = true;
                        }
                } else {
                        no_data_sec = 0.0;
                        if (stopped) {
                                stopped = false;
                                snd_pcm_prepare(s->handle);
                        }
                }

#ifdef DEBUG
                fprintf(stderr, "%ld %ld\n", buffer_size - avail_frms, buffer_size);
#else
                UNUSED(avail_frms);
#endif

                if(s->audio_desc.bps == 1) { // convert to unsigned
                        signed2unsigned(buffer, buffer, data_len);
                }

                rc = snd_pcm_writei(s->handle, buffer, frames);

                if (rc == -EPIPE) {
                        /* EPIPE means underrun */
                        fprintf(stderr, "underrun occurred\n");
                        snd_pcm_prepare(s->handle);
#if 0
                        /* fill the stream with some sasmples */
                        for (double sec = 0.0; sec < BUFFER_MAX / 1000.0; sec += (double) frames / frame->sample_rate) {
                                int frames_to_write = frames;
                                if(sec + (double) frames/frame->sample_rate > BUFFER_MAX / 1000.0) {
                                        frames_to_write = (BUFFER_MAX / 1000.0 - sec) * frame->sample_rate;
                                }
                                int rc = snd_pcm_writei(s->handle, data, frames_to_write);
                                if(rc < 0) {
                                        fprintf(stderr, "error from writei: %s\n",
                                                        snd_strerror(rc));
                                        break;
                                }
                        }
#endif
                } else if (rc < 0) {
                        fprintf(stderr, "error from writei: %s\n",
                                        snd_strerror(rc));
                }  else if (rc != (int)frames) {
                        fprintf(stderr, "short write, written %d frames (overrun)\n", rc);
                }

        }
}

int audio_play_alsa_reconfigure(void *state, int quant_samples, int channels,
                                int sample_rate, struct audio_playout_buffer *playout_buffer)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;
        snd_pcm_hw_params_t *params;
        snd_pcm_format_t format;
        unsigned int val;
        int dir;
        int rc;
        snd_pcm_uframes_t frames;

        s->audio_desc.bps = quant_samples / 8;
        s->audio_desc.ch_count = channels;
        s->audio_desc.sample_rate = sample_rate;

        if(s->thread_started) {
                pthread_join(s->thread_id, NULL);
                s->thread_started = false;
        }

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
                        format = SND_PCM_FORMAT_S24_3LE;
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
                        unsigned int min_device_channels;
                        snd_pcm_hw_params_set_channels_first(s->handle, params, &min_device_channels);
                        s->audio_desc.ch_count = min_device_channels;
                } else {
                        fprintf(stderr, "cannot set requested channel count: %s\n",
                                        snd_strerror(rc));
                        return FALSE;
                }
        }

        /* we want to resample if device doesn't support default sample rate */
        val = 1;
        rc = snd_pcm_hw_params_set_rate_resample(s->handle,
                        params, val);
        if(rc < 0) {
                fprintf(stderr, "[ALSA play.] Warnings: Unable to set resampling: %s\n",
                        snd_strerror(rc));
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

        s->playout_buffer = playout_buffer;
        pthread_create(&s->thread_id, NULL, &worker, s);
        s->thread_started = true;

        return TRUE;
}

void audio_play_alsa_help(const char *driver_name)
{
        UNUSED(driver_name);
        void **hints;

        printf("\talsa %27s default ALSA device (same as \"alsa:default\")\n", ":");
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

                printf("\t%s", name);
                int i;

                for (i = 0; i < 30 - (int) strlen(name); ++i) putchar(' ');
                printf(" : %s", desc_short);
                if(desc_long) {
                        printf(" - %s", desc_long);
                }
                printf("\n");
                hints++;
                free(tmp);
                free(name);
        }
}

void * audio_play_alsa_init(char *cfg)
{
        int rc;
        struct state_alsa_playback *s;
        char *name;

        s = calloc(1, sizeof(struct state_alsa_playback));
        if(cfg && strlen(cfg) > 0) {
                if(strcmp(cfg, "help") == 0) {
                        printf("Available ALSA playback devices:\n");
                        audio_play_alsa_help(NULL);
                        free(s);
                        return &audio_init_state_ok;
                }
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

        rc = snd_pcm_nonblock(s->handle, 0);
        if(rc < 0) {
                fprintf(stderr, "ALSA Warning: Unable to set nonblock mode.\n");
        }
        
        return s;

error:
        free(s);
        return NULL;
}

void audio_play_alsa_done(void *state)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;

        if(s->thread_started) {
                pthread_join(s->thread_id, NULL);
        }

        snd_pcm_drain(s->handle);
        snd_pcm_close(s->handle);
        free(s);
}

#endif /* HAVE_ALSA */
