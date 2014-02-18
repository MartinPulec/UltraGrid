/**
 * @file   audio/playout_buffer.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2014, CESNET z.s.p.o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
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
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "audio/playout_buffer.h"

#include "audio/utils.h"

#include "utils/ring_buffer.h"

#define BUF_SIZE (1024 * 1024)

struct audio_playout_buffer {
        pthread_mutex_t lock;
        pthread_cond_t cv;
        struct ring_buffer *buffer;
        struct audio_desc saved_desc;
        bool poisoned;

        int net_frame_size;
        int samples_per_frame;          ///< average frame size
        int samples_per_frame_avg_diff; ///< average difference to average frame size

        char *tmp;
};

int audio_playout_buffer_init(struct audio_playout_buffer **state)
{
        struct audio_playout_buffer *s = calloc(1, sizeof(struct audio_playout_buffer));
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->cv, NULL);
        s->tmp = malloc(BUF_SIZE * 3);

        *state = s;

        return 0;
}

void audio_playout_buffer_destroy(struct audio_playout_buffer *s)
{
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->cv);
        ring_buffer_destroy(s->buffer);
        free(s->tmp);
        free(s);
}

void audio_playout_buffer_write(struct audio_playout_buffer *s, struct audio_frame *frame)
{
        int d;
        pthread_mutex_lock(&s->lock);
        if (!audio_desc_eq(s->saved_desc, audio_desc_from_audio_frame(frame))) {
                ring_buffer_destroy(s->buffer);
                s->buffer = ring_buffer_init(frame->bps * frame->ch_count * frame->sample_rate);
                s->saved_desc = audio_desc_from_audio_frame(frame);
        }

        ring_buffer_write(s->buffer, frame->data, frame->data_len);

        int samples = frame->data_len / frame->bps / frame->ch_count;
        d = s->samples_per_frame - samples;
        if (d < 0) {
                d = -d;
        }
        s->samples_per_frame = (samples + 4 * s->samples_per_frame) / 5;
        s->samples_per_frame_avg_diff = (d + 4 * s->samples_per_frame_avg_diff) / 5;

        pthread_cond_signal(&s->cv);

        pthread_mutex_unlock(&s->lock);
}

void audio_playout_buffer_get_avg_frame_len(struct audio_playout_buffer *s,
                int *avg_len, int *avg_diff)
{
        pthread_mutex_lock(&s->lock);
        *avg_len = s->samples_per_frame;
        *avg_diff = s->samples_per_frame_avg_diff;
        pthread_mutex_unlock(&s->lock);
}


int audio_playout_buffer_read(struct audio_playout_buffer *s, char *buffer,
                int samples, int ch_count, int bps, bool blocking)
{
        int ret;
        int read_size = s->saved_desc.ch_count * s->saved_desc.bps * samples;
        pthread_mutex_lock(&s->lock);
        while(!s->poisoned &&
                        (!s->buffer || ring_get_current_size(s->buffer) < read_size))
        {
                if (!blocking) {
                        pthread_mutex_unlock(&s->lock);
                        return 0;
                }
                pthread_cond_wait(&s->cv, &s->lock);
        }
        if(s->poisoned) {
                ret = -1;
        } else {
                if(ch_count == s->saved_desc.ch_count && bps == s->saved_desc.bps) {
                        ret = ring_buffer_read(s->buffer, buffer, samples * ch_count * bps);
                } else {
                        ring_buffer_read(s->buffer, s->tmp, read_size);
                        ret = samples * ch_count * bps;
                        assert(ret <= BUF_SIZE);
                        if(ch_count == s->saved_desc.ch_count) {
                                change_bps(buffer, bps, s->tmp,  s->saved_desc.bps, read_size);
                        } else {
                                char *tmp_channel = s->tmp + BUF_SIZE;
                                char *write_channel;
                                int write_channel_bytes;

                                if(bps != s->saved_desc.bps) {
                                        write_channel = tmp_channel;
                                        write_channel_bytes = read_size / s->saved_desc.ch_count;
                                } else {
                                        write_channel = s->tmp + 2 * BUF_SIZE;
                                        write_channel_bytes = bps * samples;
                                }

                                demux_channel(tmp_channel, s->tmp, bps, read_size, s->saved_desc.ch_count,
                                                0);
                                if(bps != s->saved_desc.bps) {
                                        change_bps(write_channel, bps, tmp_channel, s->saved_desc.bps,
                                                        read_size / s->saved_desc.ch_count);
                                }
                                mux_channel(buffer, write_channel, bps, write_channel_bytes,
                                                ch_count, 0, 1.0);
                                // for every extra channel, copy first stream to it
                                for(int i = s->saved_desc.ch_count; i < ch_count; ++i) {
                                        mux_channel(buffer, tmp_channel, bps, read_size / s->saved_desc.ch_count, ch_count, 1, 1.0);
                                }

                                // finally, copy all remaining channels
                                for(int i = 1; i < s->saved_desc.ch_count && i < ch_count;
                                                ++i) {
                                        demux_channel(tmp_channel, s->tmp, bps, read_size,
                                                        s->saved_desc.ch_count, i);
                                        if(bps != s->saved_desc.bps) {
                                                change_bps(write_channel, bps, tmp_channel, s->saved_desc.bps,
                                                                read_size / s->saved_desc.ch_count);
                                        }
                                        mux_channel(buffer, write_channel, bps, write_channel_bytes,
                                                        ch_count, i, 1.0);
                                }

                        }
                }
        }
        pthread_mutex_unlock(&s->lock);

        return ret;
}

void audio_playout_buffer_poison(struct audio_playout_buffer *s)
{
        pthread_mutex_lock(&s->lock);
        s->poisoned = true;
        pthread_cond_signal(&s->cv);
        pthread_mutex_unlock(&s->lock);
}

