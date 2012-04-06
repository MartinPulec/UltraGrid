/*
 * FILE:    exr.c
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
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */
#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "video_codec.h"
#include "video_capture.h"

#include "tv.h"

#include "video_capture/exr.h"
//#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <glob.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/poll.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <semaphore.h>
#include "video_capture.h"
#include "utils/thread_pool.h"
#include <ImfCRgbaFile.h>


#define SIGN(x) (x / fabs(x))
#define ROUND_FROM_ZERO(x) (ceil(fabs(x)) * SIGN(x))

#define BUFFER_LEN 30
#define THREADS 6

void * job_process(void *input);


struct job {
        int frame_id;
        char *filename;
        char *out_data;

        ImfRgba *scanline;
        int used:1;

        int slot;

        struct vidcap_exr_state *state;
};

struct vidcap_exr_state {
        struct video_frame *frame;
        struct tile        *tile;

        pthread_mutex_t lock;

        pthread_cond_t reader_cv;
        volatile int reader_waiting;

        volatile int should_pause;
        pthread_cond_t pause_cv;

        unsigned int        loop:1;
        volatile unsigned int        finished;
        volatile unsigned int        should_exit_thread:1;

        struct  job         jobs[THREADS];

        char               *buffer_read[BUFFER_LEN];
        /* those are buffers which are computed
         * but not yet make public (eg. because they are disconginout - previous frame missing */
        int                 computed_buffers[BUFFER_LEN];
        /* ring buffer - start - first readable frame; end - after last readable frame
         * working read end is ahead buffer_read_end and tells which will be the next element that will
         * go (asynchronously) computed */
        volatile int        buffer_read_start, buffer_read_end, working_read_end;

        char               *buffer_send;

        pthread_t           reading_thread;
        int                 frames;
        struct timeval      t, t0;

        glob_t              glob;
        int                 index;

        int                *lut;
        float               gamma;

        struct timeval      prev_time, cur_time;

        unsigned int        should_jump:1;
        unsigned int        grab_waiting:1;
        unsigned int        playone;

        int                 min_x, min_y, max_x, max_y;

        double              speed;

        struct thread_pool *pool;
};


static void setpos(struct vidcap_exr_state *s, int i);
static void * reading_thread(void *args);
static void usage(void);

static int clamp_to_range(int value);

static int clamp_to_range(int value)
{
        if(value < 0)
                return 0;
        if(value > 65535)
                return 65535;
        return value;
}

static void usage()
{
        printf("EXR video capture usage:\n");
        printf("\t-t exr:files=<glob>[:fps=<fps>:gamma=<gamma>:loop]\n");
}

struct vidcap_type *
vidcap_exr_probe(void)
{
	struct vidcap_type*		vt;

	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_EXR_ID;
		vt->name        = "exr";
		vt->description = "OpenEXR";
	}
	return vt;
}

void *
vidcap_exr_init(char *fmt, unsigned int flags)
{
        UNUSED(flags);
	struct vidcap_exr_state *s;
        char *item;
        char *glob_pattern;
        char *save_ptr = NULL;
        int i;

	printf("vidcap_exr_init\n");

        s = (struct vidcap_exr_state *) calloc(1, sizeof(struct vidcap_exr_state));

        if(!fmt || strcmp(fmt, "help") == 0) {
                usage();
                return NULL;
        }

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->reader_cv, NULL);
        pthread_cond_init(&s->pause_cv, NULL);
        s->reader_waiting = 0;
        s->should_pause = FALSE;
        s->should_jump = FALSE;
        s->grab_waiting = FALSE;

        s->buffer_read_start = s->buffer_read_end = s->working_read_end = 0;
        s->index = 0;
        s->speed = 1.0;

        s->should_exit_thread = FALSE;
        s->finished = 0;

        s->frame = vf_alloc(1);
        s->frame->fps = 30.0;
        s->frame->frames = -1;
        s->tile = &s->frame->tiles[0];
        s->gamma = 1.0;
        s->loop = FALSE;
        s->playone = 0;

        item = strtok_r(fmt, ":", &save_ptr);
        while(item) {
                if(strncmp("files=", item, strlen("files=")) == 0) {
                        glob_pattern = item + strlen("files=");
                } else if(strncmp("fps=", item, strlen("fps=")) == 0) {
                        s->frame->fps = atof(item + strlen("fps="));
                } else if(strncmp("gamma=", item, strlen("gamma=")) == 0) {
                        s->gamma = atof(item + strlen("gamma="));
                } else if(strncmp("colorspace=", item, strlen("colorspace=")) == 0) {
                        // TODO!!!!!!
                } else if(strncmp("loop", item, strlen("loop")) == 0) {
                        s->loop = TRUE;
                }

                item = strtok_r(NULL, ":", &save_ptr);
        }

        int ret = glob(glob_pattern, 0, NULL, &s->glob);
        if (ret)
        {
                perror("Opening exr files failed");
                return NULL;
        }

        char *filename = s->glob.gl_pathv[0];

        ImfInputFile *file = ImfOpenInputFile(filename);
        const ImfHeader *hdr_info;
        hdr_info  = ImfInputHeader(file);
        ImfHeaderDataWindow(hdr_info, &s->min_x, &s->min_y, &s->max_x, &s->max_y);

        ImfCloseInputFile(file);

        s->tile->width = s->max_x - s->min_x + 1UL;
        s->tile->height = s->max_y - s->min_y + 1UL;

        s->frame->color_spec = RGB16;
        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) * s->tile->height;


        for (i = 0; i < BUFFER_LEN; ++i) {
                s->buffer_read[i] = (char *) malloc(s->tile->data_len);
        }
        s->buffer_send = (char *) malloc(s->tile->data_len);

        for(i = 0; i < THREADS; ++i) {
                s->jobs[i].scanline = (ImfRgba *) malloc(s->tile->width * sizeof(ImfRgba));
                s->jobs[i].used = FALSE;
                s->jobs[i].state = s;
        }

        s->prev_time.tv_sec = s->prev_time.tv_usec = 0;

        s->pool = thread_pool_init(THREADS, job_process);

        pthread_create(&s->reading_thread, NULL, reading_thread, s);

	return s;
}

void
vidcap_exr_finish(void *state)
{
	struct vidcap_exr_state *s = (struct vidcap_exr_state *) state;
        pthread_mutex_lock(&s->lock);
        s->should_pause = FALSE;
        pthread_cond_signal(&s->pause_cv);

        s->should_exit_thread = TRUE;
        s->finished = TRUE;
        pthread_mutex_unlock(&s->lock);
}

void
vidcap_exr_done(void *state)
{
        return;
	struct vidcap_exr_state *s = (struct vidcap_exr_state *) state;
        int i;
	assert(s != NULL);

        pthread_mutex_lock(&s->lock);
        s->should_exit_thread = TRUE;
        if(s->reader_waiting)
                pthread_cond_broadcast(&s->reader_cv);
        pthread_mutex_unlock(&s->lock);

        vf_free(s->frame);
        for (i = 0; i < BUFFER_LEN; ++i) {
                free(s->buffer_read[i]);
        }
        free(s->buffer_send);
        thread_pool_destroy(s->pool);
        free(s);
}

static int find_unused_slot(struct job * j, int count)
{
        int i;
        for (i = 0; i < count; ++i) {
                if(!j[i].used) {
                        return i;
                }
        }

        return -1;
}

static void reset_reading(struct vidcap_exr_state *s) {
        int i;
        for(i = 0; i < THREADS; ++i) {
                s->jobs[i].used = FALSE;
        }
        memset(s->computed_buffers, 0, BUFFER_LEN * sizeof(int));
}

static void * reading_thread(void *args)
{
	struct vidcap_exr_state        *s = (struct vidcap_exr_state *) args;

        memset(s->computed_buffers, 0, BUFFER_LEN * sizeof(int));

        /* temporary buffer for computed data, 0 goes for buffer_read_end, etc. */

        while(1) {
                pthread_mutex_lock(&s->lock);
                if(s->should_exit_thread) {
                        pthread_mutex_unlock(&s->lock);
                        goto after_while;
                }


                /*if((s->speed > 0.0 && my_index >= (int) s->glob.gl_pathc) ||
                                my_index < 0) {
                        ++s->finished;
                }*/

                while((s->finished || s->should_jump || s->buffer_read_start == ((s->working_read_end + 1) % BUFFER_LEN))) { /* full */
                        s->reader_waiting = TRUE;
                        pthread_cond_wait(&s->reader_cv, &s->lock);
                        s->reader_waiting = FALSE;
                        if(s->should_exit_thread) {
                                pthread_mutex_unlock(&s->lock);
                                goto after_while;
                        }
                }

                // UNLOCK - UNLOCK - UNLOCK
                pthread_mutex_unlock(&s->lock);

                int was_last = FALSE;

                if(s->index >= (int) s->glob.gl_pathc) {
                        if(s->index != (int) s->glob.gl_pathc - 1 + ROUND_FROM_ZERO(s->speed)) {
                                s->index = (int) s->glob.gl_pathc - 1;
                        }
                }

                if(s->index < 0) {
                        if(s->index != ROUND_FROM_ZERO(s->speed)) {
                                s->index = 0;
                        }
                }

                if( (s->speed > 0.0 && s->index >= (int) s->glob.gl_pathc) ||
                                s->index < 0) {
                        was_last = TRUE;
                }

                if(thread_pool_get_overall_count(s->pool) < THREADS && !was_last) {
                        int slot = find_unused_slot(s->jobs, THREADS);
                        assert(slot >= 0);

                        char *filename = s->glob.gl_pathv[s->index];

                        s->jobs[slot].frame_id = s->index;
                        s->jobs[slot].filename = filename;

                        s->jobs[slot].slot = s->working_read_end; //s->buffer_read_end;
                        s->jobs[slot].out_data = s->buffer_read[s->working_read_end]; //s->buffer_read_end;

                        s->jobs[slot].used = TRUE;

                        thread_pool_enqueue(s->pool, &s->jobs[slot]);

                        s->index += ROUND_FROM_ZERO(s->speed);
                        s->working_read_end = (s->working_read_end + 1) % BUFFER_LEN;
                }

                if(thread_pool_get_overall_count(s->pool) == THREADS || was_last) {
                        struct job * res = thread_pool_pop(s->pool);
                        res->used = FALSE;

                        s->computed_buffers[res->slot] = TRUE;
                }

                // LOCK - LOCK - LOCK
                pthread_mutex_lock(&s->lock);

                while(s->computed_buffers[s->buffer_read_end]) {
                        s->computed_buffers[s->buffer_read_end] = FALSE;

                        s->buffer_read_end = (s->buffer_read_end + 1) % BUFFER_LEN; /* and we will read next one */
                }


                pthread_mutex_unlock(&s->lock);

                if(was_last && thread_pool_get_overall_count(s->pool) == 0) {
                        s->finished = TRUE;
                }
        }
after_while:

        while(!s->should_exit_thread)
                ;


        return NULL;
}

struct video_frame *
vidcap_exr_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_exr_state 	*s = (struct vidcap_exr_state *) state;

        // LOCK - LOCK - LOCK - LOCK
        pthread_mutex_lock(&s->lock);
        while((s->should_pause || s->should_jump) && !s->playone) {
                s->grab_waiting = TRUE;
                pthread_cond_wait(&s->pause_cv, &s->lock);
                s->grab_waiting = FALSE;
        }
        if(s->playone > 0) {
                s->playone--;
        }

        if(s->finished &&
                        s->buffer_read_start == s->buffer_read_end) {
                if(s->loop) {
                        s->finished = FALSE;
                        s->grab_waiting = TRUE;
                        if(s->speed > 0.0) {
                                setpos(s, 0);
                        } else {
                                setpos(s, s->glob.gl_pathc - 1);
                        }
                        s->grab_waiting = FALSE;
                } else  {
                        pthread_mutex_unlock(&s->lock);
                        return NULL;
                }
        }

        // UNLOCK - UNLOCK - UNLOCK - UNLOCK
        pthread_mutex_unlock(&s->lock);

        while(s->buffer_read_start == s->buffer_read_end && !should_exit && !s->finished && !s->should_jump)
                ;

        if(s->should_jump)
                return NULL;

        if(s->prev_time.tv_sec == 0 && s->prev_time.tv_usec == 0) { /* first run */
                gettimeofday(&s->prev_time, NULL);
        }

        gettimeofday(&s->cur_time, NULL);
        if(s->frame->fps == 0) /* it would make following loop infinite */
                return NULL;
        while(tv_diff_usec(s->cur_time, s->prev_time) < 1000000.0 / s->frame->fps / (fabs(s->speed) < 1.0 ? fabs(s->speed) : 1.0)) {
                gettimeofday(&s->cur_time, NULL);
        }
        s->prev_time = s->cur_time;
        //tv_add_usec(&s->prev_time, 1000000.0 / s->frame->fps);

        s->tile->data = s->buffer_read[s->buffer_read_start];
        s->buffer_read[s->buffer_read_start] = s->buffer_send;
        s->buffer_send = s->tile->data;

        pthread_mutex_lock(&s->lock);
        s->buffer_read_start = (s->buffer_read_start + 1) % BUFFER_LEN;
        if(s->reader_waiting)
                        pthread_cond_broadcast(&s->reader_cv);
        pthread_mutex_unlock(&s->lock);

        s->frames++;
        s->frame->frames += ROUND_FROM_ZERO(s->speed);
        if( s->frame->frames >= (int) s->glob.gl_pathc) {
                s->frame->frames = s->glob.gl_pathc - 1;
        }
        if( s->frame->frames < 0) {
                s->frame->frames = 0;
        }


        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);
        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            fprintf(stderr, "%d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }

        *audio = NULL;

	return s->frame;
}

static void flush_pipeline(struct vidcap_exr_state *s)
{
        s->should_jump = TRUE;
        pthread_mutex_unlock(&s->lock);
        while(!s->reader_waiting || !s->grab_waiting)
                ;

        pthread_mutex_lock(&s->lock);
        reset_reading(s);
        s->buffer_read_start = s->buffer_read_end = s->working_read_end = 0;
        thread_pool_flush(s->pool);
        assert(thread_pool_get_overall_count(s->pool) == 0);

}

static void play_after_flush(struct vidcap_exr_state *s)
{
        s->finished = FALSE;
        s->should_jump = FALSE;
        pthread_cond_broadcast(&s->reader_cv);
        if(!s->should_pause)
                pthread_cond_signal(&s->pause_cv);

}

/* must be called locked !!!!! */
static void setpos(struct vidcap_exr_state *s, int i)
{
                flush_pipeline(s);

                s->index = i;
                fprintf(stderr, "New position: %d\n", s->index);
                s->frame->frames = s->index - 1;

                play_after_flush(s);
}

void vidcap_exr_command(struct vidcap *state, int command, void *data)
{
	struct vidcap_exr_state 	*s = (struct vidcap_exr_state *) state;


        if(command == VIDCAP_PAUSE) {
                pthread_mutex_lock(&s->lock);
                s->should_pause = TRUE;
                pthread_mutex_unlock(&s->lock);
        } else if(command == VIDCAP_PLAY) {
                pthread_mutex_lock(&s->lock);
                s->should_pause = FALSE;
                pthread_cond_signal(&s->pause_cv);
                pthread_mutex_unlock(&s->lock);
        } else if(command == VIDCAP_PLAYONE) {
                pthread_mutex_lock(&s->lock);
                s->playone = *(int *) data;
                pthread_cond_signal(&s->pause_cv);
                pthread_mutex_unlock(&s->lock);
        } else if(command == VIDCAP_FPS) {
                pthread_mutex_lock(&s->lock);
                s->frame->fps = *(float *) data;
                pthread_mutex_unlock(&s->lock);
        } else if(command == VIDCAP_POS) {
                pthread_mutex_lock(&s->lock);
                setpos(s, *(int *) data);
                pthread_mutex_unlock(&s->lock);
        } else if(command == VIDCAP_LOOP) {
                pthread_mutex_lock(&s->lock);
                s->loop = *(int *) data;
                pthread_mutex_unlock(&s->lock);
        } else if(command == VIDCAP_SPEED) {
                fprintf(stderr, "[OpenEXR] SPEED %f\n", *(float *) data);
                pthread_mutex_lock(&s->lock);
                flush_pipeline(s);

                s->index = s->frame->frames + 1;
                //s->frame->frames  = s->index - 1;
                s->speed = *(float *) data;

                play_after_flush(s);
                pthread_mutex_unlock(&s->lock);
                /*pthread_mutex_lock(&s->lock);
                s->loop = *(int *) data;
                pthread_mutex_unlock(&s->lock);*/
        }
}

void * job_process(void *input)
{
        struct job * job = input;
        ImfInputFile *file = ImfOpenInputFile(job->filename);

        int y;
        for (y = 0; y < (int) job->state->tile->height; ++y) {
                int x;
                uint16_t *line = (uint16_t *) job->out_data + (3 * y * job->state->tile->width);

                ImfInputSetFrameBuffer(file, job->scanline - job->state->min_x - job->state->tile->width * (job->state->min_y + y), 1,
                                      job->state->tile->width);
                ImfInputReadPixels(file, job->state->min_y + y, job->state->min_y + y);
                for(x = 0; x < (int) job->state->tile->width; ++x) {
                        line[0] = clamp_to_range(round(ImfHalfToFloat(job->scanline[x].r) * 65535.0));
                        line[1] = clamp_to_range(round(ImfHalfToFloat(job->scanline[x].g) * 65535.0));
                        line[2] = clamp_to_range(round(ImfHalfToFloat(job->scanline[x].b) * 65535.0));
                        //line[3] = clamp_to_range(round(ImfHalfToFloat(job->scanline[x].a) * 65535.0));
                        line += 3;
                }
        }

        ImfCloseInputFile(file);

        return input;
}

