/*
 * FILE:    vf.c
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
#include "host.h"
#include "video_codec.h"
#include "video_capture.h"

#include "video_file.h"
#include "tv.h"

#include "video_capture/vf.h"
//#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <glob.h>
#include <libgen.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/poll.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <semaphore.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>

#include "video_capture.h"

#define BUFFER_LEN 10

#define SIGN(x) (x / fabs(x))
#define ROUND_FROM_ZERO(x) (ceil(fabs(x)) * SIGN(x))

using namespace std;

struct vidcap_vf_state {
        struct video_desc video_prop;
        pthread_mutex_t lock;

        pthread_cond_t boss_cv;
        pthread_cond_t reader_cv;
        pthread_cond_t processing_cv;
        volatile int reader_waiting;
        volatile int processing_waiting;

        volatile int should_pause;
        pthread_cond_t pause_cv;

        unsigned int        loop:1;
        volatile unsigned int        finished:1;
        volatile unsigned int        should_exit_thread:1;

        queue<struct video_frame *> read_queue;
        queue<struct video_frame *> processed_queue;

        char               *buffer_send;

        pthread_t           reading_thread, processing_thread;
        int                 frames;
        struct timeval      t, t0;

        glob_t              glob;

        struct timeval      prev_time, cur_time;

        unsigned int        should_jump:1;
        volatile unsigned int        grab_waiting:1;
        unsigned int        play_to_buffer;

        float               speed; // TODO: remove
        float               speedup;

        int                 seq_num;

        enum filetype       file_type;
};


static void * reading_thread(void *args);
static void * processing_thread(void *args);
static void usage(void);

static void usage()
{
        printf("DPX video capture usage:\n");
        printf("\t-t dpx:files=<glob>[:fps=<fps>:gamma=<gamma>:loop]\n");
}

struct vidcap_type *
vidcap_vf_probe(void)
{
	struct vidcap_type*		vt;

	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_VF_ID;
		vt->name        = "vf";
		vt->description = "Video frame";
	}
	return vt;
}

void *
vidcap_vf_init(char *fmt, unsigned int flags)
{
        UNUSED(flags);

	struct vidcap_vf_state *s;
        char *item;
        char *glob_pattern;
        char *save_ptr = NULL;
        int i;

	printf("vidcap_vf_init\n");

        // call constructor in order to the constructors of involved objects to be called
        s = new vidcap_vf_state;

        if(!fmt || strcmp(fmt, "help") == 0) {
                usage();
                return NULL;
        }

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->processing_cv, NULL);
        pthread_cond_init(&s->reader_cv, NULL);
        pthread_cond_init(&s->pause_cv, NULL);
        pthread_cond_init(&s->boss_cv, NULL);
        s->processing_waiting = FALSE;
        s->reader_waiting = FALSE;
        s->should_pause = TRUE;
        s->should_jump = FALSE;
        s->grab_waiting = FALSE;

        s->seq_num = 0;

        s->should_exit_thread = FALSE;
        s->finished = FALSE;

        s->video_prop.tile_count = 1;
        s->video_prop.fps = 30.0;
        s->loop = FALSE;
        s->play_to_buffer = 0;
        s->speed = 1.0;
        s->speedup = 1.0;

        item = strtok_r(fmt, ":", &save_ptr);
        while(item) {
                if(strncmp("files=", item, strlen("files=")) == 0) {
                        glob_pattern = item + strlen("files=");
                        // global variable in main.c
                        char *tmp = strdup(glob_pattern);
                        video_directory = strdup(dirname(tmp));
                        free(tmp);
                } else if(strncmp("fps=", item, strlen("fps=")) == 0) {
                        s->video_prop.fps = atof(item + strlen("fps="));
                } else if(strncmp("colorspace=", item, strlen("colorspace=")) == 0) {
                        /// TODO!!!!
                } else if(strncmp("loop", item, strlen("loop")) == 0) {
                        s->loop = TRUE;
                }

                item = strtok_r(NULL, ":", &save_ptr);
        }

        int ret = glob(glob_pattern, 0, NULL, &s->glob);
        if (ret)
        {
                fprintf(stderr, "Opening VF files failedi (%s)", glob_pattern);
                perror("");
                return NULL;
        }

        // pick one file to get properties from
        char *filename = s->glob.gl_pathv[0];
        char *extension = strrchr(filename, '.');
        if(!extension) {
                throw runtime_error("Unknown extension");
        } else {
                extension += 1;  // we won't to have the dot
        }

        s->file_type = video_file::get_filetype_to_ext(extension);
        std::shared_ptr<video_file> file(
                        video_file::create(s->file_type, filename));

        struct video_desc frame_desc = file->get_video_desc();
        s->video_prop.width = frame_desc.width;
        s->video_prop.height = frame_desc.height;
        s->video_prop.color_spec = frame_desc.color_spec;
        s->video_prop.tile_count = 1;
        s->video_prop.interlacing = PROGRESSIVE;

        pthread_create(&s->reading_thread, NULL, reading_thread, s);
        pthread_create(&s->processing_thread, NULL, processing_thread, s);

        s->prev_time.tv_sec = s->prev_time.tv_usec = 0;

	return s;
}

void
vidcap_vf_finish(void *state)
{
	struct vidcap_vf_state *s = (struct vidcap_vf_state *) state;
        pthread_mutex_lock(&s->lock);
        s->should_pause = FALSE;
        pthread_cond_signal(&s->pause_cv);

        s->should_exit_thread = TRUE;
        s->finished = TRUE;
        pthread_mutex_unlock(&s->lock);
}

void
vidcap_vf_done(void *state)
{
	struct vidcap_vf_state *s = (struct vidcap_vf_state *) state;
        int i;
	assert(s != NULL);

        pthread_mutex_lock(&s->lock);
        s->should_exit_thread = TRUE;
        if(s->reader_waiting)
                pthread_cond_signal(&s->reader_cv);
        if(s->processing_waiting)
                pthread_cond_signal(&s->processing_cv);
        pthread_mutex_unlock(&s->lock);

	pthread_join(s->reading_thread, NULL);
	pthread_join(s->processing_thread, NULL);

        while(!s->read_queue.empty()) {
                struct video_frame *frame = s->read_queue.front();
                vf_free_data(frame);
                s->read_queue.pop();
        }
        while(!s->processed_queue.empty()) {
                struct video_frame *frame = s->processed_queue.front();
                vf_free_data(frame);
                s->processed_queue.pop();
        }

        delete s;
        fprintf(stderr, "vf exited\n");
}

static void * reading_thread(void *args)
{
	struct vidcap_vf_state 	*s = (struct vidcap_vf_state *) args;

        while(1) {
                pthread_mutex_lock(&s->lock);
                if(s->should_exit_thread) {
                        pthread_mutex_unlock(&s->lock);
                        goto after_while;
                }
                while(s->finished || s->should_jump || s->read_queue.size() == BUFFER_LEN) { /* full */
                        s->reader_waiting = TRUE;
                        pthread_cond_wait(&s->reader_cv, &s->lock);
                        s->reader_waiting = FALSE;
                        if(s->should_exit_thread) {
                                pthread_mutex_unlock(&s->lock);
                                goto after_while;
                        }
                }

                pthread_mutex_unlock(&s->lock);
                
                struct video_frame *frame = vf_alloc_desc(s->video_prop);
                frame->frames = s->seq_num;

                char *filename = s->glob.gl_pathv[s->seq_num];
                s->seq_num += SIGN(s->speed);

                if(s->seq_num >= (int) s->glob.gl_pathc) {
                        if(s->seq_num != (int) s->glob.gl_pathc - 1 + SIGN(s->speed)) {
                                s->seq_num = (int) s->glob.gl_pathc - 1;
                        }
                }

                if(s->seq_num < 0) {
                        if(s->seq_num != SIGN(s->speed)) {
                                s->seq_num = 0;
                        }
                }

                std::shared_ptr<video_file> file(
                        video_file::create(s->file_type, filename));
                
                int len;
                char *data = file->get_raw_data(len);
                frame->tiles[0].data = data;
                frame->tiles[0].data_len = len;

                pthread_mutex_lock(&s->lock);
                s->read_queue.push(frame);
                if(s->processing_waiting)
                        pthread_cond_signal(&s->processing_cv);

                if( (s->speed > 0.0 && s->seq_num >= (int) s->glob.gl_pathc) ||
                                s->seq_num < 0) {
                        s->finished = TRUE;
                }
                pthread_mutex_unlock(&s->lock);
        }
after_while:

        while(!s->should_exit_thread)
                ;

        return NULL;
}

static void * processing_thread(void *args)
{
	struct vidcap_vf_state 	*s = (struct vidcap_vf_state *) args;

        while(1) {
                pthread_mutex_lock(&s->lock);
                if(s->should_exit_thread) {
                        pthread_mutex_unlock(&s->lock);
                        break;
                }
                while(s->should_jump || s->processed_queue.size() == BUFFER_LEN) { /* full */
                        s->processing_waiting = TRUE;
                        pthread_cond_wait(&s->processing_cv, &s->lock);
                        s->processing_waiting = FALSE;
                        if(s->should_exit_thread) {
                                pthread_mutex_unlock(&s->lock);
                                goto after_while;
                        }
                }
                pthread_mutex_unlock(&s->lock);

                pthread_mutex_lock(&s->lock);
                if(s->should_exit_thread) {
                        break;
                        pthread_mutex_unlock(&s->lock);
                }
                while(s->read_queue.empty()) { /* empty */
                        s->processing_waiting = TRUE;
                        pthread_cond_wait(&s->processing_cv, &s->lock);
                        s->processing_waiting = FALSE;
                        if(s->should_exit_thread) {
                                pthread_mutex_unlock(&s->lock);
                                goto after_while;
                        }
                }

                assert(!s->read_queue.empty());
                struct video_frame *src = s->read_queue.front();
                struct video_frame *dst = NULL;
                s->read_queue.pop();

                pthread_mutex_unlock(&s->lock);

                if(0) {
                        // here will be processing
#if 0
                        dst = vf_alloc_desc_data(s->video_prop);
                        dst->frames = src->frames;

                        s->lut_func((int *)s->lut, dst->tiles[0].data,
                                        src->tiles[0].data, src->tiles[0].data_len);
                        vf_free_data(src);
#endif
                } else {
                        dst = src;
                }

                pthread_mutex_lock(&s->lock);

                s->processed_queue.push(dst);
                if(s->reader_waiting)
                        pthread_cond_signal(&s->reader_cv);
                pthread_cond_signal(&s->boss_cv);
                pthread_mutex_unlock(&s->lock);
        }
after_while:

        return NULL;
}

struct video_frame *
vidcap_vf_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_vf_state 	*s = (struct vidcap_vf_state *) state;

        pthread_mutex_lock(&s->lock);
        while((s->should_pause || s->should_jump) && !s->play_to_buffer) {
                s->grab_waiting = TRUE;
                pthread_cond_wait(&s->pause_cv, &s->lock);
                s->grab_waiting = FALSE;
        }
        if(s->play_to_buffer > 0)
                s->play_to_buffer--;

        if(s->finished && s->processed_queue.empty() &&
                        s->read_queue.empty()) {
                if(s->loop) {
                        abort();
#if 0
                        if(s->speed > 0.0) {
                                s->index =  0;
                                s->frame->frames = - SIGN(s->speed);
                        } else {
                                s->index = s->glob.gl_pathc - 1;
                                s->frame->frames = s->index; - SIGN(s->speed);
                        }

                        s->finished = FALSE;
                        pthread_cond_signal(&s->reader_cv);
#endif
                } else  {
                        pthread_mutex_unlock(&s->lock);
                        return NULL;
                }
        }

        while(s->processed_queue.empty()) {
                pthread_cond_wait(&s->boss_cv, &s->lock);
        }

        struct video_frame *frame = s->processed_queue.front();
        s->processed_queue.pop();

        if(s->processing_waiting)
                        pthread_cond_signal(&s->processing_cv);
        pthread_mutex_unlock(&s->lock);

        if(s->prev_time.tv_sec == 0 && s->prev_time.tv_usec == 0) { /* first run */
                gettimeofday(&s->prev_time, NULL);
        }

        // if we play regurally, we need to wait for its time
        if(s->play_to_buffer == 0) {
                gettimeofday(&s->cur_time, NULL);
                if(s->video_prop.fps == 0) /* it would make following loop infinite */
                        return NULL;
                while(tv_diff_usec(s->cur_time, s->prev_time) < 1000000.0 / s->video_prop.fps / 
                               s->speedup / (fabs(s->speed) < 1.0 ? fabs(s->speed) : 1.0)) {
                        gettimeofday(&s->cur_time, NULL);
                }
                s->prev_time = s->cur_time;
                //tv_add_usec(&s->prev_time, 1000000.0 / s->frame->fps);
        } // else we do not want to wait for next frame time

        s->frames++;

#if 0
        if( s->frame->frames >= (int) s->glob.gl_pathc) {
                s->frame->frames = s->glob.gl_pathc - 1;
        }
        if( s->frame->frames < 0) {
                s->frame->frames = 0;
        }
#endif

        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);
        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            fprintf(stderr, "%d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }

        *audio = NULL;

	return frame;
}

static void flush_pipeline(struct vidcap_vf_state *s)
{
        s->should_jump = TRUE;
        pthread_mutex_unlock(&s->lock);
        while(!s->reader_waiting || !s->processing_waiting || !s->grab_waiting)
                ;

        pthread_mutex_lock(&s->lock);
        s->finished = FALSE;

        while(!s->read_queue.empty()) {
                struct video_frame *frame = s->read_queue.front();
                vf_free_data(frame);
                s->read_queue.pop();
        }
        while(!s->processed_queue.empty()) {
                struct video_frame *frame = s->processed_queue.front();
                vf_free_data(frame);
                s->processed_queue.pop();
        }
}

static void play_after_flush(struct vidcap_vf_state *s)
{
        pthread_cond_signal(&s->reader_cv);
        pthread_cond_signal(&s->processing_cv);
        if(!s->should_pause)
                pthread_cond_signal(&s->pause_cv);

        s->should_jump = FALSE;
}

static void clamp_indices(struct vidcap_vf_state *s)
{
        if(s->seq_num < 0) {
                s->seq_num = 0;
        } else if(s->seq_num >= (int) s->glob.gl_pathc) {
                s->seq_num = s->glob.gl_pathc - 1;
        }
}


void vidcap_vf_command(void *state, int command, void *data)
{
	struct vidcap_vf_state 	*s = (struct vidcap_vf_state *) state;

        pthread_mutex_lock(&s->lock);

        if(command == VIDCAP_PAUSE) {
                fprintf(stderr, "[vf] PAUSE\n");
                s->should_pause = TRUE;
        } else if(command == VIDCAP_PLAY) {
                fprintf(stderr, "[vf] PLAY\n");
                clamp_indices(s);
                s->should_pause = FALSE;
                pthread_cond_signal(&s->pause_cv);
        } else if(command == VIDCAP_PLAYONE) {
                fprintf(stderr, "[vf] PLAYONE\n");
                s->play_to_buffer = *(int *) data;
                pthread_cond_signal(&s->pause_cv);
        } else if(command == VIDCAP_FPS) {
                fprintf(stderr, "[vf] FPS\n");
                s->video_prop.fps = *(float *) data;
        } else if(command == VIDCAP_POS) {
                flush_pipeline(s);
                s->seq_num = *(int *) data;
                clamp_indices(s);
                fprintf(stderr, "[vf] New position: %d\n", s->seq_num);
                play_after_flush(s);
        } else if(command == VIDCAP_LOOP) {
                fprintf(stderr, "[vf] LOOP %d\n", *(int *) data);
                s->loop = *(int *) data;
        } else if(command == VIDCAP_SPEED) {
                fprintf(stderr, "[vf] SPEEDUP %f\n", *(float *) data);
                s->speedup = *(float *) data;
#if 0
                fprintf(stderr, "[vf] SPEED %f\n", *(float *) data);
                pthread_mutex_lock(&s->lock);
                flush_pipeline(s);
                clamp_indices(s);
                //s->frame->frames  = s->index - ROUND_FROM_ZERO(s->speed);
                //s->index = s->frame->frames + SIGN(s->speed);
                s->speed = *(float *) data;
                play_after_flush(s);
#endif
        }

        pthread_mutex_unlock(&s->lock);
}

