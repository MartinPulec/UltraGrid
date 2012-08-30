/*
 * FILE:    dpx.c
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

#include "video_capture/dpx.h"
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

#define BUFFER_LEN 10


struct vidcap_dpx_state {
        struct video_frame *frame;
        struct tile        *tile;
        pthread_mutex_t lock;
        
        pthread_cond_t reader_cv;
        pthread_cond_t processing_cv;
        volatile int reader_waiting;
        volatile int processing_waiting;
        
        unsigned int        loop:1;
        volatile unsigned int        finished:1;
        volatile unsigned int        should_exit_thread:1;
        
        char               *buffer_read[BUFFER_LEN];
        volatile int        buffer_read_start, buffer_read_end;
        char               *buffer_processed[BUFFER_LEN];
        volatile int        buffer_processed_start, buffer_processed_end;
        
        char               *buffer_send;
        
        pthread_t           reading_thread, processing_thread;
        int                 frames;
        struct timeval      t, t0;
        
        glob_t              glob;
        int                 index;
        
        int                *lut;
        float               gamma;
        
        struct timeval      prev_time, cur_time;
        
        unsigned            big_endian:1;
        
        unsigned            dxt5_ycocg:1;
};


static void * reading_thread(void *args);
static void * processing_thread(void *args);
static void usage(void);
static void create_lut(struct vidcap_dpx_state *s);
static void apply_lut_8b(int *lut, char *out, char *in, int size);
static void apply_lut_10b(int *lut, char *out, char *in, int size);
static void apply_lut_10b_be(int *lut, char *out, char *in, int size);
static uint32_t to_native_order(struct vidcap_dpx_state *s, uint32_t num);

static void usage()
{
        printf("DPX video capture usage:\n");
        printf("\t-t dpx:files=<glob>[:fps=<fps>:gamma=<gamma>:loop]\n");
}

static uint32_t to_native_order(struct vidcap_dpx_state *s, uint32_t num)
{
        if (s->big_endian)
                return ntohl(num);
        else
                return num;
}

static void create_lut(struct vidcap_dpx_state *s)
{
}

static void apply_lut_10b(int *lut, char *out_data, char *in_data, int size)
{
        int x;
        int elems = size / 4;
        register unsigned int *in = in_data;
        register unsigned int *out = out_data;
        register int r,g,b;
        
        for(x = 0; x < elems; ++x) {
                register unsigned int val = *in++;
                r = lut[val >> 22];
                g = lut[(val >> 12) & 0x3ff];
                b = lut[(val >> 2) & 0x3ff];
                *out++ = r << 22 | g << 12 | b << 2;
        }
}

static void apply_lut_10b_be(int *lut, char *out_data, char *in_data, int size)
{
        int x;
        int elems = size / 4;
        register unsigned int *in = in_data;
        register unsigned int *out = out_data;
        register int r,g,b;
        
        for(x = 0; x < elems; ++x) {
                register unsigned int val = htonl(*in++);
                r = lut[val >> 22];
                g = lut[(val >> 12) & 0x3ff];
                b = lut[(val >> 2) & 0x3ff];
                *out++ = r << 22 | g << 12 | b << 2;
        }
}


static void apply_lut_8b(int *lut, char *out_data, char *in_data, int size)
{
        int x;
        int elems = size / 4;
        register unsigned int *in = in_data;
        register unsigned int *out = out_data;
        register int r,g,b;
        
        for(x = 0; x < elems; ++x) {
                register unsigned int val = *in++;
                r = lut[(val >> 16) & 0xff];
                g = lut[(val >> 8) & 0xff];
                b = lut[(val >> 0) & 0xff];
                *out++ = r << 16 | g << 8 | b << 0;
        }
}

struct vidcap_type *
vidcap_jpeg_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_DPX_ID;
		vt->name        = "jpeg";
		vt->description = "Digital Picture Exchange file";
	}
	return vt;
}

void *
vidcap_jpeg_init(char *fmt, unsigned int flags)
{
	struct vidcap_dpx_state *s;
        char *item;
        char *glob_pattern;
        char *save_ptr = NULL;
        int i;
        int width, height;

	printf("vidcap_dpx_init\n");

        s = (struct vidcap_dpx_state *) calloc(1, sizeof(struct vidcap_dpx_state));
        
        if(!fmt || strcmp(fmt, "help") == 0) {
                usage();
                return NULL;
        }
        
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->processing_cv, NULL);
        pthread_cond_init(&s->reader_cv, NULL);
        s->processing_waiting = FALSE;
        s->reader_waiting = FALSE;
        
        s->buffer_processed_start = s->buffer_processed_end = 0;
        s->buffer_read_start = s->buffer_read_end = 0;
        
        s->should_exit_thread = FALSE;
        s->finished = FALSE;
        
        s->frame = vf_alloc(1);
        s->frame->fps = 30.0;
        s->gamma = 1.0;
        s->loop = FALSE;
        
        item = strtok_r(fmt, ":", &save_ptr);
        while(item) {
                if(strncmp("files=", item, strlen("files=")) == 0) {
                        glob_pattern = item + strlen("files=");
                } else if(strncmp("fps=", item, strlen("fps=")) == 0) {
                        s->frame->fps = atof(item + strlen("fps="));
                } else if(strncmp("gamma=", item, strlen("gamma=")) == 0) {
                        s->gamma = atof(item + strlen("gamma="));
                } else if(strncmp("loop", item, strlen("loop")) == 0) {
                        s->loop = TRUE;
                } else if(strncmp("width=", item, strlen("width=")) == 0) {
                        s->gamma = atof(item + strlen("width="));
                } else if(strncmp("height=", item, strlen("height=")) == 0) {
                        s->gamma = atof(item + strlen("height="));
                }
                
                item = strtok_r(NULL, ":", &save_ptr);
        }
        
        int ret = glob(glob_pattern, 0, NULL, &s->glob);
        if (ret)
        {
                perror("Opening DPX files failed");
                return NULL;
        }
        
        char *filename = s->glob.gl_pathv[0];

        struct stat buf;
        assert(stat(filename, &buf) == 0);
        
        s->dxt5_ycocg = TRUE;
        
                s->dxt5_ycocg = TRUE;
                s->big_endian = FALSE;
        
                s->frame->color_spec = JPEG;
        
        s->frame->interlacing = PROGRESSIVE;
        s->tile = vf_get_tile(s->frame, 0);
        s->tile->width = width;
        s->tile->height = height;
        
        s->tile->data_len = buf.st_size;
        
        for (i = 0; i < BUFFER_LEN; ++i) {
                s->buffer_read[i] = malloc(s->tile->data_len);
                s->buffer_processed[i] = malloc(s->tile->data_len);
        }
        s->buffer_send = malloc(s->tile->data_len);
        
        pthread_create(&s->reading_thread, NULL, reading_thread, s);
        pthread_create(&s->processing_thread, NULL, processing_thread, s);
        
        s->prev_time.tv_sec = s->prev_time.tv_usec = 0;

	return s;
}

void
vidcap_jpeg_finish(void *state)
{
        UNUSED(state);
}

void
vidcap_jpeg_done(void *state)
{
	struct vidcap_dpx_state *s = (struct vidcap_dpx_state *) state;
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
        
        vf_free(s->frame);
        for (i = 0; i < BUFFER_LEN; ++i) {
                free(s->buffer_read[i]);
                free(s->buffer_processed[i]);
        }
        free(s->buffer_send);
        free(s);
        fprintf(stderr, "DPX exited\n");
}

static void * reading_thread(void *args)
{
	struct vidcap_dpx_state 	*s = (struct vidcap_dpx_state *) args;

        while(1) {
                pthread_mutex_lock(&s->lock);
                if(s->should_exit_thread) {
                        pthread_mutex_unlock(&s->lock);
                        goto after_while;
                }
                while(s->buffer_read_start == ((s->buffer_read_end + 1) % BUFFER_LEN)) { /* full */
                        s->reader_waiting = TRUE;
                        pthread_cond_wait(&s->reader_cv, &s->lock);
                        s->reader_waiting = FALSE;
                        if(s->should_exit_thread) {
                                pthread_mutex_unlock(&s->lock);
                                goto after_while;
                        }
                }
                
                pthread_mutex_unlock(&s->lock);
                                        
                char *filename = s->glob.gl_pathv[s->index++];
                int fd = open(filename, O_RDONLY);
                ssize_t bytes_read = 0;
                unsigned int file_offset = 0;
                do {
                        bytes_read += pread(fd, s->buffer_read[s->buffer_read_end] + bytes_read,
                                        s->tile->data_len - bytes_read,
                                        file_offset + bytes_read);
                } while(bytes_read < s->tile->data_len);
                
                close(fd);
                
                pthread_mutex_lock(&s->lock);
                s->buffer_read_end = (s->buffer_read_end + 1) % BUFFER_LEN; /* and we will read next one */
                if(s->processing_waiting)
                        pthread_cond_signal(&s->processing_cv);
                pthread_mutex_unlock(&s->lock);
                
                if( s->index == s->glob.gl_pathc) {
                        if(s->loop) {
                                s->index = 0;
                        } else {
                                s->finished = TRUE;
                                goto after_while;
                        }
                }
        }
after_while:
        
        while(!s->should_exit_thread)
                ;

        return NULL;
}

static void * processing_thread(void *args)
{
	struct vidcap_dpx_state 	*s = (struct vidcap_dpx_state *) args;

        while(1) {
                pthread_mutex_lock(&s->lock);
                if(s->should_exit_thread) {
                        pthread_mutex_unlock(&s->lock);
                        break;
                }
                while(s->buffer_processed_start == ((s->buffer_processed_end + 1) % BUFFER_LEN)) { /* full */
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
                while(s->buffer_read_start == s->buffer_read_end) { /* empty */
                        s->processing_waiting = TRUE;
                        pthread_cond_wait(&s->processing_cv, &s->lock);
                        s->processing_waiting = FALSE;
                        if(s->should_exit_thread) {
                                pthread_mutex_unlock(&s->lock);
                                goto after_while;
                        }
                }
                
                pthread_mutex_unlock(&s->lock);
                
                        memcpy(s->buffer_processed[s->buffer_processed_end],
                                        s->buffer_read[s->buffer_read_start], s->tile->data_len);

                
                pthread_mutex_lock(&s->lock);
                s->buffer_read_start = (s->buffer_read_start + 1) % BUFFER_LEN; /* and we will read next one */
                s->buffer_processed_end = (s->buffer_processed_end + 1) % BUFFER_LEN; /* and we will read next one */
                if(s->reader_waiting)
                        pthread_cond_signal(&s->reader_cv);
                pthread_mutex_unlock(&s->lock);
        }
after_while:

        return NULL;
}

struct video_frame *
vidcap_jpeg_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_dpx_state 	*s = (struct vidcap_dpx_state *) state;
        
        if(s->finished && s->buffer_processed_start == s->buffer_processed_end &&
                        s->buffer_read_start == s->buffer_read_end) {
                exit_uv(0);
                return NULL;
        }
        
        while(s->buffer_processed_start == s->buffer_processed_end && !should_exit && !s->finished)
                ;

        if(s->prev_time.tv_sec == 0 && s->prev_time.tv_usec == 0) { /* first run */
                gettimeofday(&s->prev_time, NULL);
        }
        gettimeofday(&s->cur_time, NULL);
        while(tv_diff_usec(s->cur_time, s->prev_time) < 1000000.0 / s->frame->fps)
                gettimeofday(&s->cur_time, NULL);
        tv_add_usec(&s->prev_time, 1000000.0 / s->frame->fps);
        
        s->tile->data = s->buffer_processed[s->buffer_processed_start];
        s->buffer_processed[s->buffer_processed_start] = s->buffer_send;
        s->buffer_send = s->tile->data;
        
        pthread_mutex_lock(&s->lock);
        s->buffer_processed_start = (s->buffer_processed_start + 1) % BUFFER_LEN;
        if(s->processing_waiting)
                        pthread_cond_signal(&s->processing_cv);
        pthread_mutex_unlock(&s->lock);

        s->frames++;
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

