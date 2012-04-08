/*
 * FILE:    tiff.c
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

#include "video_capture/tiff.h"
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

#include <tiffio.h>

#define BUFFER_LEN 10

#define SIGN(x) (x / fabs(x))
#define ROUND_FROM_ZERO(x) (ceil(fabs(x)) * SIGN(x))


typedef void (*lut_func_t)(double *lut, char *out_data, char *in_data, int size);

struct vidcap_tiff_state {
        struct video_frame *frame;
        struct tile        *tile;

        pthread_mutex_t lock;

        pthread_cond_t reader_cv;
        volatile int reader_waiting;

        volatile int should_pause;
        pthread_cond_t pause_cv;

        unsigned int        loop:1;
        volatile unsigned int        finished:1;
        volatile unsigned int        should_exit_thread:1;

        char               *buffer_read[BUFFER_LEN];
        char               *tiff_buffer;
        volatile int        buffer_read_start, buffer_read_end;

        char               *buffer_send;

        pthread_t           reading_thread;
        int                 frames;
        struct timeval      t, t0;

        glob_t              glob;
        int                 index;

        float               gamma;

        struct timeval      prev_time, cur_time;

        unsigned int        should_jump:1;
        unsigned int        grab_waiting:1;
        unsigned int        playone;

        int                 tiff_color_depth;

        double              speed;
};


static void * reading_thread(void *args);
static void usage(void);


static void apply_lut_8b(double *lut, char *out, char *in, int size);
static void apply_lut_16b(double *lut, char *out, char *in, int size);

static inline void strip16_to_8(char * dest, const char * stripData, int stripLen);



static void usage()
{
        printf("TIFF video capture usage:\n");
        printf("\t-t tiff:files=<glob>[:fps=<fps>:gamma=<gamma>:loop]\n");
}

struct vidcap_type *
vidcap_tiff_probe(void)
{
	struct vidcap_type*		vt;

	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_TIFF_ID;
		vt->name        = "tiff";
		vt->description = "Tag Image File Format";
	}
	return vt;
}

void *
vidcap_tiff_init(char *fmt, unsigned int flags)
{
        UNUSED(flags);
	struct vidcap_tiff_state *s;
        char *item;
        char *glob_pattern;
        char *save_ptr = NULL;
        int i;

	printf("vidcap_tiff_init\n");

        s = (struct vidcap_tiff_state *) calloc(1, sizeof(struct vidcap_tiff_state));

        if(!fmt || strcmp(fmt, "help") == 0) {
                usage();
                return NULL;
        }

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->reader_cv, NULL);
        pthread_cond_init(&s->pause_cv, NULL);
        s->reader_waiting = FALSE;
        s->should_pause = TRUE;
        s->should_jump = FALSE;
        s->grab_waiting = FALSE;

        s->buffer_read_start = s->buffer_read_end = 0;
        s->index = 0;

        s->should_exit_thread = FALSE;
        s->finished = FALSE;

        s->frame = vf_alloc(1);
        s->frame->fps = 30.0;
        s->frame->frames = -1;
        s->gamma = 1.0;
        s->speed = 1.0;
        s->loop = FALSE;
        s->playone = 0;

        s->frame->luts_to_apply = 0;

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
                } else if(strncmp("colorspace=", item, strlen("colorspace=")) == 0) {
                        if (strcasecmp(item + strlen("colorspace="), "XYZ") == 0) {
                                s->frame->luts_to_apply = (struct lut_list*) malloc(sizeof(struct lut_list));
                                s->frame->luts_to_apply->next = NULL;
                                s->frame->luts_to_apply->type = LUT_3D_MATRIX;
                                s->frame->luts_to_apply->lut = xyz_to_rgb_709_d65;
                        } else if(strcasecmp(item + strlen("colorspace="), "RGB_709_D65") != 0) {
                                fprintf(stderr, "WARNING!!!!! Unsupported color space: %s", item + strlen("colorspace="));
                        }
                }

                item = strtok_r(NULL, ":", &save_ptr);
        }

        int ret = glob(glob_pattern, 0, NULL, &s->glob);
        if (ret)
        {
                perror("Opening TIFF files failed");
                return NULL;
        }

        char *filename = s->glob.gl_pathv[0];
        TIFF *tif = TIFFOpen(filename, "r");
        if(!tif) {
                fprintf(stderr, "[TIFF] Failed to open file \"%s\"\n", filename);
                perror("");
                free(s);
                return NULL;
        }

        s->frame->interlacing = PROGRESSIVE;
        s->tile = vf_get_tile(s->frame, 0);

        unsigned int uval;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &uval);
        s->tile->width = uval;
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &uval);
        s->tile->height = uval;
        uint16_t usval;
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &usval);
        printf("[TIFF] Detected image size: %d %d\n", s->tile->width, s->tile->height);

        s->tile->data_len = s->tile->width * s->tile->height;


        s->tiff_color_depth = usval;

        switch(s->tiff_color_depth) {
                case 8:
                        s->tile->data_len *= 3;
                        s->frame->color_spec = RGB;
                        break;
                case 10:
                        s->tile->data_len *= 4;
                        s->frame->color_spec = DPX10;
                        fprintf(stderr, "10-bits are not yet fully supported.\n");
                        abort();
                        break;
                case 16:
                        s->tile->data_len *= 6;
                        s->frame->color_spec = RGB16;
                        s->tiff_buffer = malloc(s->tile->width * s->tile->height * 6 * 16);
                        break;

                default:
                        fprintf(stderr, "[TIFF] Unsupported color depth: %d\n", s->tiff_color_depth);
                        abort();
                        break;
        }

        for (i = 0; i < BUFFER_LEN; ++i) {
                s->buffer_read[i] = malloc(s->tile->data_len);
        }
        s->buffer_send = malloc(s->tile->data_len);

        TIFFClose(tif);

        pthread_create(&s->reading_thread, NULL, reading_thread, s);

        s->prev_time.tv_sec = s->prev_time.tv_usec = 0;

	return s;
}

void
vidcap_tiff_finish(void *state)
{
	struct vidcap_tiff_state *s = (struct vidcap_tiff_state *) state;
        pthread_mutex_lock(&s->lock);
        s->should_pause = FALSE;
        pthread_cond_signal(&s->pause_cv);

        s->should_exit_thread = TRUE;
        s->finished = TRUE;
        pthread_mutex_unlock(&s->lock);
}

void
vidcap_tiff_done(void *state)
{
        return;
	struct vidcap_tiff_state *s = (struct vidcap_tiff_state *) state;
        int i;
	assert(s != NULL);

        pthread_mutex_lock(&s->lock);
        s->should_exit_thread = TRUE;
        if(s->reader_waiting)
                pthread_cond_signal(&s->reader_cv);
        pthread_mutex_unlock(&s->lock);

	pthread_join(s->reading_thread, NULL);

        vf_free(s->frame);
        for (i = 0; i < BUFFER_LEN; ++i) {
                free(s->buffer_read[i]);
        }
        free(s->buffer_send);
        free(s);
        fprintf(stderr, "TIFF exited\n");
}

static inline void strip16_to_8(char * dest, const char * stripData, int stripLen)
{
        int x;

        register uint32_t part1, part2;
        register uint16_t r, g, b;
        register uint32_t out_data;
        register uint16_t *in;
        register uint8_t *out;

        in = (uint32_t *) stripData;
        out = (uint32_t *) dest;


        for(x = 0; x < stripLen; x += 6) {
                r = *in++;
                g = *in++;
                b = *in++;


                *out++ = r >> 8;
                *out++ = g >> 8;
                *out++ = b >> 8;
        }
}

static void apply_lut_8b(double *lut, char *out, char *in, int size)
{
        // TODO
        fprintf(stderr, "%s:%d: Unimplemented", __FILE__, __LINE__);
        abort();
}

static void apply_lut_16b(double *lut, char *out_data, char *in_data, int size)
{
        uint16_t x1, x2, x3;
        uint16_t y1, y2, y3;
        int i;

        uint16_t *in = (uint16_t *) in_data;
        uint16_t *out = (uint16_t *) out_data;


        for (i = 0; i < size; i+= 6) {
                x1 = *in++;
                x2 = *in++;
                x3 = *in++;

                y1 = lut[0] * x1 + lut[1] * x2 + lut[2] * x3;
                y2 = lut[3] * x1 + lut[4] * x2 + lut[5] * x3;
                y3 = lut[6] * x1 + lut[7] * x2 + lut[8] * x3;

                *out++ = y1;
                *out++ = y2;
                *out++ = y3;
        }
}

static void * reading_thread(void *args)
{
	struct vidcap_tiff_state        *s = (struct vidcap_tiff_state *) args;

        while(1) {
                pthread_mutex_lock(&s->lock);
                if(s->should_exit_thread) {
                        pthread_mutex_unlock(&s->lock);
                        goto after_while;
                }
                while(s->finished || s->should_jump || s->buffer_read_start == ((s->buffer_read_end + 1) % BUFFER_LEN)) { /* full */
                        s->reader_waiting = TRUE;
                        pthread_cond_wait(&s->reader_cv, &s->lock);
                        s->reader_waiting = FALSE;
                        if(s->should_exit_thread) {
                                pthread_mutex_unlock(&s->lock);
                                goto after_while;
                        }
                }

                pthread_mutex_unlock(&s->lock);


                char *filename = s->glob.gl_pathv[s->index];
                TIFF *tif = TIFFOpen(filename, "r");

                s->index += ROUND_FROM_ZERO(s->speed);

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


                tdata_t dest = s->buffer_read[s->buffer_read_end];
                tstrip_t const numberOfStrips = TIFFNumberOfStrips(tif);
                unsigned int rowsPerStrip;
                TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);

                unsigned long int const offset = vc_get_linesize(s->tile->width, s->frame->color_spec) * rowsPerStrip;
                tstrip_t strip;
                for(strip = 0u; strip < numberOfStrips; ++strip) {
                        if(TIFFReadEncodedStrip(tif, strip, dest, (tsize_t) - 1) == (tsize_t) -1)
                                fprintf(stderr, "Failed to read a strip %.", filename);
                        dest += offset;
                }

                /* TODO: figure out postprocessing of 10 b images */

                // "high level" API
                /*if(!TIFFReadRGBAImage(tif, s->tile->width, s->tile->height, dest, 0)) {
                        fprintf(stderr, "Failed to read a strip %.", filename);
                }*/
                TIFFClose(tif);


                pthread_mutex_lock(&s->lock);
                s->buffer_read_end = (s->buffer_read_end + 1) % BUFFER_LEN; /* and we will read next one */
                /*if(s->processing_waiting)
                        pthread_cond_signal(&s->processing_cv);*/
                pthread_mutex_unlock(&s->lock);

                if((s->speed > 0.0 && s->index >= (int) s->glob.gl_pathc) ||
                                s->index < 0) {
                        s->finished = TRUE;
                }
        }
after_while:

        while(!s->should_exit_thread)
                ;

        return NULL;
}

struct video_frame *
vidcap_tiff_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_tiff_state 	*s = (struct vidcap_tiff_state *) state;

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
                        if(s->speed > 0.0) {
                                s->index = s->frame->frames = 0;
                        } else {
                                s->index = s->frame->frames = s->glob.gl_pathc - 1;
                        }

                        s->finished = FALSE;
                        pthread_cond_signal(&s->reader_cv);
                } else  {
                        pthread_mutex_unlock(&s->lock);
                        return NULL;
                }
        }

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
                        pthread_cond_signal(&s->reader_cv);
        pthread_mutex_unlock(&s->lock);

        s->frames++;

        s->frame->frames += ROUND_FROM_ZERO(s->speed);

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


static void flush_pipeline(struct vidcap_tiff_state *s)
{
                s->should_jump = TRUE;
                pthread_mutex_unlock(&s->lock);
                while(!s->reader_waiting || !s->grab_waiting)
                        ;

                pthread_mutex_lock(&s->lock);
                s->finished = FALSE;

}

static void play_after_flush(struct vidcap_tiff_state *s)
{
                pthread_cond_signal(&s->reader_cv);
                if(!s->should_pause)
                        pthread_cond_signal(&s->pause_cv);

                s->should_jump = FALSE;
}

static void clamp_indices(struct vidcap_tiff_state *s)
{
        if(s->index < 0) {
                s->index = 0;
        } else if(s->index >= (int) s->glob.gl_pathc) {
                s->index = s->glob.gl_pathc - 1;
        }
}


void vidcap_tiff_command(struct vidcap *state, int command, void *data)
{
	struct vidcap_tiff_state 	*s = (struct vidcap_tiff_state *) state;


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

                flush_pipeline(s);

                s->buffer_read_start = s->buffer_read_end = 0;
                s->index = *(int *) data;
                clamp_indices(s);
                fprintf(stderr, "New position: %d\n", s->index);
                s->frame->frames = s->index - 1;

                play_after_flush(s);

                pthread_mutex_unlock(&s->lock);
        } else if(command == VIDCAP_LOOP) {
                pthread_mutex_lock(&s->lock);
                s->loop = *(int *) data;
                pthread_mutex_unlock(&s->lock);
        } else if(command == VIDCAP_SPEED) {
                fprintf(stderr, "[TIFF] SPEED %f\n", *(float *) data);
                pthread_mutex_lock(&s->lock);
                flush_pipeline(s);

                //s->frame->frames  = s->index - ROUND_FROM_ZERO(s->speed);
                s->index = s->frame->frames + ROUND_FROM_ZERO(s->speed);
                clamp_indices(s);
                s->speed = *(float *) data;

                play_after_flush(s);
                pthread_mutex_unlock(&s->lock);
        }
}

