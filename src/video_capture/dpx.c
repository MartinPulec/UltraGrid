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

typedef struct file_information
{
    uint32_t   magic_num;        /* magic number 0x53445058 (SDPX) or 0x58504453 (XPDS) */
    uint32_t   offset;           /* offset to image data in bytes */
    char vers[8];          /* which header format version is being used (v1.0)*/
    uint32_t   file_size;        /* file size in bytes */
    uint32_t   ditto_key;        /* read time short cut - 0 = same, 1 = new */
    uint32_t   gen_hdr_size;     /* generic header length in bytes */
    uint32_t   ind_hdr_size;     /* industry header length in bytes */
    uint32_t   user_data_size;   /* user-defined data length in bytes */
    char file_name[100];   /* iamge file name */
    char create_time[24];  /* file creation date "yyyy:mm:dd:hh:mm:ss:LTZ" */
    char creator[100];     /* file creator's name */
    char project[200];     /* project name */
    char copyright[200];   /* right to use or copyright info */
    uint32_t   key;              /* encryption ( FFFFFFFF = unencrypted ) */
    char Reserved[104];    /* reserved field TBD (need to pad) */
} FileInformation;

typedef struct _image_information
{
    uint16_t    orientation;          /* image orientation */
    uint16_t    element_number;       /* number of image elements */
    uint32_t   pixels_per_line;      /* or x value */
    uint32_t    lines_per_image_ele;  /* or y value, per element */
    struct _image_element
    {
        uint32_t    data_sign;        /* data sign (0 = unsigned, 1 = signed ) */
				 /* "Core set images are unsigned" */
        uint32_t    ref_low_data;     /* reference low data code value */
        float    ref_low_quantity; /* reference low quantity represented */
        uint32_t    ref_high_data;    /* reference high data code value */
        float    ref_high_quantity;/* reference high quantity represented */
        uint8_t     descriptor;       /* descriptor for image element */
        uint8_t     transfer;         /* transfer characteristics for element */
        uint8_t     colorimetric;     /* colormetric specification for element */
        uint8_t     bit_size;         /* bit size for element */
	uint16_t    packing;          /* packing for element */
        uint16_t    encoding;         /* encoding for element */
        uint32_t    data_offset;      /* offset to data of element */
        uint32_t    eol_padding;      /* end of line padding used in element */
        uint32_t    eo_image_padding; /* end of image padding used in element */
        char  description[32];  /* description of element */
    } image_element[8];          /* NOTE THERE ARE EIGHT OF THESE */

    uint8_t reserved[52];             /* reserved for future use (padding) */
} Image_Information;

typedef struct _image_orientation
{
    uint32_t   x_offset;               /* X offset */
    uint32_t   y_offset;               /* Y offset */
    float   x_center;               /* X center */
    float   y_center;               /* Y center */
    uint32_t   x_orig_size;            /* X original size */
    uint32_t   y_orig_size;            /* Y original size */
    char file_name[100];         /* source image file name */
    char creation_time[24];      /* source image creation date and time */
    char input_dev[32];          /* input device name */
    char input_serial[32];       /* input device serial number */
    uint16_t   border[4];              /* border validity (XL, XR, YT, YB) */
    uint32_t   pixel_aspect[2];        /* pixel aspect ratio (H:V) */
    uint8_t    reserved[28];           /* reserved for future use (padding) */
} Image_Orientation;

typedef struct _motion_picture_film_header
{
    char film_mfg_id[2];    /* film manufacturer ID code (2 digits from film edge code) */
    char film_type[2];      /* file type (2 digits from film edge code) */
    char offset[2];         /* offset in perfs (2 digits from film edge code)*/
    char prefix[6];         /* prefix (6 digits from film edge code) */
    char count[4];          /* count (4 digits from film edge code)*/
    char format[32];        /* format (i.e. academy) */
    uint32_t   frame_position;    /* frame position in sequence */
    uint32_t   sequence_len;      /* sequence length in frames */
    uint32_t   held_count;        /* held count (1 = default) */
    float   frame_rate;        /* frame rate of original in frames/sec */
    float   shutter_angle;     /* shutter angle of camera in degrees */
    char frame_id[32];      /* frame identification (i.e. keyframe) */
    char slate_info[100];   /* slate information */
    uint8_t    reserved[56];      /* reserved for future use (padding) */
} Motion_Picture_Film;

typedef struct _television_header
{
    uint32_t tim_code;            /* SMPTE time code */
    uint32_t userBits;            /* SMPTE user bits */
    uint8_t  interlace;           /* interlace ( 0 = noninterlaced, 1 = 2:1 interlace*/
    uint8_t  field_num;           /* field number */
    uint8_t  video_signal;        /* video signal standard (table 4)*/
    uint8_t  unused;              /* used for byte alignment only */
    float hor_sample_rate;     /* horizontal sampling rate in Hz */
    float ver_sample_rate;     /* vertical sampling rate in Hz */
    float frame_rate;          /* temporal sampling rate or frame rate in Hz */
    float time_offset;         /* time offset from sync to first pixel */
    float gamma;               /* gamma value */
    float black_level;         /* black level code value */
    float black_gain;          /* black gain */
    float break_point;         /* breakpoint */
    float white_level;         /* reference white level code value */
    float integration_times;   /* integration time(s) */
    uint8_t  reserved[76];        /* reserved for future use (padding) */
} Television_Header;

typedef void (*lut_func_t)(int *lut, char *out_data, char *in_data, int size);

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
        lut_func_t          lut_func;
        float               gamma;
        
        struct timeval      prev_time, cur_time;
        
        unsigned            big_endian:1;
        
        struct file_information file_information;
        struct _image_information image_information;
        struct _image_orientation image_orientation;
        struct _motion_picture_film_header motion_header;
        struct _television_header television_header;

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
        int x;
        int max_val;
        float gamma;
        
        max_val = 1<<s->image_information.image_element[0].bit_size;
        //if(s->image_information.image_element[0].transfer == 0)
        
        free(s->lut);
        s->lut = (unsigned int *) malloc(sizeof(unsigned int) *
                        max_val);
        
        for (x = 0;
                x < max_val;
                ++x)
        {
                s->lut[x] = pow((float) x / (max_val - 1), s->gamma) *
                                (max_val - 1);
        }
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
vidcap_dpx_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_DPX_ID;
		vt->name        = "dpx";
		vt->description = "Digital Picture Exchange file";
	}
	return vt;
}

void *
vidcap_dpx_init(char *fmt, unsigned int flags)
{
	struct vidcap_dpx_state *s;
        char *item;
        char *glob_pattern;
        char *save_ptr = NULL;
        int i;

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
        int fd = open(filename, O_RDONLY);
        if(fd == -1) {
                fprintf(stderr, "[DPX] Failed to open file \"%s\"\n", filename);
                perror("");
                free(s);
                return NULL;
        }
        
        read(fd, &s->file_information, sizeof(s->file_information));
        read(fd, &s->image_information, sizeof(s->image_information));
        read(fd, &s->image_orientation, sizeof(s->image_orientation));
        read(fd, &s->motion_header, sizeof(s->motion_header));
        read(fd, &s->television_header, sizeof(s->television_header));

        s->dxt5_ycocg = FALSE;
        
        if(s->file_information.magic_num == 'XPDS')
                s->big_endian = TRUE;
        else if(s->file_information.magic_num == 'SDPX')
                s->big_endian = FALSE;
        else if(s->file_information.magic_num == 'DXT5') {
                s->dxt5_ycocg = TRUE;
                s->big_endian = FALSE;

        } else {
                fprintf(stderr, "[DPX] corrupted file %s. "
                        "Not recognised as DPX.", filename);
                free(s);
                return NULL;
        }
        
        if(!s->dxt5_ycocg) {
                switch (s->image_information.image_element[0].bit_size)
                {
                        case 8:
                                s->frame->color_spec = RGBA;
                                s->lut_func = apply_lut_8b;
                                break;
                        case 10:
                                s->frame->color_spec = DPX10;
                                if(s->big_endian)
                                        s->lut_func = apply_lut_10b_be;
                                else
                                        s->lut_func = apply_lut_10b;
                                break;
                        default:
                                fprintf(stderr, "[DPX] Currently no support for %d-bit images.", 
                                                s->image_information.image_element[0].bit_size);
                                free(s);
                                return NULL;
                }
                
                create_lut(s);
        } else {
                s->frame->color_spec = DXT5;
                s->lut_func = NULL;
        }
        
        s->frame->interlacing = PROGRESSIVE;
        s->tile = vf_get_tile(s->frame, 0);
        s->tile->width = to_native_order(s, s->image_information.pixels_per_line);
        s->tile->height = to_native_order(s, s->image_information.lines_per_image_ele);
        
        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) * s->tile->height;
        
        for (i = 0; i < BUFFER_LEN; ++i) {
                s->buffer_read[i] = malloc(s->tile->data_len);
                s->buffer_processed[i] = malloc(s->tile->data_len);
        }
        s->buffer_send = malloc(s->tile->data_len);
        
        close(fd);

        pthread_create(&s->reading_thread, NULL, reading_thread, s);
        pthread_create(&s->processing_thread, NULL, processing_thread, s);
        
        s->prev_time.tv_sec = s->prev_time.tv_usec = 0;

	return s;
}

void
vidcap_dpx_finish(void *state)
{
        UNUSED(state);
}

void
vidcap_dpx_done(void *state)
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
                unsigned int file_offset = to_native_order(s, s->file_information.offset);
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
                
                if(s->lut_func) {
                        s->lut_func(s->lut, s->buffer_processed[s->buffer_processed_end],
                                        s->buffer_read[s->buffer_read_start], s->tile->data_len);
                } else {
                        memcpy(s->buffer_processed[s->buffer_processed_end],
                                        s->buffer_read[s->buffer_read_start], s->tile->data_len);

                }
                
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
vidcap_dpx_grab(void *state, struct audio_frame **audio)
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

