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

extern int	should_exit;

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

typedef void (*lut_func_t)(int *lut, char *data, int size);

struct vidcap_dpx_state {
        struct video_frame *frame;
        struct tile        *tile;
        sem_t               have_item;
        
        char               *buffers[2];
        int                 buffer_read;
        
        pthread_t           grabber;
        int                 frames;
        struct timeval      t, t0;
        
        glob_t              glob;
        int                 index;
        
        int                *lut;
        lut_func_t          lut_func;
        
        struct file_information file_information;
        struct _image_information image_information;
        struct _image_orientation image_orientation;
        struct _motion_picture_film_header motion_header;
};


static void * vidcap_grab_thread(void *args);
static void usage(void);
static void create_lut(struct vidcap_dpx_state *s);
static void apply_lut_8b(int *lut, char *data, int size);
static void apply_lut_10b(int *lut, char *data, int size);

static void usage()
{
        printf("DPX video capture usage:\n");
        printf("\t-t dpx:<glob>\n");
}

static void create_lut(struct vidcap_dpx_state *s)
{
        int x;
        int max_val;
        float gamma;
        
        max_val = 1<<s->image_information.image_element[0].bit_size;
        if(s->image_information.image_element[0].transfer == 0)
                gamma = 1.0;
        
        free(s->lut);
        s->lut = (int *) malloc(sizeof(int) *
                        s->image_information.image_element[0].ref_high_data);
        
        for (x = s->image_information.image_element[0].ref_low_data;
                x < s->image_information.image_element[0].ref_high_data;
                ++x)
        {
                s->lut[x] = pow((float) (x - s->image_information.image_element[0].ref_low_data) / 
                                s->image_information.image_element[0].ref_high_data, gamma) *
                                max_val;
        }
}

static void apply_lut_10b(int *lut, char *data, int size)
{
        int x;
        int elems = size / 4;
        register unsigned int *in = data;
        register int r,g,b;
        
        for(x = 0; x < elems; ++x) {
                register unsigned int val = *in;
                r = lut[val >> 22];
                g = lut[(val >> 12) & 0x3ff];
                b = lut[(val >> 2) & 0x3ff];
                *in++ = r << 22 | g << 12 | b << 2;
        }
}

static void apply_lut_8b(int *lut, char *data, int size)
{
        int x;
        int elems = size / 4;
        register unsigned int *in = data;
        register int r,g,b;
        
        for(x = 0; x < elems; ++x) {
                register unsigned int val = *in;
                r = lut[val >> 24];
                g = lut[(val >> 16) & 0xff];
                b = lut[(val >> 8) & 0xff];
                *in++ = r << 24 | g << 16 | b << 8;
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

	printf("vidcap_dpx_init\n");

        s = (struct vidcap_dpx_state *) calloc(1, sizeof(struct vidcap_dpx_state));
        
        if(!fmt || strcmp(fmt, "help") == 0) {
                usage();
                return NULL;
        } 
        
        int ret = glob(fmt, 0, NULL, &s->glob);
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
        
        s->frame = vf_alloc(1, 1);
        switch (s->image_information.image_element[0].bit_size)
        {
                case 8:
                        s->frame->color_spec = RGBA;
                        s->lut_func = apply_lut_8b;
                        break;
                case 10:
                        s->frame->color_spec = DPX10;
                        s->lut_func = apply_lut_10b;
                        break;
                default:
                        error_with_code_msg(128, "[DPX] Currently no support for %d-bit images.", 
                                        s->image_information.image_element[0].bit_size);
        }
        
        create_lut(s);
        
        s->frame->aux = 0;
        s->frame->fps = 25.0;
        s->tile = tile_get(s->frame, 0, 0);
        s->tile->width = s->image_information.pixels_per_line;
        s->tile->height = s->image_information.lines_per_image_ele;
        
        s->tile->data_len = s->tile->width * s->tile->height * 4;
        s->tile->data =
                s->buffers[0] = malloc(s->tile->data_len);
        s->buffers[1] = malloc(s->tile->data_len);
        
        ssize_t bytes_read = 0;
        do {
                bytes_read += pread(fd, s->tile->data + bytes_read,
                                s->tile->data_len - bytes_read,
                                s->file_information.offset + bytes_read);
        } while(bytes_read < s->tile->data_len);
        
        s->lut_func(s->lut, s->tile->data, s->tile->data_len);
        close(fd);
        s->buffer_read = 1;
        s->index = 1;

        sem_init(&s->have_item, 0, 0);
        pthread_create(&s->grabber, NULL, vidcap_grab_thread, s);

	return s;
}

void
vidcap_dpx_done(void *state)
{
	struct vidcap_dpx_state *s = (struct vidcap_dpx_state *) state;

	assert(s != NULL);

        vf_free(s->frame);
	pthread_join(s->grabber, NULL);
	sem_destroy(&s->have_item);
        
        free(s);
}

static void * vidcap_grab_thread(void *args)
{
	struct vidcap_dpx_state 	*s = (struct vidcap_dpx_state *) args;
        struct timeval cur_time;
        struct timeval prev_time;
        
        gettimeofday(&prev_time, NULL);

        while(!should_exit && s->index < s->glob.gl_pathc) {
                gettimeofday(&cur_time, NULL);
                
                if(tv_diff_usec(cur_time, prev_time) > 1000000.0 / s->frame->fps) {
                        sem_post(&s->have_item);
                        tv_add_usec(&prev_time, 1000000.0 / s->frame->fps);
                } else {
                        continue;
                }
                
                char *filename = s->glob.gl_pathv[s->index++];
                int fd = open(filename, O_RDONLY);
                ssize_t bytes_read = 0;
                do {
                        bytes_read += pread(fd, s->buffers[s->buffer_read] + bytes_read,
                                        s->tile->data_len - bytes_read,
                                        s->file_information.offset + bytes_read);
                } while(bytes_read < s->tile->data_len);
                
                s->lut_func(s->lut, s->buffers[s->buffer_read], s->tile->data_len);
                
                s->tile->data = s->buffers[s->buffer_read];
                s->buffer_read = (s->buffer_read + 1) % 2; /* and we will read next one */
                close(fd);
                if( s->index == s->glob.gl_pathc)
                        s->index = 0;
        }
        
        return NULL;
}

struct video_frame *
vidcap_dpx_grab(void *state, struct audio_frame **audio)
{
        
	struct vidcap_dpx_state 	*s = (struct vidcap_dpx_state *) state;

        sem_wait(&s->have_item);

        s->frames++;
        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);    
        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            fprintf(stderr, "%d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }  

	return s->frame;
}

