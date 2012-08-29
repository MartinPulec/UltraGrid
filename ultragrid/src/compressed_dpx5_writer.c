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

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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

void write_all(int fd, char *data, size_t bytes);

void write_all(int fd, char *data, size_t bytes) {
        do {
                ssize_t written = write(fd, data, bytes);
                assert(written > 0);
                bytes -= written;
                data += written;
        } while(bytes > 0);
}

int main(int argc, char **argv) {
        int width, height;
        char *in_filename;
        char *out_filename;

        FileInformation file_information;
        Image_Information image_information;
        Image_Orientation image_orientation;
        Motion_Picture_Film motion_header;
        Television_Header television_header;

        assert(argc == 5);

        width = atoi(argv[1]);
        height = atoi(argv[2]);
        in_filename = argv[3];
        out_filename = argv[4];

        memset(&file_information, 0, sizeof(file_information));
        memset(&image_information, 0, sizeof(image_information));
        memset(&image_orientation, 0, sizeof(image_orientation));
        memset(&motion_header, 0, sizeof(motion_header));
        memset(&television_header, 0, sizeof(television_header));

        file_information.magic_num = 'DXT5';
        file_information.offset = sizeof(file_information) +
                sizeof(image_information) +
                sizeof(image_orientation) +
                sizeof(motion_header) +
                sizeof(television_header);

        image_information.pixels_per_line = width;
        image_information.lines_per_image_ele = height;

        int in_fd = open(in_filename, O_RDONLY);
        int out_fd = open(out_filename, O_WRONLY | O_CREAT | O_EXCL, 0666);

        assert(in_fd != -1 && out_fd != -1);

        write_all(out_fd, (char *) &file_information, sizeof(file_information));
        write_all(out_fd, (char *) &image_information, sizeof(image_information));
        write_all(out_fd, (char *) &image_orientation, sizeof(image_orientation));
        write_all(out_fd, (char *) &motion_header, sizeof(motion_header));
        write_all(out_fd, (char *) &television_header, sizeof(television_header));


        size_t bytes = width * height;

        char *buffer = malloc(1024 * 1024);

        do {
                int data_to_read = 1024 * 1024;
                if(data_to_read > bytes) {
                        data_to_read = bytes;
                }
                ssize_t read_bytes = read(in_fd, buffer, data_to_read);
                assert(read_bytes > 0);

                char *ptr = buffer;

                do {
                        ssize_t written = write(out_fd, ptr, read_bytes);
                        assert(written > 0);
                        read_bytes -= written;
                        ptr += written;
                } while(read_bytes > 0);

                bytes -= data_to_read;
        } while(bytes > 0);

        assert(close(in_fd) == 0);
        assert(close(out_fd) == 0);

        free(buffer);

        return 0;
}



