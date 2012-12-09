#include "config.h"
#include "config_unix.h"

#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <omp.h>
#include <tv.h>

#include "../include/Utils.h"

#include "video.h"

using namespace std;

Utils::Utils()
{
    //ctor
}

Utils::~Utils()
{
    //dtor
}


wxString Utils::FromCDouble(double value, int precision)
{
    wxString ret;
    ret << value;
    ret.Replace(wxT(","), wxT("."));

    return ret;
}

//do a nonblocking connect
//  return -1 on a system call error, 0 on success
//  sa - host to connect to, filled by caller
//  sock - the socket to connect
//  timeout - how long to wait to connect
int
Utils::conn_nonb(struct sockaddr_in sa, int sock, int timeout)
{
    int flags = 0, error = 0, ret = 0;
    fd_set  rset, wset;
    socklen_t   len = sizeof(error);
    struct timeval  ts;

    ts.tv_sec = timeout;
    ts.tv_usec = 0;

    //clear out descriptor sets for select
    //add socket to the descriptor sets
    FD_ZERO(&rset);
    FD_SET(sock, &rset);
    wset = rset;    //structure assignment ok

    //set socket nonblocking flag
    if( (flags = fcntl(sock, F_GETFL, 0)) < 0)
        return -1;

    if(fcntl(sock, F_SETFL, flags | O_NONBLOCK) < 0)
        return -1;

    //initiate non-blocking connect
    if( (ret = connect(sock, (struct sockaddr *)&sa, sizeof(sa))) < 0 )
        if (errno != EINPROGRESS)
            return -1;

    if(ret == 0)    //then connect succeeded right away
        goto done;

    //we are waiting for connect to complete now
    if( (ret = select(sock + 1, &rset, &wset, NULL, (timeout) ? &ts : NULL)) < 0)
        return -1;
    if(ret == 0){   //we had a timeout
        errno = ETIMEDOUT;
        return -1;
    }

    //we had a positivite return so a descriptor is ready
    if (FD_ISSET(sock, &rset) || FD_ISSET(sock, &wset)){
        if(getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &len) < 0)
            return -1;
    }else
        return -1;

    if(error){  //check if we had a socket error
        errno = error;
        return -1;
    }

done:
    //put socket back in blocking mode
    if(fcntl(sock, F_SETFL, flags) < 0)
        return -1;

    return 0;
}

bool Utils::boolFromString(string str)
{
    if (str.compare(std::string("true")) == 0) {
        return true;
    } else {
        return false;
    }
}

void rgb2yuv422(unsigned char *in, unsigned int width, unsigned int height)
{
        unsigned int i, j;
        int r, g, b;
        int y, u, v, y1, u1, v1;
        unsigned char *dst;

        dst = in;

        for (j = 0; j < height; j++) {
                for (i = 0; i < width; i += 2) {
                        r = *(in++);
                        g = *(in++);
                        b = *(in++);
                        in++;   /*skip alpha */

                        y = r * 0.299 + g * 0.587 + b * 0.114;
                        u = b * 0.5 - r * 0.168736 - g * 0.331264;
                        v = r * 0.5 - g * 0.418688 - b * 0.081312;
                        //y -= 16;
                        if (y > 255)
                                y = 255;
                        if (y < 0)
                                y = 0;
                        if (u < -128)
                                u = -128;
                        if (u > 127)
                                u = 127;
                        if (v < -128)
                                v = -128;
                        if (v > 127)
                                v = 127;
                        u += 128;
                        v += 128;

                        r = *(in++);
                        g = *(in++);
                        b = *(in++);
                        in++;   /*skip alpha */

                        y1 = r * 0.299 + g * 0.587 + b * 0.114;
                        u1 = b * 0.5 - r * 0.168736 - g * 0.331264;
                        v1 = r * 0.5 - g * 0.418688 - b * 0.081312;
                        if (y1 > 255)
                                y1 = 255;
                        if (y1 < 0)
                                y1 = 0;
                        if (u1 < -128)
                                u1 = -128;
                        if (u1 > 127)
                                u1 = 127;
                        if (v1 < -128)
                                v1 = -128;
                        if (v1 > 127)
                                v1 = 127;
                        u1 += 128;
                        v1 += 128;

                        *(dst++) = (u + u1) / 2;
                        *(dst++) = y;
                        *(dst++) = (v + v1) / 2;
                        *(dst++) = y1;
                }
        }
}

unsigned char *tov210(unsigned char *in, unsigned int width,
                      unsigned int aligned_x, unsigned int height, double bpp)
{
        struct packed {
                unsigned a:10;
                unsigned b:10;
                unsigned c:10;
                unsigned p1:2;
        } *p;
        unsigned int i, j;

        unsigned int linesize = aligned_x * bpp;

        unsigned char *dst = (unsigned char *)malloc(aligned_x * height * bpp);
        unsigned char *src;
        unsigned char *ret = dst;

        for (j = 0; j < height; j++) {
                p = (struct packed *)dst;
                dst += linesize;
                src = in;
                in += width * 2;
                for (i = 0; i < width; i += 3) {
                        unsigned int u, y, v;

                        u = *(src++);
                        y = *(src++);
                        v = *(src++);

                        p->a = u << 2;
                        p->b = y << 2;
                        p->c = v << 2;
                        p->p1 = 0;

                        p++;

                        u = *(src++);
                        y = *(src++);
                        v = *(src++);

                        p->a = u << 2;
                        p->b = y << 2;
                        p->c = v << 2;
                        p->p1 = 0;

                        p++;
                }
        }
        return ret;
}


void Utils::toV210(char *src, char *dst, int width, int height)
{
    struct packed_in {
        unsigned c:10;
        unsigned b:10;
        unsigned a:10;
        unsigned p1:2;
    };

    struct packed_out {
        unsigned a:10;
        unsigned b:10;
        unsigned c:10;
        unsigned p1:2;
    } ;


#pragma omp parallel for
    for(int j = 0; j < height; ++j) {
    struct packed_in *in = (struct packed_in *) src + j * width;
    struct packed_out *out = (struct packed_out *) dst + j * width * 4 / 6;

        for (int i = 0; i < width; i += 6) {
            register long int y_1, u, v;
            register long int y_2;

           register struct packed_in in1, in2;

            in1 = *in++;
            in2 = *in++;
            y_1 = (11993 * in1.a + 40239 * in1.b + 4063 * in1.c) / 65536 + 64;
            y_2 = (11993 * in2.a + 40239 * in2.b + 4063 * in2.c) / 65536 + 64;
            u = (-6619 / 2 * (in1.a + in2.a) - 22151 / 2 * (in1.b + in2.b) + 29870 / 2 * (in1.c + in2.c)) / 65536 + 512;
            v = (28770 / 2 * (in1.a + in2.a) - 26148 / 2 * (in1.b + in2.b) - 2621 / 2 * (in1.c + in2.c))  / 65536 + 512;

            out->a = u;
            out->b = y_1;
            out->c = v;
            out->p1 = 3;
            out += 1;
            out->a = y_2;

            in1 = *in++;
            in2 = *in++;
            y_1 = (11993 * in1.a + 40239 * in1.b + 4063 * in1.c) / 65536 + 64;
            y_2 = (11993 * in2.a + 40239 * in2.b + 4063 * in2.c) / 65536 + 64;
            u = (-6619 / 2 * (in1.a + in2.a) - 22151 / 2 * (in1.b + in2.b) + 29870 / 2 * (in1.c + in2.c)) / 65536 + 512;
            v = (28770 / 2 * (in1.a + in2.a) - 26148 / 2 * (in1.b + in2.b) - 2621 / 2 * (in1.c + in2.c))  / 65536 + 512;

            out->b = u;
            out->c = y_1;
            out->p1 = 3;
            out += 1;
            out->a = v;
            out->b = y_2;

            in1 = *in++;
            in2 = *in++;
            y_1 = (11993 * in1.a + 40239 * in1.b + 4063 * in1.c) / 65536 + 64;
            y_2 = (11993 * in2.a + 40239 * in2.b + 4063 * in2.c) / 65536 + 64;
            u = (-6619 / 2 * (in1.a + in2.a) - 22151 / 2 * (in1.b + in2.b) + 29870 / 2 * (in1.c + in2.c)) / 65536 + 512;
            v = (28770 / 2 * (in1.a + in2.a) - 26148 / 2 * (in1.b + in2.b) - 2621 / 2 * (in1.c + in2.c))  / 65536 + 512;

            out->c = u;
            out->p1 = 3;
            out += 1;
            out->a = y_1;
            out->b = v;
            out->c = y_2;
            out->p1 = 3;
            out += 1;
        }
    }
}

void Utils::scale(int sw, int sh, int src_pitch_pix, int *s, int dw, int dh, int *d, int color)
{
	float yadd = (float)sh/dh;
	float xadd = (float)sw/dw;

	int y;

	if(color == (R|G|B)) {
	    // simple case
#pragma omp parallel for
        for(y = 0; y < dh; y += 1) {
            int *dst = d + y * dw;
            int *src_line = s + src_pitch_pix * (int) (y * yadd);
            for(int x = 0; x < dw; ++x) {
                *dst++ = src_line[(int) (x * xadd)];
            }
        }
	} else if(color < LUMA) {
	    // color selection

        unsigned int mask = 0;
        if(color & R) {
            mask |= 0x3ff << 22;
        }
        if(color & G) {
            mask |= 0x3ff << 12;
        }
        if(color & B) {
            mask |= 0x3ff << 2;
        }

#pragma omp parallel for
        for(y = 0; y < dh; y += 1) {
            unsigned int *dst = (unsigned int *) d + y * dw;
            unsigned int *src_line = (unsigned int *) s + src_pitch_pix * (int) (y * yadd);
            for(int x = 0; x < dw; ++x) {
                *dst++ = src_line[(int) (x * xadd)] & mask;
            }
        }
	} else if(color == LUMA) {
#pragma omp parallel for
        for(y = 0; y < dh; y += 1) {
            struct packed {
                unsigned a:10;
                unsigned b:10;
                unsigned c:10;
                unsigned p1:2;
            };

            struct packed *dst = (struct packed *) d + y * dw;
            struct packed *src_line = (struct packed *) s + src_pitch_pix * (int) (y * yadd);
            for(int x = 0; x < dw; ++x) {
                struct packed packed = src_line[(int) (x * xadd)];
                register unsigned int val = packed.a * 0.0722f + packed.b * 0.7152f + packed.c * 0.2126f;

                packed.a = val;
                packed.b = val;
                packed.c = val;

                *dst++ = packed;
            }
        }
	} else if(color == MONO) {
#pragma omp parallel for
        for(y = 0; y < dh; y += 1) {
            struct packed {
                unsigned a:10;
                unsigned b:10;
                unsigned c:10;
                unsigned p1:2;
            };

            unsigned int *dst = (unsigned int *) d + y * dw;
            struct packed *src_line = (struct packed *) s + src_pitch_pix * (int) (y * yadd);
            for(int x = 0; x < dw; ++x) {
                struct packed packed = src_line[(int) (x * xadd)];
                register unsigned int val = packed.a * 0.0722f + packed.b * 0.7152f + packed.c * 0.2126f;
                if(val > 1024/2) {
                    *dst++ = 0xffffffffu;
                } else {
                    *dst++ = 0u;
                }
            }
        }
	}
}

string Utils::VideoDescSerialize(struct video_desc *mode)
{
    ostringstream ostr;

    string interlacingFlag;
    switch (mode->interlacing) {
        case PROGRESSIVE:       interlacingFlag = "p"; break;
        case UPPER_FIELD_FIRST: interlacingFlag = "uff"; break;
        case LOWER_FIELD_FIRST: interlacingFlag = "lff"; break;
        case INTERLACED_MERGED: interlacingFlag = "i"; break;
        case SEGMENTED_FRAME:   interlacingFlag = "psf"; break;
    }

    ostr << mode->width << "_" << mode->height << "_" << interlacingFlag << "_"<< mode->fps;
    return ostr.str();
}

struct video_desc Utils::VideoDescDeserialize(string modeStr)
{
    char *str = strdup(modeStr.c_str());
    char *to_be_deleted = str;
    char *item, *save_ptr;
    int counter = 0;
    struct video_desc ret;

    const char *interlacingFlag;

    while((item = strtok_r(str, "_", &save_ptr))) {
        switch(counter++) {
            case 0:
                ret.width = atoi(item);
                break;
            case 1:
                ret.height = atoi(item);
                break;
            case 2:
                interlacingFlag = item;
                break;
            case 3:
                ret.fps = atof(item);
        }

        str = NULL;
    }

    if(counter != 4) {
        memset(&ret, 0, sizeof(ret));
        return ret;
    }

    if(strcmp(interlacingFlag, "p") == 0) {
        ret.interlacing = PROGRESSIVE;
    } else if(strcmp(interlacingFlag, "uff") == 0) {
        ret.interlacing = UPPER_FIELD_FIRST;
    } else if(strcmp(interlacingFlag, "lff") == 0) {
        ret.interlacing = LOWER_FIELD_FIRST;
    } else if(strcmp(interlacingFlag, "i") == 0) {
        ret.interlacing = INTERLACED_MERGED;
    } else if(strcmp(interlacingFlag, "psf") == 0) {
        ret.interlacing = SEGMENTED_FRAME;
    }

    free(to_be_deleted);

    return ret;
}
