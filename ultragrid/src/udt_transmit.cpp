/*
 * FILE:     udt.c
 * AUTHOR:  Colin Perkins <csp@csperkins.org>
 *          Ladan Gharai
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2004 University of Southern California
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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

#include <udt.h>
#include <stdexcept>
#include <iostream>

#include "config.h"
#include "config_unix.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "udt_transmit.h"
#include "video.h"
#include "video_codec.h"

using namespace std;

#define TTL_MS 1000

struct udt_transmit {
        udt_transmit(char *address, unsigned int port)
        {
                UDT::startup();
                socket = UDT::socket(AF_INET, SOCK_DGRAM, 0);

                struct addrinfo hints, *res;
                int err;

                memset(&hints, 0, sizeof(hints));
                hints.ai_family = AF_INET;
                hints.ai_socktype = SOCK_DGRAM;
                char port_str[6];
                snprintf(port_str, 5, "%u", port);
                err = getaddrinfo(address, port_str, &hints, &res);

                if(err) {
                        throw std::runtime_error(std::string("getaddrinfo: ") + gai_strerror(err) + " (" + address + ")");
                }

                std::string what;

                err = UDT::ERROR;

                struct timeval tv;
                tv.tv_sec = 3;  /* 5 Secs Timeout */
                tv.tv_usec = 0;

                //UDT::setsockopt(socket, /* unused */ 0, UDT_SNDTIMEO, (const char *) &tv, sizeof(struct timeval));
                //UDT::setsockopt(socket, /* unused */ 0, UDT_RCVTIMEO, (const char *) &tv, sizeof(struct timeval));

                err = UDT::connect(socket, res->ai_addr, res->ai_addrlen);

                if (err == UDT::ERROR) {
                        throw std::runtime_error(std::string("connect: ") + UDT::getlasterror().getErrorMessage());
                }


                freeaddrinfo(res);
        }

        ~udt_transmit() {
                UDT::close(socket);
                UDT::cleanup();

        }

        void send(struct video_frame *frame)
        {
                send_description(frame);

                int res = UDT::sendmsg(socket, frame->tiles[0].data, frame->tiles[0].data_len, TTL_MS, 0);
                if(res == UDT::ERROR) {
                        std::cerr << res << " " << UDT::getlasterror().getErrorMessage();
                }
        }

        void send_description(struct video_frame *frame)
        {
                assert(frame->tile_count == 1);

                struct tile *tile = vf_get_tile(frame, 0);
                video_payload_hdr_t payload_hdr;
                uint32_t tmp;
                unsigned int fps, fpsd, fd, fi;

                payload_hdr.hres = htons(tile->width);
                payload_hdr.vres = htons(tile->height);
                payload_hdr.fourcc = htonl(get_fourcc(frame->color_spec));
                payload_hdr.length = htonl(tile->data_len);
                payload_hdr.frame = htonl(frame->frames);
                // unused payload_hdr.substream_bufnum
                // unused payload_hdr.offset

                /* word 6 */
                tmp = frame->interlacing << 29;
                fps = round(frame->fps);
                fpsd = 1;
                if(fabs(frame->fps - round(frame->fps) / 1.001) < 0.005)
                        fd = 1;
                else
                        fd = 0;
                fi = 0;

                tmp |= fps << 19;
                tmp |= fpsd << 15;
                tmp |= fd << 14;
                tmp |= fi << 13;
                payload_hdr.il_fps = htonl(tmp);

                int res = UDT::sendmsg(socket, (char *) &payload_hdr, sizeof(payload_hdr), TTL_MS, 0);
                if(res == UDT::ERROR) {
                        std::cerr << res << " " << UDT::getlasterror().getErrorMessage();
                }
        }

        private:

        UDTSOCKET socket;
};

struct udt_transmit *udt_transmit_init(char *address, unsigned int port)
{
        udt_transmit *s = 0;

        try {
            s = new udt_transmit(address, port);
        } catch (...) {
        }

        return s;
}

void udt_transmit_done(struct udt_transmit *s)
{
    delete s;
}

void udt_send(struct udt_transmit  *s, struct video_frame *frame)
{
        s->send(frame);
}
