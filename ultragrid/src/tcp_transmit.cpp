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

#include <stdexcept>
#include <iostream>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>


#include "config.h"
#include "config_unix.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "tcp_transmit.h"
#include "video.h"
#include "video_codec.h"

using namespace std;

struct tcp_transmit {
        tcp_transmit(char *address, unsigned int *port)
        {
            if((this->socket_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
                perror("socket()");
                throw std::runtime_error("socket");
            }

            struct sockaddr_in s_in;

            s_in.sin_family = AF_INET;
            s_in.sin_addr.s_addr = INADDR_ANY;
            s_in.sin_port = 0; // means random port

            if (bind(this->socket_fd, (struct sockaddr *)&s_in, sizeof(s_in)) != 0) {
                    perror("bind");
                    throw std::runtime_error("bind");
            }

            socklen_t addrlen = sizeof(s_in);
            getsockname(this->socket_fd, (sockaddr*) &s_in, &addrlen);
            *port = ntohs(s_in.sin_port);
            std::cerr << __FILE__ << ":" << __LINE__ << ": Bound port: " << *port << std::endl;

            if (listen(this->socket_fd, 8)) {
                    perror("listen()");
                    throw std::runtime_error("listen");
            }
        }

        void Accept() {
            struct sockaddr_in s_peer;
            socklen_t addrlen = sizeof(s_peer);
            this->data_socket_fd = accept(this->socket_fd, (struct sockaddr *)&s_peer, &addrlen);
            std::cerr << __FILE__ << ":" << __LINE__ << ": " << "Accepted connection." << std::endl;
        }

        ~tcp_transmit() {
        }

        bool send(struct video_frame *frame)
        {
            bool rc;
            rc = send_description(frame);
            if(!rc)
                return false;


            ssize_t res, total;

            total = 0;

            char *data = frame->tiles[0].data;
            int data_len = frame->tiles[0].data_len;

            do {
                //std::cerr << __FILE__ << ":" << __LINE__ <<  std::endl;
                res = write(this->data_socket_fd, (const char *) data + total, data_len - total);
                //std::cerr << __FILE__ << ":" << __LINE__ << ": (total sent " << total << "/" << data_len << ")"<< std::endl;
                if(res == -1) {
                    std::cerr << __FILE__ << ":" << __LINE__ << ": Connection timeout" << std::endl;
                    return false;
                } else if (res == 0) {
                    std::cerr << __FILE__ << ":" << __LINE__ << ": Connection closed (total sent " << total << "/" << data_len << ")"<< std::endl;
                    return false;
                } else {
                    total += res;
                }
            } while(total < data_len);

            return true;
        }

        bool send_description(struct video_frame *frame)
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

                ssize_t res, total;

                total = 0;

                do {
                    res = write(this->data_socket_fd, (const char *) &payload_hdr + total, sizeof(payload_hdr) - total);
                    if(res == -1) {
                        std::cerr << __FILE__ << ":" << __LINE__ << ": Connection timeout" << std::endl;
                        return false;
                    } else if (res == 0) {
                        std::cerr << __FILE__ << ":" << __LINE__ << ": Connection closed" << std::endl;
                        return false;
                    } else {
                        total += res;
                    }
                } while(total < sizeof(payload_hdr));

                return true;
        }

        private:

        int socket_fd;
        int data_socket_fd;
};

void *tcp_transmit_init(char *address, unsigned int *port)
{
        tcp_transmit *s = 0;

        try {
            s = new tcp_transmit(address, port);
        } catch (...) {
        }

        return (void *) s;
}

void tcp_transmit_accept(void *state)
{
    tcp_transmit *s = (tcp_transmit *) state;

    s->Accept();
}



void tcp_transmit_done(void *state)
{
    tcp_transmit *s = (tcp_transmit *) state;

    delete s;
}

void tcp_send(void *state, struct video_frame *frame)
{
    tcp_transmit *s = (tcp_transmit *) state;

    s->send(frame);
}
