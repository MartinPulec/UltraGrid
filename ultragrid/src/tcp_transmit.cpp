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

#include "abstract_transmit.h"

#include <stdexcept>
#include <iostream>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>


#include "config.h"
#include "config_unix.h"

#include "audio/audio.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "tcp_transmit.h"
#include "video.h"
#include "video_codec.h"

using namespace std;

tcp_transmit::tcp_transmit(char *address, unsigned int *port)
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

void tcp_transmit::accept() {
        struct sockaddr_in s_peer;
        socklen_t addrlen = sizeof(s_peer);
        this->data_socket_fd = ::accept(this->socket_fd, (struct sockaddr *)&s_peer, &addrlen);
        std::cerr << __FILE__ << ":" << __LINE__ << ": " << "Accepted connection." << std::endl;
}

tcp_transmit::~tcp_transmit() {
}

bool tcp_transmit::send(struct video_frame *frame, struct audio_frame *audio)
{
        size_t total = 0;
        if(!send_description(frame, audio)) {
                return false;
        }

        while(total < frame->tiles[0].data_len) {
                int res = write(this->data_socket_fd, frame->tiles[0].data + total,
                                frame->tiles[0].data_len - total);
                if(res == -1) {
                        perror("");
                        return false;
                }
                total += res;
        }
#if 0
        if(res != frame->tiles[0].data_len) {
                std::cerr << "Sent only " << res << "B, " << frame->tiles[0].data_len << "B was scheduled!" << std::endl;
        }
#endif
        if(audio) {
                total = 0;
                while(total < audio->data_len) {
                        int res = write(this->data_socket_fd, audio->data + total,
                                        audio->data_len - total);
                        if(res == -1) {
                                perror("");
                                return false;
                        }
                        total += res;
                }
#if 0
                if(res != audio->data_len) {
                        std::cerr << "Sent only " << res << "B, " << audio->data_len << "B was scheduled!" << std::endl;
                }
#endif
        }

        return true;
}

bool tcp_transmit::send_description(struct video_frame *frame, struct audio_frame *audio)
{
        assert(frame->tile_count == 1);

        struct tile *tile = vf_get_tile(frame, 0);
        uint32_t payload_hdr[PCKT_HDR_BASE_LEN + 
                PCKT_EXT_INFO_LEN +
                PCKT_HDR_AUDIO_LEN];
        size_t length = sizeof(payload_hdr);

        if(!format_description(frame, audio, payload_hdr, &length)) {
                return false;
        }
        size_t total = 0;
        while(total < length) {
                int res = write(this->data_socket_fd, (char *) &payload_hdr + total,
                                length - total);
                if(res == -1) {
                        perror("");
                        return false;
                }
                total += res;
        }

        return true;
}

