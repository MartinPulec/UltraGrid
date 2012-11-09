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
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <udt.h>
#include <stdexcept>
#include <iostream>

#include "audio/audio.h"
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
                this->address = strdup(address);
                this->port = port;

                UDT::startup();
                socket = UDT::socket(AF_INET, SOCK_DGRAM, 0);
        }

        void accept() {
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

                int timeout = 3;
                //UDT::setsockopt(socket, /* unused */ 0, UDT_SNDTIMEO, (const char *) &timeout, sizeof(int));
                //UDT::setsockopt(socket, /* unused */ 0, UDT_RCVTIMEO, (const char *) &timeout, sizeof(int));

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

        void send(struct video_frame *frame, struct audio_frame *audio)
        {
                send_description(frame, audio);

                int res = UDT::sendmsg(socket, frame->tiles[0].data, frame->tiles[0].data_len, TTL_MS, 0);
                if(res == UDT::ERROR) {
                        std::cerr << res << " " << UDT::getlasterror().getErrorMessage();
                }
                if(res != frame->tiles[0].data_len) {
                        std::cerr << "Sent only " << res << "B, " << frame->tiles[0].data_len << "B was scheduled!" << std::endl;
                }
                if(audio) {
                        int res = UDT::sendmsg(socket, audio->data, audio->data_len, TTL_MS, 0);
                        if(res == UDT::ERROR) {
                                std::cerr << res << " " << UDT::getlasterror().getErrorMessage();
                        }
                        if(res != audio->data_len) {
                                std::cerr << "Sent only " << res << "B, " << audio->data_len << "B was scheduled!" << std::endl;
                        }
                }
        }

        void send_description(struct video_frame *frame, struct audio_frame *audio)
        {
                assert(frame->tile_count == 1);

                struct tile *tile = vf_get_tile(frame, 0);
                uint32_t payload_hdr[PCKT_HDR_BASE_LEN + 
                        PCKT_EXT_INFO_LEN +
                        PCKT_HDR_AUDIO_LEN];
                uint32_t tmp;
                unsigned int fps, fpsd, fd, fi;

                payload_hdr[PCKT_LENGTH] = htonl(tile->data_len);
                payload_hdr[PCKT_HRES_VRES] = htonl(tile->width << 16 | tile->height);
                payload_hdr[PCKT_FOURCC] = htonl(get_fourcc(frame->color_spec));
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
                payload_hdr[PCKT_IL_FPS] = htonl(tmp);

                int next_header = audio == NULL ? 0 : 1;
                payload_hdr[PCKT_SEQ_NEXT_HDR] = htonl(frame->frames << 1 | next_header);

                if(audio) {
                        payload_hdr[PCKT_HDR_BASE_LEN] = htonl(PCKT_EXT_AUDIO_TYPE << 28 |
                                        PCKT_HDR_AUDIO_LEN << 12 |
                                        0); 

                        uint32_t *audio_hdr = payload_hdr + PCKT_HDR_BASE_LEN + PCKT_EXT_INFO_LEN;
                        audio_hdr[PCKT_EXT_AUDIO_LENGTH] = htonl(audio->data_len);
                        audio_hdr[PCKT_EXT_AUDIO_QUANT_SAMPLE_RATE] = htonl((audio->bps * 8) << 26 |
                                        audio->sample_rate);
                        audio_hdr[PCKT_EXT_AUDIO_CHANNEL_COUNT] = htonl(audio->ch_count);
                        audio_hdr[PCKT_EXT_AUDIO_TAG] = htonl(0x1); // PCM
                }

                size_t length = PCKT_HDR_BASE_LEN;
                if(audio) {
                        length += PCKT_EXT_INFO_LEN + PCKT_HDR_AUDIO_LEN;
                }
                int res = UDT::sendmsg(socket, (char *) &payload_hdr,
                                length, TTL_MS, 0);
                if(res == UDT::ERROR) {
                        std::cerr << res << " " << UDT::getlasterror().getErrorMessage();
                }
        }

        private:

        char * address;
        int port;

        UDTSOCKET socket;
};

void *udt_transmit_init(char *address, unsigned int *port)
{
        udt_transmit *s = 0;

        try {
            s = new udt_transmit(address, *port);
        } catch (std::exception &e) {
                std::cerr << "UDT: unable to initalize transmit: " << e.what() << std::endl;
        }

        return s;
}

void udt_transmit_accept(void *state)
{
        udt_transmit *s = (udt_transmit *) state;

        try {
            s->accept();
        } catch (std::exception &e) {
                std::cerr << "UDT: unable to accept: " << e.what() << std::endl;
        }
}

void udt_transmit_done(void *state)
{
    udt_transmit *s = (udt_transmit *) state;
    delete s;
}

void udt_send(void *state, struct video_frame *frame, struct audio_frame *audio)
{
    udt_transmit *s = (udt_transmit *) state;
    s->send(frame, audio);
}

