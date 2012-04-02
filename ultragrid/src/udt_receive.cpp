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
#include "udt_receive.h"
#include "video.h"

using namespace std;

struct udt_recv {
        udt_recv(char *address, unsigned int port)
        {
                UDT::startup();
                socket = UDT::socket(AF_INET, SOCK_DGRAM, 0);

                if(socket == UDT::INVALID_SOCK) {
                        throw std::runtime_error(std::string("socket: ") + UDT::getlasterror().getErrorMessage());
                }

                int err;

                err = UDT::ERROR;

#if 0
                struct timeval tv;
                tv.tv_sec = 5;  /* 5 Secs Timeout */
                tv.tv_usec = 0;
                setsockopt(this->fd, SOL_SOCKET, SO_RCVTIMEO,(struct timeval *)&tv,sizeof(struct timeval));
                setsockopt(this->fd, SOL_SOCKET, SO_SNDTIMEO,(struct timeval *)&tv,sizeof(struct timeval));
#endif
                sockaddr_in my_addr;
                my_addr.sin_family = AF_INET;
                my_addr.sin_port = htons(port);
                my_addr.sin_addr.s_addr = INADDR_ANY;
                memset(&(my_addr.sin_zero), '\0', 8);

                err = UDT::bind(socket, (sockaddr *) &my_addr, sizeof(my_addr));

                if (err == UDT::ERROR) {
                        throw std::runtime_error(std::string("bind: ") + UDT::getlasterror().getErrorMessage());
                }

                err = UDT::listen(socket, 10);

                if (err == UDT::ERROR) {
                        throw std::runtime_error(std::string("listen: ") + UDT::getlasterror().getErrorMessage());
                }
        }

        void accept()
        {
                int namelen;
                sockaddr_in their_addr;

                recver = UDT::accept(socket, (sockaddr*)&their_addr, &namelen);
                if(recver == UDT::INVALID_SOCK) {
                        throw std::runtime_error(std::string("accept: ") + UDT::getlasterror().getErrorMessage());
                }
        }


        int receive(char *buffer, int *len)
        {
                int res = UDT::recvmsg(recver, buffer, *len);
                if(recver == UDT::ERROR) {
                        std::cerr << UDT::getlasterror().getErrorMessage();
                        return 0;
                }
                *len = res;

                return 1;
        }

        private:

        UDTSOCKET socket;
        UDTSOCKET recver;
};

struct udt_recv *udt_receive_init(char *address, unsigned int port)
{
        struct udt_recv *s;

        s = new udt_recv(address, port);

        return s;
}

void udt_receive_done(struct udt_recv *s)
{
}

int udt_receive(struct udt_recv  *udt_receive, char *buffer, int *len)
{
        return udt_receive->receive(buffer, len);
}

int udt_receive_accept(struct udt_recv  *udt_receive)
{
    int ret = 1;
    try {
        udt_receive->accept();
    } catch (...) {
        ret = 0;
    }

    return ret;
}
