/*
 * FILE:     tcp.c
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

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "config.h"
#include "config_unix.h"
#include "tcp_receive.h"
#include "video.h"
#include "client-gui/include/Utils.h"

using namespace std;

struct tcp_recv {
        tcp_recv(const char *address, unsigned int port)
        {
        }

        bool accept(const char *remote_host, int remote_port)
        {
                struct addrinfo hints, *res, *res0;
                int err;

                memset(&hints, 0, sizeof(hints));
                hints.ai_family = AF_UNSPEC;
                hints.ai_socktype = SOCK_STREAM;
                char port_str[10];
                snprintf(port_str, 10, "%d", remote_port);
                err = getaddrinfo(remote_host, port_str, &hints, &res0);

                if(err) {
                    throw std::runtime_error(std::string("getaddrinfo: ") + gai_strerror(err) + " (" + remote_host + ")");
                }

                this->socket_fd = -1;

                std::string what;

                for (res = res0; res; res = res->ai_next) {
                    this->socket_fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
                    if  (this->socket_fd < 0) {
                        what = std::string("socket failed: ") + strerror(errno);
                        continue;
                    }

                    struct timeval tv;
                    tv.tv_sec = 5;  /* 5 Secs Timeout */
                    tv.tv_usec = 0;
                    setsockopt(this->socket_fd, SOL_SOCKET, SO_RCVTIMEO,(struct timeval *)&tv,sizeof(struct timeval));
                    setsockopt(this->socket_fd, SOL_SOCKET, SO_SNDTIMEO,(struct timeval *)&tv,sizeof(struct timeval));

                    //if(connect(this->fd, res->ai_addr, res->ai_addrlen) == -1) {
                    if(Utils::conn_nonb(* (struct sockaddr_in *) res->ai_addr, this->socket_fd, 5)) {
                        this->socket_fd = -1;

                        std::stringstream out;
                        out << remote_port;
                        what = std::string("connect failed ") + remote_host + ":" + out.str() + ")" + strerror(errno);
                        continue;
                    }

                    break; /* okay we got one */
                }

                freeaddrinfo(res0);

                if(this->socket_fd < 0 ) {
                    std::cerr << what << std::endl;
                    return false;
                }

                return true;
        }

        void disconnect()
        {
                close(this->socket_fd);
        }


        int receive(char *buffer, int *len)
        {
            ssize_t ret = read(this->socket_fd, buffer, *len);

            *len = ret;
            if(ret == 0) {
                std::cerr << "tcp receive: timeout" << std::endl;

                return 0;
            } else if(ret == -1) {
                std::cerr << "tcp receive: error" << std::endl;

                return 0;
            }

            return 1;
        }

        private:

        int socket_fd;

        //int tcp_epoll_id;
};

void *tcp_receive_init(const char *address, unsigned int port)
{
    struct tcp_recv *s = 0;

    try {
            s = new tcp_recv(address, port);
    } catch (std::exception &e) {
            std::cerr << e.what() << std::endl;
    }

    return (void *) s;
}

void tcp_receive_done(void *state)
{
    if(state == 0)
        return;

    tcp_recv *s = (tcp_recv *) state;

    delete s;
}

int tcp_receive(void *state, char *buffer, int *len)
{
    if(state == 0)
        return 0;

    tcp_recv *s = (tcp_recv *) state;

    return s->receive(buffer, len);
}

int tcp_receive_accept(void *state, const char *remote_host, int remote_port)
{
    if(state == 0)
        return 0;

    tcp_recv *s = (tcp_recv *) state;

    int ret = 1;
    try {
        if(!s->accept(remote_host, remote_port)) {
            ret = 0;
        }
    } catch (std::exception &e) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": Catched exception: " << e.what() << std::endl;
        ret = 0;
    }

    return ret;
}

int tcp_receive_disconnect(void *state)
{
    if(state == 0)
        return 0;

    tcp_recv *s = (tcp_recv *) state;

    s->disconnect();

    return 1;
}
