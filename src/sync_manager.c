/*
 * FILE:   lib_common.h
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2003 University of Southern California
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
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
#endif // HAVE_CONFIG_H

#include "sync_manager.h"

#define BIND_PORT 5200
#define ADDRESS "255.255.255.255"

struct sync_manager {
        bool master;
        int fd;
        uint32_t seq;

        struct in_addr addr;
};

struct sync_manager *sync_init(bool master)
{
        struct sync_manager *s = malloc(sizeof(struct sync_manager));
        struct sockaddr_in s_in;
        int res;

        assert(s != NULL);

        s->master = master;
        s->fd = socket(AF_INET, SOCK_DGRAM, 0);
        assert(s->fd != -1);
        int value = 1;
        res = setsockopt(s->fd, SOL_SOCKET, SO_BROADCAST, &value, sizeof(value));
        assert(res == 0);


        s_in.sin_family = AF_INET;
        s_in.sin_addr.s_addr = INADDR_ANY;
        s_in.sin_port = htons(BIND_PORT);

        res = bind(s->fd, (struct sockaddr *) &s_in, sizeof(s_in));
        assert(res == 0);

        res = inet_pton(AF_INET, ADDRESS, &s->addr);
        assert(res = 1);

        s->seq = 0;

        return s;
}

bool  sync_wait(struct sync_manager *s)
{
        if(!s) {
                return;
        }

        if(s->master) {
                struct sockaddr_in s_in;
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = s->addr.s_addr;
                s_in.sin_port = htons(BIND_PORT);

                ssize_t sent_bytes = sendto(s->fd, &s->seq, sizeof(s->seq), 0, (struct sockaddr *) &s_in, sizeof(s_in));
                s->seq++;
                if(sent_bytes == sizeof(uint32_t)) {
                        return true;
                } else {
                        fprintf(stderr, "Failed to send token.\n");
                }
        } else {
                struct timeval timeout;
                timeout.tv_sec = 2;
                timeout.tv_usec = 0;

                fd_set select_fd;
                FD_ZERO(&select_fd);
                FD_SET(s->fd, &select_fd);

                int res = select(s->fd + 1, &select_fd, NULL, NULL, &timeout);
                uint32_t curr_seq;
                if(res == 1) {
                        read(s->fd, &curr_seq, sizeof(uint32_t));
                        if(curr_seq != s->seq + 1) {
                                fprintf(stderr, "Expectinq seq %d, got %d.\n", s->seq + 1, curr_seq);
                        }
                        s->seq = curr_seq;
                        return true;
                } else if(res == 0) {
                        fprintf(stderr, "Timeout.\n");
                        return false;
                } else if(res == -1) {
                        fprintf(stderr, "Error.\n");
                        return false;
                }
        }

        // newer reach here
        abort();
}

void sync_destroy(struct sync_manager *s)
{
        if(s) {
                close(s->fd);
                free(s);
        }
}

