/*
 * FILE:  
 *
 * Copyright (c) 2012 CESNET z.s.p.o.
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
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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

#include <map>
#include <stdint.h>
#include <cstdio>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <iostream>

#include <signal.h>
#include <sys/wait.h>


using namespace std;

#include "sp_server.h"
#include "streaming_server_handlers.h"
#include "streaming_server.h"

static volatile bool should_exit = false;

class streaming_server;

streaming_server *server;
sp_serv *sp_server;

void state_change(enum session_state_change state_change, uint32_t id, const char *hostname, void *udata);
void recv(uint32_t id, struct msg *message, responder * resp, void *udata);

const int port = 5100;

void signal_handler(int sig)
{
        if(sig == SIGCHLD) {
                pid_t pid = wait(NULL);
                server->died(pid);
                return;
        }
        std::cout << "Exiting." << std::endl;
        should_exit = true;
}

void streaming_server::run() {
        struct timeval timeout;

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        signal(SIGQUIT, signal_handler);
        signal(SIGHUP, signal_handler);
        signal(SIGABRT, signal_handler);
        signal(SIGCHLD, signal_handler);
        signal(SIGPIPE, SIG_IGN);

        sp_serv server(5100, state_change, recv, (void *) this);
        sp_server = &server;

        while (!should_exit) {
                timeout.tv_sec = 1;
                timeout.tv_usec = 0;

                server.recv(&timeout);
                server.update();
        }
}

bool streaming_server::read_file(std::string path, char *buffer, int *buf_len)
{
        int fd;
        int rd;
        std::string filename;

        filename = path;
#ifdef DEBUG
        std::cerr << "Requestd file: " << filename;
#endif
        fd = open(filename.c_str(), O_RDONLY);
        if(fd < 0) {
                return false;
        } else {
                rd = read(fd, buffer, *buf_len);
        std::cerr << "Requestd file: " << rd;
                *buf_len = rd;
                close(fd);
        }
        return true;
}

void streaming_server::died(pid_t pid)
{
        std::map<uint32_t, session_handler *>::iterator it;
        it = mapping.begin();
        for(; it != mapping.end() ; ++it) {
                bool positive = (*it).second->informDied(pid);;
                if(positive) {
                        sp_server->force_close((*it).first);
                }
        }
}

void state_change(enum session_state_change state_change, uint32_t id, const char *hostname, void *udata)
{
        streaming_server *serv = (streaming_server *) udata;
        map<uint32_t, session_handler *> *mapping = &serv->mapping;
        session_handler *handler;

        if(state_change == session_created) {
                handler = new session_handler(hostname);
                mapping->insert(std::pair<uint32_t, session_handler *>(id, handler));
        } else if (state_change == session_disconnected || state_change == session_timeout) {
                std::map<uint32_t, session_handler *>::iterator it;
                it = mapping->find(id);
                if(it != mapping->end()) {
                        handler = (*it).second;
                        delete handler;
                        mapping->erase((*it).first);
                }
        }
}


void recv(uint32_t id, struct msg *message, responder * resp, void *udata)
{
        streaming_server *serv = (streaming_server *) udata;
        map<uint32_t, session_handler *> *mapping = &serv->mapping;
        std::map<uint32_t, session_handler *>::iterator it;
        session_handler *handler;

        it = mapping->find(id);
        if(it != mapping->end()) {
                handler = (*it).second;
                handler->handle(message, serv, resp);
        } else {
                fprintf(stderr, "Could not find session.\n");
        }
}

int main_sp(void) {
        streaming_server serv;
        server = &serv;
        serv.run();
}

