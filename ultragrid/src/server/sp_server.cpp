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

#include <arpa/inet.h>
#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <sstream>

#include "sp_server.h"

static std::string valid_commands_str[] = { std::string("SETUP"),
        std::string("PLAY"),
        std::string("PAUSE"),
        std::string("TEARDOWN"),
        std::string("KEEPALIVE"),
        std::string("GET"),
        std::string("SET_PARAMETER")
};

uint32_t sp_rand32();

uint32_t sp_rand32() {
        uint32_t rand1, rand2;
        rand1 = rand();
        rand2 = rand();

        return (rand1 & 0xFFFF) << 16 | (rand2 & 0xFFFF);
}

sp_serv::sp_serv(uint16_t port, sp_state_change_t sp_state_change_callback, sp_recv_t sp_recv_callback, void *udata) :
        valid_commands(valid_commands_str, valid_commands_str + sizeof(valid_commands_str) / sizeof(std::string))
{
        struct sp_serv *s;
        struct sockaddr_in s_in;
        struct timeval tv;

        this->state_change_callback = sp_state_change_callback;
        this->recv_callback = sp_recv_callback;
        this->udata = udata;


        if ((this->socket_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
                perror("socket()");
                throw std::runtime_error("socket");
        }

        tv.tv_sec = 5;  /* 5 Secs Timeout */
        tv.tv_usec = 0;
        setsockopt(this->socket_fd, SOL_SOCKET, SO_RCVTIMEO,(struct timeval *)&tv,sizeof(struct timeval));
        setsockopt(this->socket_fd, SOL_SOCKET, SO_SNDTIMEO,(struct timeval *)&tv,sizeof(struct timeval));

        s_in.sin_family = AF_INET;
        s_in.sin_addr.s_addr = INADDR_ANY;
        s_in.sin_port = htons(port);

        if (bind(this->socket_fd, (struct sockaddr *)&s_in, sizeof(s_in)) != 0) {
                perror("bind");
                throw std::runtime_error("bind");
        }

        if (listen(this->socket_fd, 128)) {
                perror("listen()");
                throw std::runtime_error("listen");
        }

        FD_ZERO(&this->select_fd);
        FD_SET(this->socket_fd, &this->select_fd);
        this->max_fd = this->socket_fd;


}

sp_serv::~sp_serv() 
{
        for(std::map<int, client_session>::iterator it = this->clients.begin();
                        it != this->clients.end();
                        ++it) {
                struct msg message;
                struct resp_msg response;

                FD_CLR((*it).first, &this->select_fd);

                message.type = msg_teardown;
                if(send_msg(&(*it).second, &message, &response)) {
                        if(response.code == 200) {
                                int rc;
                                char unused;
                                rc = read((*it).first, &unused, 1);
#ifdef DEBUG
                                std::cerr << "Disconnect: read returned " << rc << "." << std::endl;
#endif
                        }
                }
                close((*it).first);
                                

                this->state_change_callback(session_disconnected, (*it).second.id, NULL, this->udata);
                clients.erase(it);
        }
        shutdown(this->socket_fd, SHUT_RDWR);
        close(this->socket_fd);
}

void sp_serv::force_close(uint32_t id)
{
        for(std::map<int, client_session>::iterator it = this->clients.begin();
                        it != this->clients.end();
                        ++it) {
                struct msg message;
                struct resp_msg response;

                if((*it).second.id != id) continue;

                FD_CLR((*it).first, &this->select_fd);

                message.type = msg_teardown;
                if(send_msg(&(*it).second, &message, &response)) {
                        if(response.code == 200) {
                                int rc;
                                char unused;
                                rc = read((*it).first, &unused, 1);
#ifdef DEBUG
                                std::cerr << "Disconnect: read returned " << rc << "." << std::endl;
#endif
                        }
                }
                close((*it).first);
                                

                this->state_change_callback(session_disconnected, (*it).second.id, NULL, this->udata);
                clients.erase(it);
        }
}

void sp_serv::recv(struct timeval *timeout)
{
        fd_set fd;
        int rc;

        memcpy(&fd, &this->select_fd, sizeof(fd_set));
        rc = select(this->max_fd + 1, &fd, NULL, NULL, timeout); 

#ifdef DEBUG
        std::cerr << "Selected " << rc << " ports." << std::endl;
#endif

        if(rc > 0) {
                if(FD_ISSET(this->socket_fd, &fd)) {
                        int new_fd;
                        struct sockaddr addr;
                        socklen_t len;
                        len = sizeof(addr);
                        new_fd = accept (this->socket_fd, &addr, &len);
                        if(new_fd == -1) {
                                throw std::runtime_error("Unable to accept connection.");
                                // TODO: handle error
                        } else {
                                char hostname[1024];
                                if(!inet_ntop(AF_INET, (const void* ) &(((struct sockaddr_in *) &addr)->sin_addr),
                                                                             hostname, sizeof(hostname))) {
                                        throw std::runtime_error("Unable to get client IP address.");
                                }
                                                
                                client_session new_session;
                                new_session.id = sp_rand32();
                                new_session.fd = new_fd;
                                clients.insert(std::pair<int, client_session>(new_fd, new_session));
                                FD_SET(new_fd, &this->select_fd);
                                if(this->max_fd < new_fd)
                                        this->max_fd = new_fd;
#ifdef DEBUG
        std::cerr << "New session " << new_session.id << " (" << hostname << ")" << std::endl;
#endif
                                this->state_change_callback(session_created, new_session.id, hostname, this->udata);
                        }
                }
                for(std::map<int, client_session>::iterator it = this->clients.begin();
                                it != this->clients.end();
                                ++it) {
                        if(FD_ISSET((*it).first, &fd)) {
                                struct msg message;
                                int rd;
                                int total = 0;

                                /*while((rd = read((*it).first, this->buffer + rd, buffer_len)) > 0) {
                                        total += rd;
                                }*/
                                total = ::recv((*it).first, this->buffer, buffer_len, 0);
                                if(total) {
                                        int i = 0;
                                        int first = 0;
                                        while(i < total) {

                                                if(this->buffer[i] == '\0') {
#ifdef DEBUG
                                                        std::cerr << "Message (" << (*it).first << "): " << std::endl;
                                                        std::cerr.write(this->buffer + first, i - first);
#endif
                                                        if(parse_message(&(*it).second,
                                                                                &message,
                                                                                this->buffer + first,
                                                                                i - first)) {
                                                                responder responder(this, &(*it).second);
                                                                this->recv_callback((*it).second.id, &message, &responder, this->udata);
                                                        }

                                                        first = i + 1;
                                                }
                                                ++i;
                                                if(i == total && this->buffer[i - 1] != '\0') {
                                                        total += ::recv((*it).first, this->buffer + total, buffer_len - total, 0);
                                                }
                                        }

                                } else {
#ifdef DEBUG
                                        std::cerr << "Client disconnected " << std::endl;
#endif
                                        this->state_change_callback(session_disconnected, (*it).second.id, NULL, this->udata);
                                        FD_CLR((*it).first, &this->select_fd);
                                        close((*it).first);
                                        clients.erase(it);

                                }
                        }
                }
        }
}

bool sp_serv::parse_message(struct client_session *client, struct msg *msg, char *bytes, int bytes_len)
{
        int start;
        int pos;
        std::string command;
        struct resp_msg response;

        bytes[bytes_len] = '\0';

        for(start = 0; start < bytes_len; ++start) {
                if(!isspace(bytes[start])) {
                        break;
                }
        }

        for(pos = start; pos < bytes_len; ++pos) {
                if(isspace(bytes[pos])) {
                        break;
                }
        }

        command.assign(bytes + start, pos - start);

        if(valid_commands.find(command) == valid_commands.end()) {
                response.code = 451;
                response.message = "Parameter Not Understood";
                response.body_len = 0;

                send_response(client, &response);
                return false;
        }

        for( ; pos < bytes_len; ++pos) {
                if(!isspace(bytes[pos])) {
                        break;
                }
        }

        if(command == std::string("SETUP")) {
                msg->type = msg_setup;
                msg->data = bytes + pos;
        } else if(command == std::string("PLAY")) {
                msg->type = msg_play;
                msg->data = bytes + pos;
        } else if(command == std::string("PAUSE")) {
                msg->type = msg_pause;
                msg->data = bytes + pos;
        } else if(command == std::string("TEARDOWN")) {
                msg->type = msg_teardown;
        } else if(command == std::string("KEEPALIVE")) {
                msg->type = msg_keepalive;
        } else if(command == std::string("GET")) {
                int len = strlen(bytes + pos);
                while(isspace(bytes[pos + --len])) {
                        bytes[pos + len] = '\0';
                }
                msg->type = msg_get;
                msg->data = bytes + pos;
        } else if(command == std::string("SET_PARAMETER")) {
                msg->type = msg_set_parameter;
                msg->data = bytes + pos;
        }

        return true;
}

void sp_serv::send_response(struct client_session *client, struct resp_msg *response)
{
        const int hdr_size = 1000;
        char buff[hdr_size];
        int len;

        len = snprintf(buff, hdr_size, "%u %s\r\n", (unsigned int) response->code, response->message);
        if(response->body_len > 0) {
                len += snprintf(buff + len, hdr_size - len, "Content-Length: %u\r\n", (unsigned int) response->body_len);
        }
        len += snprintf(buff + len, hdr_size - len, "\r\n");

        write(client->fd, buff, len);
        if(!response->body_len) {
                if(buff[len - 1] != '\0') {
                        char end = '\0';
                        write(client->fd, &end, 1);
                }
        } else {
                write(client->fd, response->body, response->body_len);
                char end = '\0';
                write(client->fd, &end, 1);
        }
}

bool sp_serv::send_msg(struct client_session *client, struct msg *message, struct resp_msg * /*out*/ response)
{
        const int hdr_size = 1000;
        const char *msg;
        int hdr_len;

        assert(message->type = msg_teardown);

        if(message->type == msg_teardown) {
                msg = "TEARDOWN";
                hdr_len = strlen(msg);
        }
        write(client->fd, msg, hdr_len);

        int rc;
        rc = ::recv(client->fd, this->buffer, buffer_len, 0);
        if(rc == -1) {
                return false;
        } else {
            char * ptr = this->buffer;
            response->code = atoi(ptr);

            while(!isspace(*ptr) && ptr < this->buffer + rc)
                ++ptr;
            while(isspace(*ptr) && ptr < this->buffer + rc)
                ++ptr;
            response->message = ptr;
            while((*ptr != '\n' && *ptr != '\r') && ptr < this->buffer + rc)
                ++ptr;
            *ptr = '\0';
            ++ptr;

            while((*ptr == '\n' || *ptr == '\r') && ptr < this->buffer + rc)
                ++ptr;
            if(strncasecmp("Content-Length:", ptr, strlen("Content-Length:")) != 0)
                response->body_len = 0;
            else {
                ptr += sizeof("Content-Length:");
                response->body_len = atoi(ptr);
                while(*ptr != '\r' && ptr < this->buffer + rc)
                    ++ptr;
                ptr += 4; /* \r\n\r\n */

                while(rc != ptr - this->buffer + response->body_len)
                        rc += ::recv(client->fd, this->buffer + rc, buffer_len, 0);

                response->body = ptr;
                if(response->body + response->body_len != this->buffer + rc) {
                    std::string msg;
                    std::stringstream str;

                    str << "Immature end of packet: " << rc << " B (expected " << response->body + response->body_len - this->buffer << " B)";
                    msg = str.str();
                    throw std::runtime_error(msg);
                }
            }
            return true;
        }
}

void sp_serv::send_response(int id, struct resp_msg *response)
{
        struct client_session *session;
        std::map<int, client_session>::iterator it;

        it = clients.find(id);
        if(it != clients.end()) {
                this->send_response(&(*it).second, response);
        } else {
                std::cerr << "Warning: Unable to send response for id " << id << std::endl;
        }

}

void sp_serv::update()
{
}

responder::responder(sp_serv *server, struct client_session *session) :
        server(server), session(session)
{
}

bool responder::send_response(struct resp_msg *response)
{
        server->send_response(session, response);
}


