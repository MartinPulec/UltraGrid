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
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "streaming_server_handlers.h"
#include "sp_callbacks.h"
#include "sp_server.h"
#include <sys/stat.h>
#include <unistd.h>

#include <assert.h>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <string.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#include <sys/types.h>          /* See NOTES */
#include <sys/socket.h>


#define MAX_PATH_LEN 4096

extern int uv_argc;
extern char **uv_argv;

using namespace std;

//class UltraGrid
session_handler::session_handler(const char *r)
        : pid(0), state(Init), receiver(r), use_tcp(false)
{
}

session_handler::~session_handler()
{
        if(pid) {
                KillUltraGridProcess();
        }
}

bool session_handler::informDied(pid_t pid)
{
        if(this->pid == pid) {
                return true;
                this->pid = 0;
        } else {
                return false;
        }
}

void session_handler::handle(struct msg *message, streaming_server* serv, responder * resp)
{
        struct resp_msg response;
        char buffer[1000];

        response.code = -1;
        response.message = NULL;
        response.body = NULL;
        response.body_len = 0;

        if(message->type == msg_setup) {
                if(state != Init) {
                        response.code = 455;
                        response.message = "Method Not Valid in This State";
                        response.body_len = 0;
                } else {
                        char path_buff[MAX_PATH_LEN + 1];
                        struct stat sb;

#if 0
                        snprintf(path_buff, MAX_PATH_LEN, "%s/%s/", PATH_PREFIX, message->data);
#else
                        snprintf(path_buff, MAX_PATH_LEN, "%s/", message->data);
#endif
                        if(stat(path_buff, &sb) == -1) {
                                response.code = 503;
                                response.message = "Service Unavailable";
                                response.body = "Directory not found";
                                response.body_len = strlen(response.body);
                        } else {
                                path = std::string(path_buff);

                                char fd_str[6];
                                int fd[2];
                                if( socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) {
                                        perror("");
                                        response.code = 500;
                                        response.message = "Internal Server Error";
                                        response.body = "Pipe Failed";
                                        response.body_len = strlen(response.body);
                                } else {
                                        fcntl(fd[1], F_SETFD, FD_CLOEXEC);
                                        comm_fd = fd[1];

                                        struct timeval tv;
                                        tv.tv_sec = 1;  /* 5 Secs Timeout */
                                        tv.tv_usec = 500 * 1000;
                                        setsockopt(comm_fd, SOL_SOCKET, SO_RCVTIMEO,(struct timeval *)&tv,sizeof(struct timeval));
                                        setsockopt(comm_fd, SOL_SOCKET, SO_SNDTIMEO,(struct timeval *)&tv,sizeof(struct timeval));

                                        snprintf(fd_str, 6, "%d", fd[0]);

                                        pid = fork();
                                        if(pid == 0) { /* a child */
                                                char * args[100];

                                                char dpx_arg[MAX_PATH_LEN + 1];
                                                if(strcmp(color_space.c_str(), "file") == 0) {
                                                        snprintf(dpx_arg, MAX_PATH_LEN, "%s:files=%s/*.%s", video_format.c_str(), path.c_str(), glob_ext.c_str());
                                                } else {
                                                        snprintf(dpx_arg, MAX_PATH_LEN, "%s:colorspace=%s:files=%s/*.%s", video_format.c_str(), color_space.c_str(), path.c_str(), glob_ext.c_str());
                                                }
                                                int index = 0;

                                                args[index++] = uv_argv[0];
                                                args[index++] = "-t";
                                                args[index++] = dpx_arg;
                                                args[index++] = "-m";
                                                args[index++] = "1500";

                                                if(!audio.empty()) {
                                                        args[index++] = "-a";
                                                        args[index++] = strdup(audio.c_str());
                                                }

                                                if(!compression.empty()) {
                                                        args[index++] = "-c";
                                                        args[index++] = strdup(compression.c_str());
                                                }

                                                args[index++] = "-C";
                                                args[index++] = fd_str;

#if 0
                                                if(strncmp(compression.c_str(), "JPEG", 4) == 0) {
                                                        args[index++] = "-f";
                                                        args[index++] = "mult:2";
                                                }
#endif

                                                if(this->use_tcp) {
                                                        args[index++] = "-T";
                                                }

                                                args[index++] = strdup(receiver.c_str());

                                                args[index] = 0;

                                                execvp(uv_argv[0], args);

                                        } else { /* parent */
                                                if(pid == -1) { /* cannot fork */
                                                        response.code = 500;
                                                        response.message = "Internal Server Error";
                                                        response.body = "Fork Failed";
                                                        response.body_len = strlen(response.body);
                                                } else {
                                                        close(fd[0]);
                                                        int message_len;
                                                        int ret;
                                                        ssize_t total = 0;

                                                        ret = read(fd[1], &message_len, sizeof(message_len));
                                                        if(ret != sizeof(message_len))
                                                                goto launch_err;
                                                        do {
                                                                ret = read(fd[1], buffer + total, message_len - total);
                                                                if(ret <= 0) 
                                                                        goto launch_err;
                                                                total += ret;
                                                        } while (total < message_len);

                                                        state = Ready;
                                                        response.code = 201;
                                                        assert(total < sizeof(buffer));
                                                        buffer[total] = '\0';
                                                        response.message = "Created";
                                                        response.body = buffer;
                                                        response.body_len = sizeof(buffer);

                                                        if(0) {
launch_err:
                                                                response.code = 500;
                                                                response.message = "Internal Server Error";
                                                                response.body = "Getting Port";
                                                                response.body_len = strlen(response.body);
                                                        }

                                                }
                                        }
                                }
                        }
                }
                resp->send_response(&response);
        } else if(message->type == msg_play) {
                response.code = 200;
                response.message = "OK";
                response.body_len = 0;
                if(message->data[0] != '\0') {
                        int len;
                        char msg_text[40];
                        snprintf(msg_text, 40, "SETPOS %s", message->data);

                        len = strlen(msg_text);
                        write(comm_fd, &len, sizeof(len));
                        write(comm_fd, msg_text, len);
                }
                if(state == Ready) {
                        int len;
                        const char *message = "PLAY";

                        len = strlen(message);
                        write(comm_fd, &len, sizeof(len));
                        write(comm_fd, message, len);
                        state = Playing;
                } else if(state == Playing) {
                        // no action needed - just jump
                } else {
                        response.code = 455;
                        response.message = "Method Not Valid in This State (PLAY)";
                        response.body_len = 0;
                }
                resp->send_response(&response);
        } else if(message->type == msg_pause) {
                response.code = 200;
                response.message = "OK";
                response.body_len = 0;

                if(message->data[0] != '\0') {
                        int len;
                        char msg_text[40];
                        char *save_ptr = NULL;
                        char *tmp = strdup(message->data);
                        char *pos = strtok_r(tmp, " ", &save_ptr);
                        snprintf(msg_text, 40, "SETPOS %s", pos);

                        char *count = strtok_r(NULL, " ", &save_ptr);

                        len = strlen(msg_text);
                        write(comm_fd, &len, sizeof(len));
                        write(comm_fd, msg_text, len);

                        snprintf(msg_text, 40, "PLAYONE %s", count);
                        len = strlen(msg_text);
                        write(comm_fd, &len, sizeof(len));
                        write(comm_fd, msg_text, len);
                        free(tmp);
                }
                if(state == Playing) {
                        int len;
                        const char *message = "PAUSE";

                        len = strlen(message);
                        write(comm_fd, &len, sizeof(len));
                        write(comm_fd, message, len);
                        state = Ready;
                } else if(state == Ready) {
                        // no action needed
                } else {
                        response.code = 455;
                        response.message = "Method Not Valid in This State (PAUSE)";
                        response.body_len = 0;
                }
                resp->send_response(&response);
        } else if(message->type == msg_teardown) {
                // any state ...
                KillUltraGridProcess();
                state = Init;
                response.code = 200;
                response.message = "OK";
                response.body_len = 0;
                resp->send_response(&response);
        } else if(message->type == msg_keepalive) {
                response.code = 200;
                response.message = "OK";
                response.body_len = 0;
        } else if(message->type == msg_get) {
                int bufsize = 1024 * 1024;
                char * buf = new char [bufsize];
                char *home = getenv("HOME");
                if(serv->read_file(std::string(home) + "/.ugsrc/" + std::string(message->data), buf, &bufsize)) {
                        response.code = 200;
                        response.message = "OK";
                        response.body_len = bufsize;
                        response.body = buf;
                } else {
                        response.code = 404;
                        response.message = "File Not Found";
                        response.body_len = 0;
                }
                resp->send_response(&response);
                delete [] buf;
        } else if(message->type == msg_set_parameter) {
                char *data = strdup(message->data);
                char *save_ptr;
                char *item;
                response.body_len = 0;
                item = strtok_r(data, " ", &save_ptr);
                response.code = 200;
                response.message = "OK";
                response.body_len = 0;
                if(strcmp(item, "compression") == 0) {
                        response.code = 200;
                        response.message = "OK";
                        char *compression = strtok_r(NULL, " ", &save_ptr);
                        char *quality = strtok_r(NULL, " ", &save_ptr);

                        if(strcmp(compression, "none") == 0) {
                                this->compression = std::string();
                        } else if (strcmp(compression, "JPEG") == 0) {
                                std::stringstream strstream;

                                strstream << "JPEG:" << quality;
                                this->compression  = strstream.str();
                        } else if (strcmp(compression, "DXT1") == 0) {
                                this->compression  = "RTDXT:DXT1";
                        } else if (strcmp(compression, "DXT5") == 0) {
                                this->compression  = "RTDXT:DXT5";
                        } else if (strcmp(compression, "J2K") == 0) {
                                this->compression  = "J2K";
                        } else {
                                response.code = 451;
                                response.message = "Parameter Not Understood";
                        }
                } else if(strcmp(item, "audio") == 0) {
                        response.code = 200;
                        response.message = "OK";

                        audio = string(save_ptr);
                        cerr << "Audio:" << strtok_r(NULL, " ", &save_ptr);
                } else if(strcmp(item, "quality") == 0) {
                        response.code = 200;
                        response.message = "OK";

                        char *quality = strtok_r(NULL, " ", &save_ptr);
                        char buff[20];
                        snprintf(buff, 20, "QUALITY %s", quality);

                        int len = strlen(buff);
                        write(comm_fd, &len, sizeof(len));
                        write(comm_fd, buff, len);
                } else if(strcmp(item, "module") == 0) {
                        response.code = 200;
                        response.message = "OK";

                        char buff[1024];
                        snprintf(buff, sizeof(buff), "MODULE %s", save_ptr);

                        int len = strlen(buff);
                        write(comm_fd, &len, sizeof(len));
                        write(comm_fd, buff, len);
                } else if(strcmp(item, "fps") == 0) {
                        char *fps_str = strtok_r(NULL, " ", &save_ptr);
                        /* replace decimal coma with dot */
                        if(strchr(fps_str, ',')) *strchr(fps_str, ',') = '.';
                        this->fps = atof(fps_str);
                        int len;
                        char buff[20];
                        snprintf(buff, 20, "FPS %2.2f", this->fps);

                        len = strlen(buff);
                        write(comm_fd, &len, sizeof(len));
                        write(comm_fd, buff, len);
                } else if(strcmp(item, "format") == 0) {
                        char *video_format = strtok_r(NULL, " ", &save_ptr);
                        if(strcmp(video_format, "DPX") == 0) {
                                this->video_format = "dpx";
                                this->glob_ext = "dpx";
                        }
                        else if(strcmp(video_format, "TIFF") == 0) {
                                this->video_format = "tiff";
                                this->glob_ext = "tif*";
                        }
                        else if(strcmp(video_format, "EXR") == 0) {
                                this->video_format = "exr";
                                this->glob_ext = "exr";
                        }
                        char *color_space = strtok_r(NULL, " ", &save_ptr);
                        if(color_space) {
                                this->color_space = color_space;
                        }
                } else if(strcmp(item, "use_tcp") == 0) {
                        char *use_tcp_cstr = strtok_r(NULL, " ", &save_ptr);
                        if(strcmp(use_tcp_cstr, "true") == 0)
                                this->use_tcp = true;
                } else if(strcmp(item, "loop") == 0) {
                        char *onOrOff = strtok_r(NULL, " ", &save_ptr);
                        char buff[20];
                        snprintf(buff, 20, "LOOP %s", onOrOff);

                        int len = strlen(buff);
                        write(comm_fd, &len, sizeof(len));
                        write(comm_fd, buff, len);
                } else if(strcmp(item, "speed") == 0) {
                        char *ratio = strtok_r(NULL, " ", &save_ptr);
                        char buff[20];
                        snprintf(buff, 20, "SPEED %s", ratio);

                        int len = strlen(buff);
                        write(comm_fd, &len, sizeof(len));
                        write(comm_fd, buff, len);
                } else {
                        response.code = 451;
                        response.message = "Parameter Not Understood";
                }

                free(data);

                assert(response.code != -1);
                assert(response.message != NULL);
                assert((response.body == NULL && response.body_len == 0) || 
                                (response.body != NULL && response.body_len > 0));

                resp->send_response(&response);
        }
}

void session_handler::KillUltraGridProcess()
{
        kill(pid, SIGTERM);
        pid = 0;
}

