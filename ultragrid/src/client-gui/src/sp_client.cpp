#include <assert.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdexcept>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <sstream>
#include <errno.h>
#include <fcntl.h>
#include <iostream>

#include <sys/select.h>
#include <sys/time.h>


/* According to earlier standards */
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif

#include "tv.h"

#include "../include/sp_client.h"
#include "../include/AsyncMsgHandler.h"
#include "../include/ConnectionClosedException.h"
#include "../include/Utils.h"

#include "tv.h"

struct payload;
struct sp_thread_data;

struct payload {
    payload(int max_len) :
        data(new char[max_len]),
        len(0),
        next(0)
    {}
    ~payload() {
        delete [] data;
    }

    char *data;
    int len;

    struct payload *next;
};

void * sp_data_receiver_thread(void *args);

void * sp_data_receiver_thread(void *args)
{
    struct sp_thread_data *state = (struct sp_thread_data *) args;
    const int unit_len = 1024;

    struct payload *buffer;

    buffer = new payload(unit_len);

    int rc;

    while(!state->closed) {
        rc = recv(state->fd, buffer->data, unit_len, 0);

        pthread_mutex_lock(&state->lock);
        if(rc == -1) {
            if(state->closed == true) {
                pthread_mutex_unlock(&state->lock);
                goto end;
            }

            // timeout
            if(state->recv_waiting) {
                pthread_cond_signal(&state->recv_cv);
            }
        } else if (rc == 0) {
            // closed
            state->closed = true;
            if(state->recv_waiting) {
                pthread_cond_signal(&state->recv_cv);
            }
        } else {
            buffer->len = rc;
            if(state->head == 0) {
                state->head = state->end = buffer;
            } else {
                state->end->next = buffer;
                state->end = state->end->next;
            }

            buffer = new payload(unit_len);

            if(state->recv_waiting) {
                pthread_cond_signal(&state->recv_cv);
            }
        }

        if(state->msgHandler) {
            if(state->head || state->closed) {
                state->parent->ProcessIncomingData();
            }
        }
        pthread_mutex_unlock(&state->lock);
    }

end:
    state->release();
    return NULL;
}

static std::string valid_commands_str[] = {
    std::string("TEARDOWN")
};

sp_client::sp_client(bool aioH) :
    valid_commands(valid_commands_str, valid_commands_str + sizeof(valid_commands_str) / sizeof(std::string)),
    fd(-1),
    msgHandler(0ul),
    asyncIOHandle(aioH),
    expectingAsyncResponse(0),
    rtt(0),
    rtt_measurments(0),
    thread_data(0)
{
    buffer_len = 1024*1024;
    buffer = new char[buffer_len];
}

sp_client::~sp_client()
{
    if(this->fd != -1) {
        disconnect();
    }

    delete [] buffer;
}

void sp_client::connect_to(std::string host, int port)
{
    struct addrinfo hints, *res, *res0;
    int err;

    expectingAsyncResponse = 0;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[6];
    snprintf(port_str, 5, "%u", port);
    err = getaddrinfo(host.c_str(), port_str, &hints, &res0);

    if(err) {
        throw std::runtime_error(std::string("getaddrinfo: ") + gai_strerror(err) + " (" + host + ")");
    }

    this->fd = -1;

    std::string what;

    for (res = res0; res; res = res->ai_next) {
        this->fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if  (this->fd < 0) {
            what = std::string("socket failed: ") + strerror(errno);
            continue;
        }

        struct timeval tv;
        tv.tv_sec = 0;  /* 5 Secs Timeout */
        tv.tv_usec = 200 * 1000;
        setsockopt(this->fd, SOL_SOCKET, SO_RCVTIMEO,(struct timeval *)&tv,sizeof(struct timeval));
        setsockopt(this->fd, SOL_SOCKET, SO_SNDTIMEO,(struct timeval *)&tv,sizeof(struct timeval));

        //if(connect(this->fd, res->ai_addr, res->ai_addrlen) == -1) {
        if(Utils::conn_nonb(* (struct sockaddr_in *) res->ai_addr, this->fd, 5)) {
            this->fd = -1;
            what = std::string("connect failed: ") + strerror(errno);
            continue;
        }

        break; /* okay we got one */
    }

    if(this->fd < 0 ) {
        throw std::runtime_error(what);
    }

    freeaddrinfo(res0);

    pthread_t thread_id;

    if(this->thread_data) {
        this->thread_data->release();
        this->thread_data = 0;
    }

    this->thread_data = new struct sp_thread_data(this, this->fd);

    this->thread_data->acquire(); //for thread

    pthread_create(&thread_id, NULL, sp_data_receiver_thread, (void *) this->thread_data);
    pthread_detach(thread_id);

    if(asyncIOHandle) {
        setAsyncNotify();
    }
}

void sp_client::setAsyncNotify()
{
    if(!this->thread_data)
        return;

    pthread_mutex_lock(&this->thread_data->lock);
    this->thread_data->msgHandler = this->msgHandler;
    if(this->thread_data->head) {
        ProcessIncomingData();
    }
    pthread_mutex_unlock(&this->thread_data->lock);
}

void sp_client::unsetAsyncNotify()
{
    if(!this->thread_data)
        return;

    pthread_mutex_lock(&this->thread_data->lock);
    this->thread_data->msgHandler = 0;
    if(this->thread_data->head) {
        ProcessIncomingData();
    }
    pthread_mutex_unlock(&this->thread_data->lock);
}

void sp_client::disconnect()
{
    rtt = 0;
    rtt_measurments = 0;

    if(asyncIOHandle) {
        unsetAsyncNotify();
    }

    if(this->fd != -1) {
        close(this->fd);
        this->thread_data->closed = true;
        this->thread_data->release();
        this->thread_data = 0;
        this->fd = -1;
    }
}

bool sp_client::send(struct message *message, struct response *response, bool nonblock)
{
    int rc;
    bool res;

    struct timeval t0, t1;

    if(asyncIOHandle && !nonblock) {
        unsetAsyncNotify();
    }

    char *buffer = (char *) malloc(message->len + 1);
    strncpy(buffer, message->msg, message->len);
    buffer[message->len] = '\0';

    if(!nonblock) {
        gettimeofday(&t0, NULL);
    }

    int ret = write(this->fd, buffer, message->len + 1);
#ifdef DEBUG
    std::cerr << "Message : \"";
    write(2, buffer, message->len + 1);
    std::cerr << "\"" << std::endl;
#endif
    if(ret == -1) {
        throw ConnectionClosedException();
    }

    assert(ret == message->len + 1);
    free(buffer);

    if(nonblock) {
        expectingAsyncResponse++;
    } else {
        rc = recv_data(this->buffer, this->buffer_len, 3);
        if(rc == -1) {
            throw std::runtime_error(std::string("Timeout"));
        } else if (rc == 0) {
            throw ConnectionClosedException();
        }

        int i = 0;

        while(i < rc) {
            if(this->buffer[i] == '\0')
                break;

            ++i;
            if(i == rc) {
                int ret = recv_data(this->buffer + rc, this->buffer_len - rc, 3);
                if (ret > 0) {
                    rc += ret;
                } else if (ret == 0) {
                    throw ConnectionClosedException();
                } else {
                    throw std::runtime_error(std::string("Timeout"));
                }
            }
        }

        /* i is from now index of '\0' */

        const char *ptr = strstr(this->buffer, "Content-Length:");
        if(!ptr) {
            ProcessResponse(this->buffer, i + 1, response);
        } else {
            ptr += sizeof("Content-Length:");
            int body_len = atoi(ptr);
            while(*ptr != '\r' && ptr < this->buffer + rc)
                ++ptr;
            ptr += 4; /* \r\n\r\n */
            int header_len = ptr - this->buffer;
            int total_len = header_len + body_len;
            while(rc < total_len) {
                ssize_t ret;
                ret = recv_data(this->buffer + rc, this->buffer_len - rc, 3);
                if(ret == 0) {
                    throw std::runtime_error(std::string("Timeout"));
                } else if(ret == 1) {
                    throw ConnectionClosedException();
                }

                rc += ret;
            }
            res = ProcessResponse(this->buffer, total_len, response);
        }
    }

    if(!nonblock) {
        gettimeofday(&t1, NULL);
        int new_rtt = (rtt_measurments * rtt + tv_diff_usec(t1, t0) / 1000) / (rtt_measurments + 1);
        rtt = new_rtt;
        rtt_measurments++;
    }

    // TODO: doresit zacatek dalsiho packetu

    if(asyncIOHandle && !nonblock) {
        setAsyncNotify();
    }

    return res;
}

bool sp_client::ProcessResponse(char *data, int len, struct response *response)
{
    char *ptr;

    if(len <= 0 || !isdigit(data[0])) {
        return false;
    }

    ptr = data;
    response->code = atoi(ptr);

    while(!isspace(*ptr) && ptr < this->buffer + len)
        ++ptr;
    while(isspace(*ptr) && ptr < this->buffer + len)
        ++ptr;
    response->msg = ptr;
    while((*ptr != '\n' && *ptr != '\r') && ptr < this->buffer + len)
        ++ptr;
    response->msg_len = ptr - response->msg;

    while((*ptr == '\n' || *ptr == '\r') && ptr < this->buffer + len)
        ++ptr;
    if(strncasecmp("Content-Length:", ptr, strlen("Content-Length:")) != 0)
        response->body_len = 0;
    else {
        ptr += sizeof("Content-Length:");
        response->body_len = atoi(ptr);
        while(*ptr != '\r' && ptr < this->buffer + len)
            ++ptr;
        ptr += 4; /* \r\n\r\n */

        response->body = ptr;
    }

    return true;
}

void sp_client::ProcessIncomingData()
{
    int rc;
    int flags;

    //set socket nonblocking flag
    if( (flags = fcntl(this->fd, F_GETFL, 0)) < 0)
        return;

    if(fcntl(this->fd, F_SETFL, flags | O_NONBLOCK) < 0)
        return;

    rc = recv_data(this->buffer, this->buffer_len, 3);

    //put socket back in blocking mode
    if(fcntl(this->fd, F_SETFL, flags) < 0)
        return;

    if(rc == -1) {
        std::cerr << "Timeout" << std::endl;
        return;
    }

    if(rc == 0) {
        msgHandler->DoDisconnect();
    }

    ProcessBuffer(this->buffer, rc);
}

void sp_client::ProcessBuffer(char *data, int len)
{
    struct response response;
    int rc = len;
    char *ptr;
    std::string command;
    int start, pos;


    ptr = this->buffer;

    for(start = 0; start < rc; ++start) {
            if(!isspace(this->buffer[start])) {
                    break;
            }
    }

    for(pos = start; pos < rc; ++pos) {
            if(isspace(this->buffer[pos])) {
                    break;
            }
    }

    if(start == rc)
    {
        std::cerr << "Misspelled message received." << std::endl;
        return;
    }

    command.assign(this->buffer + start, pos - start);

    std::cerr << "Command: " << command << " received." << std::endl;

    if(isdigit(command[0])) {
        expectingAsyncResponse--;
        return;
    }


    if(!isalpha(command[0])) {
        std::cerr << "Misspelled message received." << std::endl;
        return;
    }

    if(valid_commands.find(command) == valid_commands.end()) {
        response.code = 451;
        response.msg = "Parameter Not Understood";
        response.msg_len = strlen(response.msg);
        response.body_len = 0;
        SendResponse(&response);
    } else if(command == std::string("TEARDOWN")) {
        response.code = 200;
        response.msg = "OK";
        response.msg_len = strlen(response.msg);
        response.body_len = 0;
        if(msgHandler) {
            SendResponse(&response);
            msgHandler->DoDisconnect();
        }
    }

}

void sp_client::SendResponse(struct response *response)
{
    const int hdr_size = 1000;
    char buff[hdr_size];
    int len;

    len = snprintf(buff, hdr_size, "%u %s\r\n", (unsigned int) response->code, response->msg);
    if(response->body_len > 0) {
            len += snprintf(buff + len, hdr_size - len, "Content-Length: %u\r\n", (unsigned int) response->body_len);
    }

    len += snprintf(buff + len, hdr_size - len, "\r\n");
    if(len == hdr_size - 1) {
        std::cerr << "Warning: Possible snding buffer underflow (void sp_client::SendResponse(struct response *response))!!! " << std::endl;
        return;
    }
    buff[len] = '\0';
    len++;

    write(fd, buff, len);

    if(response->body_len)
        write(fd, response->body, response->body_len);
}

void sp_client::SetMsgHandler(AsyncMsgHandler *msgHandler)
{
    this->msgHandler = msgHandler;
}

bool sp_client::isConnected()
{
    return this->fd != -1;
}

int sp_client::GetRTTMs()
{
    if(rtt_measurments > 0) {
        return rtt;
    } else {
        return -1;
    }
}

int sp_client::recv_data(char *buffer, int len, int timeout_sec)
{
    int total = 0;

    struct timeval t0;

    gettimeofday(&t0, NULL);

    pthread_mutex_lock(&this->thread_data->lock);
    while(this->thread_data->head == 0 && !this->thread_data->closed) {
        // check timeout
        struct timeval t;
        gettimeofday(&t, NULL);
        if(tv_diff_usec(t, t0) > timeout_sec * 1000 * 1000) {
            pthread_mutex_unlock(&this->thread_data->lock);
            return -1;
        }

        this->thread_data->recv_waiting = true;
        pthread_cond_wait(&this->thread_data->recv_cv, &this->thread_data->lock);
        this->thread_data->recv_waiting = false;
    }

    if(this->thread_data->closed) {
        pthread_mutex_unlock(&this->thread_data->lock);
        return 0;
    }

    do {
        if(this->thread_data->head->len < len) {
            memcpy(buffer, this->thread_data->head->data, this->thread_data->head->len);

            len -= this->thread_data->head->len;
            buffer += this->thread_data->head->len;
            total += this->thread_data->head->len;

            struct payload * tmp = this->thread_data->head;

            this->thread_data->head = this->thread_data->head->next;

            delete tmp;
        } else {
            memcpy(buffer, this->thread_data->head->data, len);
            memmove(this->thread_data->head->data, this->thread_data->head->data + len, this->thread_data->head->len - len);
            this->thread_data->head->len = len;

            total += len;

            break;
        }
    } while (this->thread_data->head != 0);

    if(this->thread_data->head == 0)
        this->thread_data->end = 0;

    pthread_mutex_unlock(&this->thread_data->lock);

    return total;
}

