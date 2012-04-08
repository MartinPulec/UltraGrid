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

#include "../include/sp_client.h"
#include "../include/AsyncMsgHandler.h"
#include "../include/ConnectionClosedException.h"

#include "tv.h"

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
    rtt_measurments(0)
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

//do a nonblocking connect
//  return -1 on a system call error, 0 on success
//  sa - host to connect to, filled by caller
//  sock - the socket to connect
//  timeout - how long to wait to connect
inline int
conn_nonb(struct sockaddr_in sa, int sock, int timeout)
{
    int flags = 0, error = 0, ret = 0;
    fd_set  rset, wset;
    socklen_t   len = sizeof(error);
    struct timeval  ts;

    ts.tv_sec = timeout;
    ts.tv_usec = 0;

    //clear out descriptor sets for select
    //add socket to the descriptor sets
    FD_ZERO(&rset);
    FD_SET(sock, &rset);
    wset = rset;    //structure assignment ok

    //set socket nonblocking flag
    if( (flags = fcntl(sock, F_GETFL, 0)) < 0)
        return -1;

    if(fcntl(sock, F_SETFL, flags | O_NONBLOCK) < 0)
        return -1;

    //initiate non-blocking connect
    if( (ret = connect(sock, (struct sockaddr *)&sa, 16)) < 0 )
        if (errno != EINPROGRESS)
            return -1;

    if(ret == 0)    //then connect succeeded right away
        goto done;

    //we are waiting for connect to complete now
    if( (ret = select(sock + 1, &rset, &wset, NULL, (timeout) ? &ts : NULL)) < 0)
        return -1;
    if(ret == 0){   //we had a timeout
        errno = ETIMEDOUT;
        return -1;
    }

    //we had a positivite return so a descriptor is ready
    if (FD_ISSET(sock, &rset) || FD_ISSET(sock, &wset)){
        if(getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &len) < 0)
            return -1;
    }else
        return -1;

    if(error){  //check if we had a socket error
        errno = error;
        return -1;
    }

done:
    //put socket back in blocking mode
    if(fcntl(sock, F_SETFL, flags) < 0)
        return -1;

    return 0;
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
        tv.tv_sec = 5;  /* 5 Secs Timeout */
        tv.tv_usec = 0;
        setsockopt(this->fd, SOL_SOCKET, SO_RCVTIMEO,(struct timeval *)&tv,sizeof(struct timeval));
        setsockopt(this->fd, SOL_SOCKET, SO_SNDTIMEO,(struct timeval *)&tv,sizeof(struct timeval));

        //if(connect(this->fd, res->ai_addr, res->ai_addrlen) == -1) {
        if(conn_nonb(* (struct sockaddr_in *) res->ai_addr, this->fd, 5)) {
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
    if(asyncIOHandle) {
        setAsyncNotify();
    }
}

void sp_client::setAsyncNotify()
{
        if(fcntl(this->fd, F_SETFL, fcntl(this->fd, F_GETFL) | O_ASYNC) == -1)
            perror("");
        fcntl(this->fd, F_SETOWN, getpid());
}

void sp_client::unsetAsyncNotify()
{
        if(fcntl(this->fd, F_SETFL, fcntl(this->fd, F_GETFL) & ~O_ASYNC) == -1)
            perror("");
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
        rc = recv(this->fd, this->buffer, this->buffer_len, 0);
        if(rc == -1) {
            throw std::runtime_error(std::string("Timeout"));
        }

        int i = 0;

        while(i < rc) {
            if(this->buffer[i] == '\0')
                break;

            ++i;
            if(i == rc) {
                int ret = recv(this->fd, this->buffer + rc, this->buffer_len - rc, 0);
                if (ret > 0) {
                    rc += ret;
                } else if (ret == 0) {
                    throw std::runtime_error(std::string("Connection closed"));
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
                rc += recv(this->fd, this->buffer + rc, this->buffer_len - rc, 0);
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

    rc = recv(this->fd, this->buffer, this->buffer_len, 0);

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
