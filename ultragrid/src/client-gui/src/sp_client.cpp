#include "../include/sp_client.h"
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

sp_client::sp_client()
{
    this->fd = -1;
}

sp_client::~sp_client()
{

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

void sp_client::connect_to(std::string host, uint16_t port)
{
    buffer_len = 1024*1024;
    buffer = new char[buffer_len];
    struct addrinfo hints, *res, *res0;
    int err;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[6];
    snprintf(port_str, 5, "%u", port);
    err = getaddrinfo(host.c_str(), port_str, &hints, &res0);

    if(err) {
        throw std::runtime_error(std::string("getaddrinfo: ") + gai_strerror(err));
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
        }

        break; /* okay we got one */
    }

    if(this->fd < 0 ) {
        throw std::runtime_error(what);
    }

    freeaddrinfo(res0);
}

void sp_client::disconnect()
{
    if(this->fd != -1) {
        close(this->fd);
    }
}

void sp_client::send(struct message *message, struct response *response)
{
    int rc;
    char *ptr;
    write(this->fd, message->msg, message->len);
    rc = recv(this->fd, this->buffer, this->buffer_len, 0);
    if(rc == -1)
        throw std::runtime_error(std::string("Timeout"));

    ptr = this->buffer;
    response->code = atoi(ptr);

    while(!isspace(*ptr) && ptr < this->buffer + rc)
        ++ptr;
    while(isspace(*ptr) && ptr < this->buffer + rc)
        ++ptr;
    response->msg = ptr;
    while((*ptr != '\n' && *ptr != '\r') && ptr < this->buffer + rc)
        ++ptr;
    response->msg_len = ptr - response->msg;

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
        response->body = ptr;
        if(response->body + response->body_len != this->buffer + rc) {
            std::string msg;
            std::stringstream str;

            str << "Immature end of packet: " << rc << " B (expected " << response->body + response->body_len - this->buffer << " B)";
            msg = str.str();
            throw std::runtime_error(msg);
        }
    }
}
