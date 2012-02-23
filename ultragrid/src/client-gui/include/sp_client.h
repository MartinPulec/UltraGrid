#ifndef SP_CLIENT_H
#define SP_CLIENT_H

#include <stdint.h>
#include <string>

struct message {
    const char *msg;
    int len;
};

struct response {
    int code;
    char *msg;
    int msg_len; /* IN - max msg len, OUT - actual msg len */
    char *body;
    int body_len; /* IN - max body len, OUT - actual body len */
};


class sp_client
{
    public:
        sp_client();
        void connect_to(std::string host, uint16_t port);
        void disconnect();
        void send(struct message*, struct response *);
        virtual ~sp_client();
    protected:
    private:
        int fd;
        char *buffer;
        int buffer_len;
};

#endif // SP_CLIENT_H
