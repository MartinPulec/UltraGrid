#ifndef SP_CLIENT_H
#define SP_CLIENT_H

#include <string>
#include <set>

class AsyncMsgHandler;

struct message {
    const char *msg;
    int len;
};

struct response {
    int code;
    const char *msg;
    int msg_len; /* IN - max msg len, OUT - actual msg len */
    char *body;
    int body_len; /* IN - max body len, OUT - actual body len */
};


class sp_client
{
    public:
        sp_client();
        void connect_to(std::string host, int port);
        void disconnect();
        bool isConnected();
        void send(struct message*, struct response *);
        void setAsyncNotify();
        void unsetAsyncNotify();
        void ProcessIncomingData();

        void SendResponse(struct response *resp);

        void SetMsgHandler(AsyncMsgHandler *msgHandler);
        virtual ~sp_client();
    protected:
    private:
        int fd;
        char *buffer;
        int buffer_len;

        AsyncMsgHandler *msgHandler;

        const std::set<std::string> valid_commands;

        void ProcessResponse(char *data, int len, struct response *response);
        void ProcessBuffer(char *data, int len);

};

#endif // SP_CLIENT_H
