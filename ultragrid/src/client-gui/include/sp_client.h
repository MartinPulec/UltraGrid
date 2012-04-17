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
        sp_client(bool asyncIOHandle = false);
        void connect_to(std::string host, int port);
        void disconnect();
        bool isConnected();
        bool send(struct message*, struct response *, bool nonblock = false);
        void setAsyncNotify();
        void unsetAsyncNotify();
        void ProcessIncomingData();

        void SendResponse(struct response *resp);

        void SetMsgHandler(AsyncMsgHandler *msgHandler);

        int GetRTTMs();

        virtual ~sp_client();
    protected:
    private:
        bool asyncIOHandle;
        int fd;
        char *buffer;
        int buffer_len;

        int expectingAsyncResponse;

        AsyncMsgHandler *msgHandler;

        const std::set<std::string> valid_commands;

        bool ProcessResponse(char *data, int len, struct response *response);
        void ProcessBuffer(char *data, int len);

        int rtt;
        int rtt_measurments;

};

#endif // SP_CLIENT_H
