#ifndef SP_CLIENT_H
#define SP_CLIENT_H

#include <pthread.h>
#include <tr1/memory>
#include <string>
#include <set>

class AsyncMsgHandler;

void * sp_data_receiver_thread(void *args);
struct payload;


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


struct sp_thread_data {
    sp_thread_data(class sp_client *p, int filedescriptor) :
        head(0),
        end(0),
        closed(false),
        fd(filedescriptor),
        ref(1),
        recv_waiting(false),
        msgHandler(0),
        parent(p)
    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);

        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE_NP);

        pthread_mutex_init(&lock, &attr);
        pthread_mutexattr_destroy(&attr);
        pthread_cond_init(&recv_cv, NULL);
    }

    ~sp_thread_data() {
        pthread_mutex_destroy(&lock);
        pthread_cond_destroy(&recv_cv);
    }

    void acquire()
    {
        pthread_mutex_lock(&lock);
        ++ref;
        pthread_mutex_unlock(&lock);
    }

    void release()
    {
        pthread_mutex_lock(&lock);
        --ref;
        pthread_mutex_unlock(&lock);
        if(ref == 0)
            delete this;
    }

    AsyncMsgHandler *msgHandler;

    // for communication with thread
    pthread_mutex_t lock;
    pthread_cond_t recv_cv;
    volatile bool recv_waiting;
    struct payload * volatile head;
    struct payload * volatile end;
    class sp_client *parent;

    int fd;
    int ref;

    volatile bool closed;
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
        int recv_data(char *buffer, int len, int timeout_sec);


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

        friend void * sp_data_receiver_thread(void *args);

        struct sp_thread_data *thread_data;

};

#endif // SP_CLIENT_H
