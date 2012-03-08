#ifndef ASYNCMSGHANDLER_H
#define ASYNCMSGHANDLER_H

class client_guiFrame;

class AsyncMsgHandler
{
    public:
        AsyncMsgHandler(client_guiFrame *parent);
        virtual ~AsyncMsgHandler();
        void DoDisconnect();
    protected:
    private:
        client_guiFrame *parent;
};

#endif // ASYNCMSGHANDLER_H
