#ifndef CLIENTMANAGER_H
#define CLIENTMANAGER_H

#include "../include/sp_client.h"
#include <string>
#include <wx/string.h>

class AsyncMsgHandler;

class ClientManager
{
    public:
        ClientManager();
        virtual ~ClientManager();

        void connect_to(std::string host, int port);
        void disconnect();

        void set_parameter(wxString val, wxString param);
        void setup(wxString path);
        void play(int pos = -1);
        void pause(int pos = -1, int howMuch = 1);

        bool isConnected();

        void teardown();

        void ProcessIncomingData();

        void SetMsgHandler(AsyncMsgHandler *msgHandler);
    protected:
    private:
        sp_client stream_connection;
};

#endif // CLIENTMANAGER_H
