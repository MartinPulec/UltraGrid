#include "../include/AsyncMsgHandler.h"
#include "../client_guiMain.h"

#include <wx/event.h>

AsyncMsgHandler::AsyncMsgHandler(client_guiFrame *p) :
    parent(p)
{
    //ctor
}

AsyncMsgHandler::~AsyncMsgHandler()
{
    //dtor
}

void AsyncMsgHandler::DoDisconnect()
{
    wxCommandEvent event(wxEVT_DISCONNECT);
    wxPostEvent(parent, event);
}
