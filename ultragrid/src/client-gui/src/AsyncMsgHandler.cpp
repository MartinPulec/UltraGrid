#include "../include/AsyncMsgHandler.h"
#include "../client_guiMain.h"

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
    parent->DoDisconnect();
}
