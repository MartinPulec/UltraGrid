#include "../include/ClientManager.h"

#include <stdexcept>

ClientManager::ClientManager() :
    stream_connection(true)
{
    //ctor
}

ClientManager::~ClientManager()
{
    //dtor
}

void ClientManager::SetMsgHandler(AsyncMsgHandler *msgHandler)
{
    stream_connection.SetMsgHandler(msgHandler);
}

void ClientManager::connect_to(std::string host, int port)
{
    stream_connection.connect_to(host, port);
}

void ClientManager::disconnect()
{
    stream_connection.disconnect();
}

void ClientManager::set_parameter(wxString val, wxString param)
{
    wxString msgstr;
    struct message msg;
    struct response resp;
    wxCharBuffer buf;

    msgstr = L"";
    msgstr << wxT("SET_PARAMETER ") << val << wxT(" ") << param;
    buf = msgstr.mb_str();
    msg.msg = buf.data();
    msg.len = msgstr.Len();

    this->stream_connection.send(&msg, &resp);

    if(resp.code != 200) {
        wxString msg;
        msg << wxT("Unable to set video format: ") << resp.code << L" " << wxString::FromUTF8((resp.msg));
        throw std::runtime_error(std::string(msg.mb_str()));
    }
}

void ClientManager::setup(wxString path)
{
    wxString msgstr;
    struct message msg;
    struct response resp;
    wxCharBuffer buf;

    msgstr = L"";
    msgstr << wxT("SETUP ") << path;
    buf = msgstr.mb_str();
    msg.msg = buf.data();
    msg.len = msgstr.Len();

    // TODO: handle TMOUT
    this->stream_connection.send(&msg, &resp);

    if(resp.code != 201) {
        wxString msg;
        msg << wxT("Media setup error: ") << resp.code << L" " << wxString::FromUTF8((resp.msg));
        throw std::runtime_error(std::string(msg.mb_str()));
    }
}

void ClientManager::play(int pos)
{
    wxString msgstr;
    struct message msg;
    struct response resp;
    wxCharBuffer buf;

    msgstr = L"";
    msgstr << wxT("PLAY");
    if(pos != -1) {
        msgstr << wxT(" ") << pos;
    }
    buf = msgstr.mb_str();
    msg.msg = buf.data();
    msg.len = msgstr.Len();

    // TODO: handle TMOUT
    this->stream_connection.send(&msg, &resp);

    if(resp.code != 200) {
        wxString msg;
        msg << wxT("Error starting stream: ") << resp.code << L" " << wxString::FromUTF8((resp.msg));
        throw std::runtime_error(std::string(msg.mb_str()));
    }
}

void ClientManager::pause(int pos, int howMuch)
{
    wxString msgstr;
    struct message msg;
    struct response resp;
    wxCharBuffer buf;

    msgstr = L"";
    msgstr << wxT("PAUSE");
    if(pos != -1) {
        msgstr << wxT(" ") << pos << wxT(" ") << howMuch;
    }
    buf = msgstr.mb_str();
    msg.msg = buf.data();
    msg.len = msgstr.Len();

    // TODO: handle TMOUT
    this->stream_connection.send(&msg, &resp);

    if(resp.code != 200) {
        wxString msg;
        msg << wxT("Error pausing stream: ") << resp.code << L" " << wxString::FromUTF8((resp.msg));
        throw std::runtime_error(std::string(msg.mb_str()));
    }
}

void ClientManager::teardown()
{
    wxString msgstr;
    struct message msg;
    struct response resp;

    msg.msg = "TEARDOWN";
    msg.len = strlen(msg.msg);

    // TODO: handle TMOUT
    this->stream_connection.send(&msg, &resp);

    if(resp.code != 200) {
        wxString msg;
        msg << wxT("Error stopping media: ") << resp.code << L" " << wxString::FromUTF8((resp.msg));
        throw std::runtime_error(std::string(msg.mb_str()));
    }
}

void ClientManager::ProcessIncomingData()
{
    stream_connection.ProcessIncomingData();
}

bool ClientManager::isConnected()
{
    return stream_connection.isConnected();
}
