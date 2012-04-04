#include <iostream>
#include <stdexcept>

#include "../include/Player.h"
#include "../client_guiMain.h"
#include "../include/UGReceiver.h"
#include "../include/VideoEntry.h"
#include "../include/Utils.h"


Player::Player() :
    receiver(0),
    speed(1.0),
    loop(false)
{
    //ctor
}

Player::~Player()
{
    //dtor
    delete receiver;
}

void Player::Init(GLView *view_, client_guiFrame *parent_, Settings *settings_)
{
    view = view_;
    parent = parent_;
    settings = settings_;

    buffer.SetGLView(view);
    receiver = new UGReceiver((const char *) "wxgl", &buffer);
}

//called upon refresh
void Player::Notify()
{
    // we just want to prefill buffer
    if(current_frame > buffer.GetUpperBound()) {
        return;
    }

    std::tr1::shared_ptr<char> res = buffer.GetFrame(current_frame);
    if(res.get()) { // not empty
        view->putframe(res);
        parent->UpdateTimer(current_frame);
    }

    buffer.DropFrames(current_frame - 20, current_frame + 20);

    current_frame ++;
}

void Player::Play(VideoEntry &item, double fps)
{
    wxString failedPart;

    try {
        wxString tmp;
        wxString hostname;
        wxString path;

        tmp = item.URL;
        tmp = tmp.Mid(wxString(L"rtp://").Len());
        hostname = tmp.BeforeFirst('/');
        path = tmp.AfterFirst('/');
        wxString video_format = item.format;

        //this->playList.pop_back();

        failedPart = wxT("connect");
        this->connection.connect_to(std::string(hostname.mb_str()), 5100);

        failedPart = wxT("format setting");
        this->connection.set_parameter(wxT("format"), video_format + wxT(" ") + item.colorSpace);
        failedPart = wxT("compression setting");
        this->connection.set_parameter(wxT("compression"), wxString(settings->GetValue(std::string("compression"), std::string("none")).c_str(), wxConvUTF8) << wxT(" ") +
                wxString(settings->GetValue(std::string("jpeg_qual"), std::string("80")).c_str(), wxConvUTF8));

        failedPart = wxT("setup");
        this->connection.setup(wxT("/") + path);
        failedPart = wxT("setting FPS");
        this->connection.set_parameter(wxT("fps"), Utils::FromCDouble(fps, 2));
        failedPart = wxT("setting loop");
        connection.set_parameter(wxT("loop"), loop ? wxT("ON") : wxT("OFF"));
        failedPart = wxT("setting speed");
        connection.set_parameter(wxT("speed"), Utils::FromCDouble(speed, 2));

        failedPart = wxT("playing");


        receiver->Accept();
        this->fps = fps;
        Start(1000/fps);
        current_frame = 0;


        connection.play();
    } catch (std::exception &e) {
        Stop();
        wxString msg = wxString::FromUTF8(e.what());
        msg += wxT("(") + failedPart + wxT(")");
        std::string newMsg =std::string(msg.mb_str());

        throw std::runtime_error(newMsg.c_str());
    }
}

void Player::Stop()
{
    wxTimer::Stop();
    receiver->Disconnect();
    buffer.Reset();
    speed = 1.0;

    //connection.teardown();
    connection.disconnect();
}

double Player::GetSpeed()
{
    return speed;
}

void Player::ProcessIncomingData()
{
    connection.ProcessIncomingData();
}

void Player::JumpAndPlay(wxString frame)
{
    connection.play(frame);
}

void Player::JumpAndPause(wxString frame)
{
    connection.pause(frame);
}

void Player::Play()
{
    connection.play();
}

void Player::Pause()
{
    connection.pause();
}

void Player::SetFPS(double fps)
{
    if(!connection.isConnected())
        return;
    connection.set_parameter(wxT("fps"), wxString::Format(wxT("%2.2f"),fps));
}

void Player::SetLoop(bool val)
{
    // preference
    loop = val;

    // shall we change it also for active connection?
    if(!connection.isConnected())
        return;

    wxString str;
    if(val == true) {
        str = wxT("ON");
    } else {
        str = wxT("OFF");
    }

    connection.set_parameter(wxT("loop"), str);
}

void Player::SetSpeed(double val)
{
    if(connection.isConnected()) {
        connection.set_parameter(wxT("speed"), Utils::FromCDouble(val, 2));
    }
}

void Player::SetMsgHandler(AsyncMsgHandler *msgHandler)
{
    connection.SetMsgHandler(msgHandler);
}
