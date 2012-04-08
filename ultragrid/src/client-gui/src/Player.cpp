#include <iostream>
#include <stdexcept>

#include "../include/Player.h"
#include "../client_guiMain.h"
#include "../include/UGReceiver.h"
#include "../include/VideoEntry.h"
#include "../include/Utils.h"

#define SIGN(x) ((int) (x / fabs(x)))
#define ROUND_FROM_ZERO(x) (ceil(fabs(x)) * SIGN(x))


Player::Player() :
    receiver(0),
    speed(1.0),
    loop(false),
    state(sInit),
    buffer(),
    connection(),
    onFlyManager(connection, buffer)
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
    if(scheduledPlayone) {
        if(!Playone()) {
            if(onFlyManager.LastRequestIsDue(this->fps)) {
                onFlyManager.RequestAdditionalBuffers(current_frame, total_frames, SIGN(this->speed));
            }
            ScheduleOneFrame();
        } else {
            scheduledPlayone = false;
        }
    } else { // regular play
        // there is no frame beyond our buffer
        if(speed > 0.0) {
            if(GetCurrentFrame() >= buffer.GetUpperBound()) {
                return;
            }
        } else {
            if(GetCurrentFrame() <= buffer.GetLowerBound() )
            {
                return;
            }
        }

        std::tr1::shared_ptr<char> res;

        res = buffer.GetFrame(GetCurrentFrame());
        while(!res.get()) { // not empty
            SetCurrentFrame(GetCurrentFrame() + ROUND_FROM_ZERO(speed));

            if(GetCurrentFrame() < 0 || GetCurrentFrame() >= total_frames) {
                goto update_state;
            }

            res = buffer.GetFrame(GetCurrentFrame());
        }

        if(res.get()) {
            view->putframe(res);
            parent->UpdateTimer(GetCurrentFrame() );
        }

        DropOutOfBoundFrames(20);

        SetCurrentFrame(GetCurrentFrame() + SIGN(speed));


    update_state:

        if(((GetSpeed() > 0.0 && GetCurrentFrame() >= total_frames) ||
               (GetSpeed() < 0.0 && GetCurrentFrame() < 0))) {
                   if(!loop) {
                        parent->DoPause();
                    } else {
                        Pause();
                        DropOutOfBoundFrames();
                        JumpAndPlay(GetSpeed() > 0.0  ? 0 : total_frames - 1);
                        //Play();
                    }
        }
    }
}

bool Player::Playone()
{
    std::tr1::shared_ptr<char> res;

    res = buffer.GetFrame(GetCurrentFrame());
    if(res.get()) { // not empty
        view->putframe(res);

        DropOutOfBoundFrames(20);

        parent->UpdateTimer(GetCurrentFrame() );

        return true;
    } else { // not in buffer
        return false;
    }
}

void Player::DropOutOfBoundFrames(int interval)
{
    if (interval) {
        if(total_frames < 2 * interval)
            return;

        if(GetCurrentFrame() < interval || GetCurrentFrame() + interval > total_frames - 1) {
            int min, max;

            // not mistake - exactly one value over/underruns buffer size
            max = (GetCurrentFrame() - interval + total_frames) % total_frames;
            min = (GetCurrentFrame() + interval) % total_frames;

            buffer.DropFrames(min, max);
        } else {
            int val = GetCurrentFrame() - 20;
            buffer.DropFrames(0, val);

            val = GetCurrentFrame() + 20;
            buffer.DropFrames(val, total_frames);
        }
    } else {
        buffer.DropFrames(0, total_frames);
    }
}

void Player::Play(VideoEntry &item, double fps, int start_frame)
{
    wxString failedPart;

    try {
        wxString tmp;
        wxString hostname;
        wxString path;

        total_frames = item.total_frames;
        SetCurrentFrame(start_frame);

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
        /*failedPart = wxT("setting loop");
        connection.set_parameter(wxT("loop"), loop ? wxT("ON") : wxT("OFF"));*/
        failedPart = wxT("setting speed");
        connection.set_parameter(wxT("speed"), Utils::FromCDouble(speed, 2));

        failedPart = wxT("playing");


        receiver->Accept();
        this->fps = fps;
        Start(1000/fps);

        connection.play(start_frame);
        //connection.play();
    } catch (std::exception &e) {
        StopPlayback();
        wxString msg = wxString::FromUTF8(e.what());
        msg += wxT("(") + failedPart + wxT(")");
        std::string newMsg =std::string(msg.mb_str());

        throw std::runtime_error(newMsg.c_str());
    }
}

void Player::StopPlayback()
{
    wxTimer::Stop();
    receiver->Disconnect();
    buffer.Reset();

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

void Player::JumpAndPlay(int frame)
{
    current_frame = frame;
    std::cerr <<  "Current frame: " << current_frame << std::endl;
    connection.play(frame);
    wxTimer::Start();
}

void Player::JumpAndPause(int frame)
{
    current_frame = frame;
    std::cerr <<  "Current frame: " << current_frame << std::endl;

    if(onFlyManager.LastRequestIsDue(this->fps)) {
        onFlyManager.RequestAdditionalBuffers(current_frame, total_frames, SIGN(this->speed));
    }

    if(!Playone()) {
        ScheduleOneFrame();
    }
}

void Player::Play()
{
    SchedulePlay();
    connection.play();
}

void Player::Pause()
{
    wxTimer::Stop();
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

    //connection.set_parameter(wxT("loop"), str);
}

void Player::SetSpeed(double val)
{
    speed = val;
    if(connection.isConnected()) {
        connection.set_parameter(wxT("speed"), Utils::FromCDouble(val, 2));
    }
}

void Player::SetMsgHandler(AsyncMsgHandler *msgHandler)
{
    connection.SetMsgHandler(msgHandler);
}

void Player::ChangeState(enum playerState newState)
{
    state = newState;
}

enum playerState Player::GetState()
{
    return state;
}

void Player::SetCurrentFrame(int frame)
{
    current_frame = frame;
    std::cerr << "Player::SetCurrentFrame(int frame)" << frame << std::endl;
}

int Player::GetCurrentFrame()
{
    return current_frame;
}

int Player::GetTotalFrames()
{
    return total_frames;
}

void Player::ScheduleOneFrame()
{
    scheduledPlayone = true;
    wxTimer::Start(-1, wxTIMER_ONE_SHOT);
}

void Player::SchedulePlay()
{
    scheduledPlayone = false;
    wxTimer::Start(-1, wxTIMER_CONTINUOUS);
}
