#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <iostream>
#include <stdexcept>
#include <string.h>

#include "tv.h"

#include "../include/Player.h"
#include "../client_guiMain.h"
#include "../include/UGReceiver.h"
#include "../include/VideoEntry.h"
#include "../include/Utils.h"

#define SIGN(x) ((int) (x / fabs(x)))
#define ROUND_FROM_ZERO(x) (ceil(fabs(x)) * SIGN(x))

// number of frames in every direction
#define OOB_FRAMES 128

Player::Player() :
    receiver(0),
    speed(1.0),
    loop(false),
    state(sInit),
    buffer(),
    connection(),
    onFlyManager(connection, buffer),
    scheduledPlayone(false),
    display_configured(false)
{
    //ctor
}

Player::~Player()
{
    //dtor
    display_done(this->hw_display);
    delete receiver;
}

void Player::Init(GLView *view_, client_guiFrame *parent_, Settings *settings_)
{
    view = view_;
    parent = parent_;
    settings = settings_;

    buffer.SetGLView(view);
    std::string use_tcp_str = settings->GetValue(std::string("use_tcp"), std::string("false"));
    bool use_tcp;
    if(use_tcp_str.compare(std::string("true")) == 0) {
        use_tcp = true;
    } else {
        use_tcp = false;
    }

    char *save_ptr = NULL;

    char *dev_str = strdup(settings->GetValue(std::string("hw_display"), std::string("none")).c_str());
    char *device = strtok_r(dev_str, ":", &save_ptr);

    char *config = save_ptr;


    this->hw_display = client_initialize_video_display(device,
                                                config, 0);
    free(dev_str);

    if(this->hw_display == NULL) {
        this->hw_display = client_initialize_video_display("none", NULL, 0);
    }

    view->setHWDisplay(this->hw_display);

    receiver = new UGReceiver((const char *) "wxgl", this, use_tcp);
}

//called upon refresh
// overloaded wxTimer::Notify
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
        std::tr1::shared_ptr<char> res;

        if(GetCurrentFrame() < 0 || GetCurrentFrame() >= total_frames) {
            goto update_state;
        }
/*
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
*/

        // prefatching
        if(GetCurrentFrame() > buffer.GetUpperBound() - OOB_FRAMES / 2 && GetCurrentFrame() == 0) {
            goto schedule_next;
        }


        // ZRYCHLI
        //fprintf(stderr, "%d, %d\n",GetCurrentFrame(), buffer.GetUpperBound() - OOB_FRAMES / 2);

        if(GetCurrentFrame() > buffer.GetUpperBound() - OOB_FRAMES / 2 // nacetli jsme vice nez pol bufferu
            && speed_status != 1
           ) {
               connection.set_parameter(wxT("speed"), Utils::FromCDouble(speed * 1.01, 2));
               speed_status = 1;
        } else if(GetCurrentFrame() < buffer.GetUpperBound() - OOB_FRAMES
            && speed_status != -1
           ) {
               connection.set_parameter(wxT("speed"), Utils::FromCDouble(speed / 1.01, 2));
               speed_status = -1;
        }

        res = buffer.GetFrame(GetCurrentFrame());
        if(!res.get()) { // not empty
/*
            if(buffer.GetLastReceivedFrame() == -1) {
                goto schedule_next;
            }
            if(SIGN(speed) == 1 && buffer.GetLastReceivedFrame() > GetCurrentFrame()
               || SIGN(speed) == -1 && buffer.GetLastReceivedFrame() < GetCurrentFrame()
               ) {
                    //SetCurrentFrame(GetCurrentFrame() + SIGN(speed));
                }

            //fprintf(stderr, "%d %d %d %d\n", GetCurrentFrame(), buffer.GetLowerBound(), buffer.GetUpperBound(),  HasFrame(GetCurrentFrame()))  ;
            if(GetCurrentFrame() < 0 || GetCurrentFrame() >= total_frames) {
                goto update_state;
            }

            res = buffer.GetFrame(GetCurrentFrame());
            */
fprintf(stderr,"N");
            goto schedule_next;
        }


        if(res.get()) {
            {
                struct timeval t;

                gettimeofday(&t, NULL);
                double fps = this->fps;

                while(tv_diff(t, last_frame) < 1/fps) {
                    gettimeofday(&t, NULL);
                }
                if(tv_diff(t, last_frame) > 1/fps + 0.001) {
                    fprintf(stderr, "Frame delayed more than 1 ms\n");
                }
                last_frame = t;
            }

            view->putframe(res, display_configured);
            parent->UpdateTimer(GetCurrentFrame() );
#if 0
            struct video_frame *frame = display_get_frame(this->hw_display);
            if(frame) {
                memcpy(frame->tiles[0].data, res.get(), frame->tiles[0].data_len);
            }
            display_put_frame(this->hw_display);
#endif
        }

        DropOutOfBoundFrames(OOB_FRAMES);

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
                        //SetCurrentFrame(GetSpeed() > 0.0  ? 0 : total_frames - 1);
                        //Play();
                    }
        }
    }

schedule_next:
    wxTimer::Start(1, wxTIMER_ONE_SHOT);
}

bool Player::Playone()
{
    std::tr1::shared_ptr<char> res;

    res = buffer.GetFrame(GetCurrentFrame());
    if(res.get()) { // not empty
        view->putframe(res, display_configured);

        DropOutOfBoundFrames(OOB_FRAMES);

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
            int val = GetCurrentFrame() - OOB_FRAMES;
            buffer.DropFrames(0, val);

            val = GetCurrentFrame() + OOB_FRAMES;
            buffer.DropFrames(val, total_frames);
        }
    } else {
        buffer.Reset();
    }
}

void Player::Play(VideoEntry &item, double fps, int start_frame)
{
    wxString failedPart;

    currentVideo = &item;

    last_wanted = 0;
    speed_status = 0;
    gettimeofday(&this->last_frame, NULL);

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

        failedPart = wxT("TCP/UDP setting");
        wxString use_tcp_str = wxString(settings->GetValue(std::string("use_tcp"), std::string("false")).c_str(), wxConvUTF8);
        this->connection.set_parameter(wxT("use_tcp"), use_tcp_str);

        failedPart = wxT("setup");
        int port = this->connection.setup(wxT("/") + path);
        failedPart = wxT("setting FPS");
        this->connection.set_parameter(wxT("fps"), Utils::FromCDouble(fps, 2));
        /*failedPart = wxT("setting loop");
        connection.set_parameter(wxT("loop"), loop ? wxT("ON") : wxT("OFF"));*/
        failedPart = wxT("setting speed");
        connection.set_parameter(wxT("speed"), Utils::FromCDouble(speed, 2));

        failedPart = wxT("playing");

        receiver->Accept(hostname.mb_str(), port);
        this->fps = fps;
        SchedulePlay();

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
#ifdef DEBUG
    std::cerr << "Player::SetCurrentFrame(int frame)" << frame << std::endl;
#endif
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
    wxTimer::Start(1, wxTIMER_ONE_SHOT);
}

std::tr1::shared_ptr<char> Player::getframe()
{
    return buffer.getframe();
}

void Player::reconfigure(int width, int height, int codec, int data_len)
{
    buffer.reconfigure(width, height, codec, data_len);
    struct video_desc desc;

    desc.width = width;
    desc.height = height;
    desc.color_spec = UYVY;
    desc.fps = currentVideo->fps;
    desc.interlacing = PROGRESSIVE;
    desc.tile_count = 1;
    //desc.colorspace;

    if(display_reconfigure(this->hw_display, desc)) {
        display_configured = true;
    } else {
        display_configured = false;
    }
}

void Player::putframe(std::tr1::shared_ptr<char> data, unsigned int frames)
{
    buffer.putframe(data, frames);
}
