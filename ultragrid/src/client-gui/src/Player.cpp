#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <iostream>
#include <stdexcept>
#include <string.h>
#include <wx/msgdlg.h>

#include "tv.h"

#include "../include/Player.h"
#include "../client_guiMain.h"
#include "../include/UGReceiver.h"
#include "../include/VideoEntry.h"
#include "../include/Utils.h"

#include "audio/audio.h"
#include "audio/audio_playback.h"
#include "audio/playback/sdi.h"

#define SIGN(x) ((int) (x / fabs(x)))
#define ROUND_FROM_ZERO(x) (ceil(fabs(x)) * SIGN(x))

// number of frames in every direction
#define OOB_FRAMES 128

using namespace std;

Player::Player() :
    receiver(0),
    speed(1.0),
    loop(false),
    state(sInit),
    buffer(),
    connection(),
    onFlyManager(connection, buffer),
    scheduledPlayone(false),
    display_configured(false),
    hw_display(NULL),
    audio_playback_device(NULL)
{
    //ctor
}

Player::~Player()
{
    //dtor
    display_done(this->hw_display);
    audio_playback_done(this->audio_playback_device);
    delete receiver;
}

void Player::Init(GLView *view_, client_guiFrame *parent_, Settings *settings_)
{
    view = view_;
    parent = parent_;
    settings = settings_;

    buffer.SetGLView(view);
    std::string use_tcp_str = settings->GetValue(std::string("use_tcp"), std::string("false"));
    bool use_tcp = Utils::boolFromString(use_tcp_str);

    std::string DisableGLPreviewStr = settings->GetValue(std::string("disable_gl_preview"), std::string("false"));
    bool DisableGLPreview = Utils::boolFromString(DisableGLPreviewStr);

    char *save_ptr = NULL;

    char *dev_str = strdup(settings->GetValue(std::string("hw_display"), std::string("none")).c_str());
    char *device = strtok_r(dev_str, ":", &save_ptr);

    char *config = save_ptr;


    this->hw_display = client_initialize_video_display(device,
                                                config, 0);
    free(dev_str);

    if(this->hw_display == NULL) {
        wxString msg;
        msg << wxT("Unable to initialilze video device identified '") << wxString::FromUTF8(device) << wxT("'.");
        wxMessageBox( msg, wxT("Video initialization error"), wxICON_EXCLAMATION);
        this->hw_display = client_initialize_video_display("none", NULL, 0);
    }

    string audio_playback = settings->GetValue(std::string("audio_playback"), std::string("none"));
    this->audio_playback_device = audio_playback_init(audio_playback.c_str());
    if(!this->audio_playback_device) {
        wxString msg;
        msg << wxT("Unable to initialize audio device identified '") << wxString::FromUTF8(audio_playback.c_str()) << wxT("'.");
        wxMessageBox( msg, wxT("Audio initialization error"), wxICON_EXCLAMATION);
        this->audio_playback_device = audio_playback_init("none");
    }

    if(audio_playback_does_receive_sdi(this->audio_playback_device)) {
        sdi_register_display(this->audio_playback_device, this->hw_display);
    }

    view->setHWDisplay(this->hw_display);
    view->SetGLDisplay(!DisableGLPreview);

    receiver = new UGReceiver(&buffer, this, use_tcp);
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
            return;
        }
    } else { // regular play
        std::tr1::shared_ptr<Frame> res;

        if(GetCurrentFrame() < 0 || GetCurrentFrame() >= total_frames) {
            goto update_state;
        }

        // prefatching
        if(GetCurrentFrame() > buffer.GetUpperBound() - OOB_FRAMES / 2 && GetCurrentFrame() == 0) {
            goto schedule_next;
        }

        // ZRYCHLI
        if(GetCurrentFrame() > buffer.GetUpperBound() - OOB_FRAMES / 2 // nacetli jsme vice nez pol bufferu
            && speed_status != 1
           ) {
               connection.set_parameter(wxT("speed"), Utils::FromCDouble(speed * 1.01, 2));
               speed_status = 1;
        } else if(GetCurrentFrame() < buffer.GetUpperBound() - OOB_FRAMES
            && speed_status != -1
           ) {
#if 0 // not yet implemnted, see below for temporal implementation
               connection.set_parameter(wxT("speed"), Utils::FromCDouble(speed / 1.01, 2));
               speed_status = -1;
#endif
        }

        // ZPOMAL !!!!
        if(GetCurrentFrame() < buffer.GetUpperBound() - OOB_FRAMES * 2 // nacetli jsme vice nez pol bufferu
            && !slow_down
           ) {
               slow_down = true;
               connection.pause();
        }
        if(GetCurrentFrame() > buffer.GetUpperBound() - OOB_FRAMES
           && slow_down)
        {
            slow_down = false;
            connection.play();
        }

        res = buffer.GetFrame(GetCurrentFrame());
        cout << "GetCurrentFrame "   << GetCurrentFrame() << endl;
        if(!res.get()) { // empty
            fprintf(stderr,"N");
            goto schedule_next;
        } else {
            if (!scheduledPlayone) {
                struct timeval t;

                gettimeofday(&t, NULL);
                double fps = this->fps;

                while(tv_diff(t, last_frame) < 1/fps) {
                    gettimeofday(&t, NULL);
                }

                float delay_ms = (tv_diff(t, last_frame)  - 1/fps) * 1000;
                if(delay_ms > 1) {
                    cout << "Frame delayed %.2f ms" << endl;
                }
                last_frame = t;
            }
            view->putframe(res->video, display_configured);

            struct audio_frame audio;
            audio.data = res->audio.get();
            audio.data_len = res->audio_len;
            audio.max_size = res->audio_len;

            if(audio.data_len) {
                audio_playback_put_frame(this->audio_playback_device, &audio);
            }

            parent->UpdateTimer(GetCurrentFrame() );
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
                        return;
                    }
        }
    }

schedule_next:
    if(!scheduledPlayone) {
        wxTimer::Start(1, wxTIMER_ONE_SHOT);
    } else {
        scheduledPlayone = false;
    }
}

bool Player::Playone()
{
    std::tr1::shared_ptr<Frame> res;

    res = buffer.GetFrame(GetCurrentFrame());
    if(res.get()) { // not empty
        view->putframe(res->video, display_configured);

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
            // do we really reach here? Or it is relic from looping?:P
#if 0
            int min, max;

            // not mistake - exactly one value over/underruns buffer size
            max = (GetCurrentFrame() - interval + total_frames) % total_frames;
            min = (GetCurrentFrame() + interval) % total_frames;

            buffer.DropFrames(min, max);
#endif
        } else {
            int val = GetCurrentFrame() - OOB_FRAMES;
            buffer.DropFrames(0, val);

            // Do not do this!!!! We will possibly erase frame that won't be sent anymore (currently)
            /*val = GetCurrentFrame() + OOB_FRAMES;
            buffer.DropFrames(val, total_frames);
            */
        }
    } else {
        buffer.Reset();
    }
}

void Player::Play(VideoEntry &item, double fps, int start_frame)
{
    wxString failedPart;

    codec_t transmit_codec, display_codec = RGB;

    currentVideo = &item;

    last_wanted = 0;
    speed_status = 0;
    gettimeofday(&this->last_frame, NULL);

    slow_down = false;

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

        // COMPRESSION
        string compression = settings->GetValue(std::string("compression"), std::string("none"));
        if(compression == string("JPEG")) {
            transmit_codec = JPEG;
        } else if(compression == string("J2K")) {
            transmit_codec = J2K;
            display_codec = v210;
        }
        this->connection.set_parameter(wxT("compression"), wxString(compression.c_str(), wxConvUTF8) << wxT(" ") +
                wxString(settings->GetValue(std::string("jpeg_qual"), std::string("80")).c_str(), wxConvUTF8));

        failedPart = wxT("TCP/UDP setting");
        wxString use_tcp_str = wxString(settings->GetValue(std::string("use_tcp"), std::string("false")).c_str(), wxConvUTF8);
        this->connection.set_parameter(wxT("use_tcp"), use_tcp_str);

        if(item.audioFile.Cmp(_T("none")) != 0) {
            failedPart = wxT("audio setting");
            this->connection.set_parameter(wxT("audio"), item.audioFile);
        }

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

        receiver->reinitializeDecompress(transmit_codec, display_codec);

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
    this->SetCurrentFrame(frame);
    std::cerr <<  "Current frame: " << current_frame << std::endl;
    this->connection.play(frame);

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
    wxTimer::Start(1, wxTIMER_ONE_SHOT);
}

void Player::SchedulePlay()
{
    scheduledPlayone = false;
    audio_playback_reset(this->audio_playback_device);
    wxTimer::Start(1, wxTIMER_ONE_SHOT);
}

std::tr1::shared_ptr<Frame> Player::getframe()
{
    return buffer.getframe();
}

void Player::reconfigure(int width, int height, int codec, int data_len, struct audio_desc *audio_desc)
{
    struct video_desc desc;
    size_t maxAudioDataLen = 0;

    if(audio_desc) {
        // hold up to one minute
        maxAudioDataLen = audio_desc->ch_count * audio_desc->bps * audio_desc->sample_rate;
    }

    buffer.reconfigure(width, height, codec, data_len, maxAudioDataLen);

    desc.width = width;
    desc.height = height;
    desc.color_spec = v210;
    desc.fps = currentVideo->fps;
    desc.interlacing = PROGRESSIVE;
    desc.tile_count = 1;
    //desc.colorspace;

    if(display_reconfigure(this->hw_display, desc)) {
        display_configured = true;
    } else {
        wxMessageBox( wxT("Error occured during display reconfiguration!"), wxT("Unable to reconfigure display!"), wxICON_EXCLAMATION);
        display_configured = false;
    }

    if(audio_desc) {
        if(!audio_playback_reconfigure(this->audio_playback_device, audio_desc->bps * 8, audio_desc->ch_count,
                    audio_desc->sample_rate)) {
            wxMessageBox( wxT("Error occured during audio reconfiguration!"), wxT("Unable to reconfigure audio!"), wxICON_EXCLAMATION);
        }
    }
}

void Player::putframe(std::tr1::shared_ptr<Frame> data, unsigned int frames)
{
    abort();
#if 0
    buffer.putframe(data, frames);
#endif
}
