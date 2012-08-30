#ifndef PLAYER_H
#define PLAYER_H

#include <sys/time.h>

#include <wx/string.h>
#include <wx/timer.h>

#include "../include/ClientManager.h"
#include "../include/Settings.h"
#include "../include/Observer.h"
#include "../include/VideoBuffer.h"
#include "../include/VideoBufferOnFlyManager.h"

#include "video_display.h"

class GLView;
class VideoBuffer;
class ClientManager;
class client_guiFrame;
class UGReceiver;
class VideoEntry;
class AsyncMsgHandler;

enum playerState {
    sInit,
    sReady,
    sPlaying
};


class Player : public wxTimer
{
    public:
        Player();
        virtual ~Player();

        void Init(GLView *view, client_guiFrame *parent, Settings *settings);

        void Notify();

        void Play(VideoEntry &item, double fps, int start_frame);
        void StopPlayback();

        double GetSpeed();

        void JumpAndPlay(int frame);
        void JumpAndPause(int frame);

        void Play();
        void Pause();

        void SetFPS(double fps);
        void SetLoop(bool val);

        void SetSpeed(double val);

        void SetMsgHandler(AsyncMsgHandler *msgHandler);
        void ChangeState(enum playerState newState);
        enum playerState GetState();

        int GetTotalFrames();

        int GetCurrentFrame();

        std::tr1::shared_ptr<char> getframe();
        void reconfigure(int width, int height, int codec, int data_len);
        void putframe(std::tr1::shared_ptr<char> data, unsigned int frames);

    protected:
    private:
        void SetCurrentFrame(int frame);

        bool Playone();

        void RequestAdditionalBuffers();

        void ScheduleOneFrame();
        void SchedulePlay();

        void DropOutOfBoundFrames(int interval = 0);

        GLView *view;
        VideoBuffer buffer;
        client_guiFrame *parent;
        UGReceiver *receiver;
        Settings *settings;

        struct display  *hw_display;
        bool display_configured;

        ClientManager connection;

        VideoBufferOnFlyManager onFlyManager;

        double fps;
        VideoEntry *currentVideo;
        double speed;
        int total_frames;
        bool loop;
        int current_frame;
        enum playerState state;
        bool scheduledPlayone;
        struct timeval last_frame;
        int last_wanted;

        int speed_status;
};

#endif // PLAYER_H
