#ifndef PLAYER_H
#define PLAYER_H

#include <wx/string.h>
#include <wx/timer.h>
#include "../include/VideoBuffer.h"
#include "../include/ClientManager.h"
#include "../include/Settings.h"
#include "../include/VideoBufferOnFlyManager.h"

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
        void ProcessIncomingData();

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

        ClientManager connection;

        VideoBufferOnFlyManager onFlyManager;

        double fps;
        double speed;
        int total_frames;
        bool loop;
        int current_frame;
        enum playerState state;
        bool scheduledPlayone;
};

#endif // PLAYER_H
