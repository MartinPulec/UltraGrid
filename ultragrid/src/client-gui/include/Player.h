#ifndef PLAYER_H
#define PLAYER_H

#include <wx/string.h>
#include <wx/timer.h>
#include "../include/VideoBuffer.h"
#include "../include/ClientManager.h"
#include "../include/Settings.h"

class GLView;
class VideoBuffer;
class ClientManager;
class client_guiFrame;
class UGReceiver;
class VideoEntry;
class AsyncMsgHandler;


class Player : public wxTimer
{
    public:
        Player();
        virtual ~Player();

        void Init(GLView *view, client_guiFrame *parent, Settings *settings);

        void Notify();

        void Play(VideoEntry &item, double fps);
        void Stop();

        double GetSpeed();
        void ProcessIncomingData();

        void JumpAndPlay(wxString frame);
        void JumpAndPause(wxString frame);

        void Play();
        void Pause();

        void SetFPS(double fps);
        void SetLoop(bool val);

        void SetSpeed(double val);

        void SetMsgHandler(AsyncMsgHandler *msgHandler);

    protected:
    private:
        GLView *view;
        VideoBuffer buffer;
        client_guiFrame *parent;
        UGReceiver *receiver;
        Settings *settings;

        ClientManager connection;

        double fps;
        double speed;
        bool loop;
        int current_frame;
};

#endif // PLAYER_H
