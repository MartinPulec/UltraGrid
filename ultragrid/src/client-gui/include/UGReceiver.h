#ifndef UGRECEIVER_H_INCLUDED
#define UGRECEIVER_H_INCLUDED

class client_guiFrame;
class GLView;
class Player;
struct state_uv;

struct display *client_initialize_video_display(const char *requested_display,
                                                char *fmt, unsigned int flags);

class UGReceiver {
    public:
        UGReceiver(const char *display, Player *player, bool use_tcp);
        ~UGReceiver();
        void Accept(const char *remote_host, int remote_port);
        void Disconnect();
    private:
        client_guiFrame *parent;
        struct state_uv *uv;
};

#endif // UGRECEIVER_H_INCLUDED
