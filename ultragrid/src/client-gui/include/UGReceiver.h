#ifndef UGRECEIVER_H_INCLUDED
#define UGRECEIVER_H_INCLUDED

class client_guiFrame;
class GLView;
struct state_uv;

class UGReceiver {
    public:
        UGReceiver(client_guiFrame * const p, const char *display, GLView *gl);
    private:
        client_guiFrame *parent;
        struct state_uv *uv;
};

#endif // UGRECEIVER_H_INCLUDED
