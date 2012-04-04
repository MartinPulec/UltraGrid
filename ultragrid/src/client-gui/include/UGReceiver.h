#ifndef UGRECEIVER_H_INCLUDED
#define UGRECEIVER_H_INCLUDED

class client_guiFrame;
class GLView;
class VideoBuffer;
struct state_uv;

class UGReceiver {
    public:
        UGReceiver(const char *display, VideoBuffer *gl);
        ~UGReceiver();
        void Accept();
        void Disconnect();
    private:
        client_guiFrame *parent;
        struct state_uv *uv;
};

#endif // UGRECEIVER_H_INCLUDED
