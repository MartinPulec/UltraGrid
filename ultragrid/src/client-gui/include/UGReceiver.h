#ifndef UGRECEIVER_H_INCLUDED
#define UGRECEIVER_H_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "video.h"

class client_guiFrame;
class GLView;
class Player;
struct state_uv;

struct display *client_initialize_video_display(const char *requested_display,
                                                char *fmt, unsigned int flags);
struct video_desc;
struct audio_desc;
struct state_decompress;

class UGReceiver {
    public:
        UGReceiver(const char *display, Player *player, bool use_tcp);
        ~UGReceiver();
        void Accept(const char *remote_host, int remote_port);
        void Disconnect();

        /**
        * @param[out] audio_lenght Contains 0 if there is no audio, > 0 otherwise
        * @retval true if header parsed successfully
        * @retval false if not
        */
        static bool ParseHeader(uint32_t *hdr, size_t hdr_len,
                             struct video_desc *video_desc, size_t *video_length,
                             struct audio_desc *audio_desc, size_t *audio_length);
        static void Reconfigure(struct state_uv *uv, struct video_desc);
    private:
        client_guiFrame *parent;
        struct state_uv *uv;
};

#endif // UGRECEIVER_H_INCLUDED
