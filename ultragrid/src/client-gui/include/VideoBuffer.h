#ifndef VIDEOBUFFER_H
#define VIDEOBUFFER_H

#include <map>
#include <tr1/memory>

#include <pthread.h>

#include "audio/audio.h" // audio_desc
#include "video.h" // video_desc

#include "../include/Observable.h"


class GLView;

struct Frame {
    Frame(size_t videoLen, size_t audioLen);

    std::tr1::shared_ptr<char> video;
    std::tr1::shared_ptr<char> audio;
    size_t audio_len;
    size_t video_len;
    struct audio_desc audio_desc;
    struct video_desc video_desc;
};

#if 0
struct Frame {
    struct audio_desc audio_desc;
    struct video_desc video_desc;
    char *audio;
    char *video;
    size_t audio_len;
    size_t video_len;

    void alloc() {
        this->audio = new char[audio_len];
        this->video = new char[video_len];
    }
};
#endif

typedef std::tr1::shared_ptr<Frame> shared_frame;

class VideoBuffer: public Observable
{
    public:
        VideoBuffer();
        virtual ~VideoBuffer();

        void SetGLView(GLView *view);

        /* ext API for receiver */
        std::tr1::shared_ptr<Frame> getframe();
        void reconfigure(int width, int height, int codec, int data_len, size_t maxAudioDataLen);
        void putframe(std::tr1::shared_ptr<Frame> data);

        /* ext API for player */
        std::tr1::shared_ptr<Frame> GetFrame(int frame);
        int GetUpperBound();
        int GetLowerBound();
        int GetLastReceivedFrame();
        bool HasFrame(int number);
        void DropFrames(int low, int high);

        void Reset();

    protected:
    private:
        std::map<int, std::tr1::shared_ptr<Frame> > buffered_frames;
        GLView *view;

        pthread_mutex_t lock;

        size_t videoDataLen;
        size_t maxAudioDataLen;

        int last_frame;

        void DropUnusedFrames();
};

#endif // VIDEOBUFFER_H
