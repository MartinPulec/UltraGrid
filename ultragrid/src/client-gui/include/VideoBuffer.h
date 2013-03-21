#ifndef VIDEOBUFFER_H
#define VIDEOBUFFER_H

#include <map>
#include <tr1/memory>

#include <pthread.h>

#include "audio/audio.h" // audio_desc
#include "video.h" // video_desc

#include "../include/Observable.h"

#include "Frame.h"
#include "Decompress.h"

class GLView;

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
        void reinitializeDecompress(codec_t codec, codec_t compress);

    protected:
    private:
        void DropUnusedFrames();

        std::map<int, std::tr1::shared_ptr<Frame> > m_buffered_frames;

        pthread_mutex_t m_lock;

        size_t m_videoDataLen;
        size_t m_maxAudioDataLen;

        int m_last_frame;

        Decompress m_decompress;
};

#endif // VIDEOBUFFER_H
