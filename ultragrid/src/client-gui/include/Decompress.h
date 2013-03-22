#ifndef DECOMPRESS_H
#define DECOMPRESS_H

#include <tr1/memory>
#include <queue>
#include <wx/thread.h>

#include "audio/audio.h" // audio_desc
#include "video.h" // audio_desc
#include "video_decompress.h" // audio_desc

struct Frame;
class VideoBuffer;
class Decompress;

class DecompressThread : public wxThread
{
    public:
        DecompressThread(Decompress *decompress);
        virtual ~DecompressThread();

    protected:
        virtual ExitCode Entry();
    private:
        Decompress                 *parent;
        wxMutex                     lock;
        wxCondition                 cv;
        std::queue<std::tr1::shared_ptr<Frame> > in_queue;

        friend class Decompress;
};

class Decompress
{
    public:
        Decompress(VideoBuffer *buffer);
        virtual ~Decompress();

        void push(std::tr1::shared_ptr<Frame> frame);
        void pushDecompressedFrame(std::tr1::shared_ptr<Frame> frame);

        void reintializeDecompress(codec_t compress);
        /**
         * Waits until all frames are read from decompress
         */
        void waitFree();

    protected:
    private:
        DecompressThread            *thread;
        VideoBuffer                 *buffer;

        struct video_desc           savedVideoDesc;
        codec_t                      in_codec;
        codec_t                      out_codec;

        struct state_decompress    *decompress;
        friend class DecompressThread;
};


#endif // DECOMPRESS_H
