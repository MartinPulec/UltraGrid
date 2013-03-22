#ifndef VIDEOBUFFER_H
#define VIDEOBUFFER_H

#include <condition_variable>
#include <map>
#include <mutex>
#include <tr1/memory>

#include <pthread.h>

#include "audio/audio.h" // audio_desc
#include "video.h" // video_desc

#include "../include/Observable.h"

#include "Frame.h"
#include "Decompress.h"

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
typedef std::map<int, std::tr1::shared_ptr<Frame> > frame_map;

class VideoBuffer: public Observable
{
    public:
        VideoBuffer();
        virtual ~VideoBuffer();

        /* ext API for receiver */
        void put_received_frame(std::tr1::shared_ptr<Frame> data);

        /* API for decompress */
        void put_decompressed_frame(std::tr1::shared_ptr<Frame> data);

        /* ext API for player */
        std::tr1::shared_ptr<Frame> GetFrame(int frame);
        int GetUpperBound();
        int GetLowerBound();
        int GetLastReceivedFrame();
        bool HasFrame(int number);
        void DropFrames(int low, int high);

        void Reset();
        void reinitializeDecompress(codec_t compress);

    protected:
    private:
        void DropUnusedFrames();
        void SetDefaultValues();
        void DecompressFrames();
        void DiscardOldDecompressedFrames();

        std::map<int, std::tr1::shared_ptr<Frame> > m_received_frames;
        std::map<int, std::tr1::shared_ptr<Frame> > m_decompressed_frames;

        // master lock
        std::mutex m_lock;
        // for access to m_decompressed_frames
        std::mutex m_decompressed_frames_lock;

        std::condition_variable m_frame_decompressed;

        int m_last_frame_received;

        Decompress m_decompress;
        std::map<int, bool> m_decompress_enqueued;
        int        m_dec_req_first;
};

#endif // VIDEOBUFFER_H
