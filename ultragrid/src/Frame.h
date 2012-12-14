#ifndef FRAME_H
#define FRAME_H

#include <tr1/memory>

#include "audio/audio.h"
#include "cuda_memory_pool.h"
#include "video.h"


struct CharPtrDeleter
{
    void operator()(char *ptr) const
    {
        delete[] ptr;
    }
};

struct CudaDeleter
{
    CudaDeleter(size_t size_) : size(size_)
    {
    }

    void operator()(char *ptr) const
    {
        cuda_free(ptr, size);
    }

    size_t size;
};

struct Frame {
    Frame(size_t maxAudioLen_, size_t maxVideoLen_) :
            audio(std::tr1::shared_ptr<char> (new char[maxAudioLen_], CharPtrDeleter())),
            video(std::tr1::shared_ptr<char> ((char *) cuda_alloc(maxVideoLen_), CudaDeleter(maxVideoLen_))),
            max_video_len(maxVideoLen_), max_audio_len(maxAudioLen_) {}


    std::tr1::shared_ptr<char> video;
    std::tr1::shared_ptr<char> audio;
    size_t max_audio_len;
    size_t audio_len;
    size_t max_video_len;
    size_t video_len;
    struct audio_desc audio_desc;
    struct video_desc video_desc;
};

#endif // FRAME_H
