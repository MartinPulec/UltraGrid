#ifndef FRAME_H
#define FRAME_H

#include <tr1/memory>

#include "audio/audio.h"
#include "video.h"


struct CharPtrDeleter
{
    void operator()(char *ptr) const
    {
        delete[] ptr;
    }
};

struct Frame {
    Frame(size_t audioLen_, size_t videoLen_) :
            audio(std::tr1::shared_ptr<char> (new char[audioLen_], CharPtrDeleter())),
            video(std::tr1::shared_ptr<char> (new char[videoLen_], CharPtrDeleter())),
            video_len(videoLen_), audio_len(audioLen_) {}


    std::tr1::shared_ptr<char> video;
    std::tr1::shared_ptr<char> audio;
    size_t audio_len;
    size_t video_len;
    struct audio_desc audio_desc;
    struct video_desc video_desc;
};

#endif // FRAME_H
