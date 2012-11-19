#include "config.h"
#include "config_unix.h"

#include "../include/VideoBuffer.h"
#include "../include/GLView.h"

#include <iostream>

using namespace std::tr1;
using namespace std;

struct CharPtrDeleter
{
    void operator()(char *ptr) const
    {
        delete[] ptr;
    }
};

Frame::Frame(size_t maxVideoLen_, size_t maxAudioLen_) :
            audio(shared_ptr<char> (new char[maxAudioLen_], CharPtrDeleter())),
            video(shared_ptr<char> (new char[maxVideoLen_], CharPtrDeleter())),
            audioLen(0), maxAudioLen(maxAudioLen_)
{
}

VideoBuffer::VideoBuffer() :
    maxAudioDataLen(0)
{
    //ctor
    int ret = pthread_mutex_init(&lock, NULL);
    assert(ret == 0);
    last_frame = -1;
}

VideoBuffer::~VideoBuffer()
{
    //dtor
    pthread_mutex_destroy(&lock);
}

void VideoBuffer::SetGLView(GLView *view)
{
    this->view = view;
}

void VideoBuffer::putframe(shared_ptr<Frame> data, unsigned int frames)
{
    pthread_mutex_lock(&lock);
//#ifdef DEBUG
    std::cerr << "Buffer: Received frame " << frames << std::endl;
//#endif


    buffered_frames.insert(std::pair<int, std::tr1::shared_ptr<Frame> >(frames, data));
    last_frame = frames;

    pthread_mutex_unlock(&lock);

    //Observable::notifyObservers();
}

shared_ptr<Frame> VideoBuffer::getframe()
{
    return shared_ptr<Frame> (new Frame(videoDataLen, maxAudioDataLen));
}

void VideoBuffer::reconfigure(int width, int height, int codec, int videoDataLen, size_t maxAudioDataLen)
{
    pthread_mutex_lock(&lock);
    {
        this->videoDataLen = videoDataLen;
        this->maxAudioDataLen = maxAudioDataLen;

        buffered_frames.clear();
        last_frame = -1;

        this->view->reconfigure(width, height, codec);
    }
    pthread_mutex_unlock(&lock);
}

void VideoBuffer::DropFrames(int low, int high)
{
    pthread_mutex_lock(&lock);
    {
        map<int, shared_frame>::iterator low_it, high_it;

        low_it = buffered_frames.lower_bound (low);
        high_it = buffered_frames.upper_bound (high);

        //buffered_frames.erase(buffered_frames.begin(), low_it);
        //buffered_frames.erase(high_it, buffered_frames.end());
        buffered_frames.erase(low_it, high_it);

        //before_min = (before_min > low ? before_min : low);
        //after_max = (after_max < high ? after_max : high);
    }
    pthread_mutex_unlock(&lock);
}

/* ext API for player */
std::tr1::shared_ptr<Frame> VideoBuffer::GetFrame(int frame)
{
    std::tr1::shared_ptr<Frame>    res;

    pthread_mutex_lock(&lock);
    {
        std::map<int, std::tr1::shared_ptr<Frame> >::iterator it = buffered_frames.find(frame);
        if(it == buffered_frames.end()) {
            res = std::tr1::shared_ptr<Frame>();
        } else {
            res = it->second;
        }
    }
    pthread_mutex_unlock(&lock);

    return res;
}

int VideoBuffer::GetLowerBound()
{
    int before_min;

    pthread_mutex_lock(&lock);
    {
        if(buffered_frames.begin() != buffered_frames.end()) {
            before_min = buffered_frames.begin()->first - 1;
        } else {
            before_min = -1;
        }
    }
    pthread_mutex_unlock(&lock);

    return before_min;
}

int VideoBuffer::GetUpperBound()
{
    int after_max;

    pthread_mutex_lock(&lock);
    {
        if(buffered_frames.begin() != buffered_frames.end()) {
            after_max = buffered_frames.rbegin()->first + 1;
        } else {
            after_max = 0;
        }
    }
    pthread_mutex_unlock(&lock);

    return after_max;
}

void VideoBuffer::Reset()
{
    pthread_mutex_lock(&lock);
    {
        buffered_frames.clear();
        last_frame = -1;
    }
    pthread_mutex_unlock(&lock);
}

// Currently unused?
bool VideoBuffer::HasFrame(int number)
{
    bool ret;

    pthread_mutex_lock(&lock);
    {
        ret = buffered_frames.find(number) != buffered_frames.end();
    }
    pthread_mutex_unlock(&lock);

    return ret;
}

int VideoBuffer::GetLastReceivedFrame()
{
    return last_frame;
}
