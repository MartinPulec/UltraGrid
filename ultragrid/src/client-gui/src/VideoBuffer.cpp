#include <iostream>

#include "config.h"
#include "config_unix.h"

#include "../include/VideoBuffer.h"
#include "../include/GLView.h"


using namespace std::tr1;
using namespace std;

struct CharPtrDeleter
{
    void operator()(char *ptr) const
    {
        delete[] ptr;
    }
};


VideoBuffer::VideoBuffer() :
    before_min(-1),
    after_max(0)
{
    //ctor
    pthread_mutex_init(&lock, NULL);
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

void VideoBuffer::putframe(shared_ptr<char> data, unsigned int frames)
{
    pthread_mutex_lock(&lock);
#ifdef DEBUG
    std::cerr << "Buffer: Received frame " << frames << std::endl;
#endif


    buffered_frames.insert(std::pair<int, std::tr1::shared_ptr<char> >(frames, data));
    current_item = frames;

    after_max = frames + 1;

    pthread_mutex_unlock(&lock);
}

shared_ptr<char> VideoBuffer::getframe()
{
    return shared_ptr<char> (new char[data_len], CharPtrDeleter());
}

void VideoBuffer::reconfigure(int width, int height, int codec, int data_len)
{
    pthread_mutex_lock(&lock);
    this->data_len = data_len;

    buffered_frames.clear();

    this->view->reconfigure(width, height, codec);
    pthread_mutex_unlock(&lock);
}

void VideoBuffer::DropFrames(int low, int high)
{
    pthread_mutex_lock(&lock);
    map<int, shared_frame>::iterator low_it, high_it;

    low_it = buffered_frames.lower_bound (low);
    high_it = buffered_frames.upper_bound (high);

    buffered_frames.erase(buffered_frames.begin(), low_it);
    buffered_frames.erase(high_it, buffered_frames.end());

    before_min = low;
    after_max = high;
    pthread_mutex_unlock(&lock);
}

/* ext API for player */
std::tr1::shared_ptr<char> VideoBuffer::GetFrame(int frame)
{
    std::tr1::shared_ptr<char>    res;

    pthread_mutex_lock(&lock);
    std::map<int, std::tr1::shared_ptr<char> >::iterator it = buffered_frames.find(frame);
    if(it == buffered_frames.end()) {
        res = std::tr1::shared_ptr<char>();
    } else {
        res = it->second;
    }

    pthread_mutex_unlock(&lock);

    return res;
}

int VideoBuffer::GetLowerBound()
{
    return before_min;
}

int VideoBuffer::GetUpperBound()
{
    return after_max;
}

void VideoBuffer::Reset()
{
    after_max = 0;
    before_min = -1;

    buffered_frames.clear();
}
