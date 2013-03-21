#include "config.h"
#include "config_unix.h"

#include "../include/VideoBuffer.h"

#include <iostream>

using namespace std::tr1;
using namespace std;

VideoBuffer::VideoBuffer() :
    m_maxAudioDataLen(0),
    m_decompress(this)
{
    //ctor
    int ret = pthread_mutex_init(&m_lock, NULL);
    assert(ret == 0);
    m_last_frame = -1;
}

VideoBuffer::~VideoBuffer()
{
    //dtor
    pthread_mutex_destroy(&m_lock);
}

void VideoBuffer::putframe(std::tr1::shared_ptr<Frame> data)
{
    pthread_mutex_lock(&m_lock);

    int seq_num = data->video_desc.seq_num ;
//#ifdef DEBUG
    std::cerr << "Buffer: Received frame " << seq_num << std::endl;
//#endif


    m_buffered_frames.insert(std::pair<int, std::tr1::shared_ptr<Frame> >(seq_num, data));
    m_last_frame = seq_num;

    pthread_mutex_unlock(&m_lock);

    //Observable::notifyObservers();
}

std::tr1::shared_ptr<Frame> VideoBuffer::getframe()
{
    return std::tr1::shared_ptr<Frame> (new Frame(m_videoDataLen, m_maxAudioDataLen));
}

void VideoBuffer::reconfigure(int width, int height, int codec, int videoDataLen, size_t maxAudioDataLen)
{
    pthread_mutex_lock(&m_lock);
    {
        m_videoDataLen = videoDataLen;
        m_maxAudioDataLen = maxAudioDataLen;

        m_buffered_frames.clear();
        m_last_frame = -1;
    }
    pthread_mutex_unlock(&m_lock);
}

void VideoBuffer::DropFrames(int low, int high)
{
    pthread_mutex_lock(&m_lock);
    {
        map<int, shared_frame>::iterator low_it, high_it;

        low_it = m_buffered_frames.lower_bound (low);
        high_it = m_buffered_frames.upper_bound (high);

        //buffered_frames.erase(buffered_frames.begin(), low_it);
        //buffered_frames.erase(high_it, buffered_frames.end());
        m_buffered_frames.erase(low_it, high_it);

        //before_min = (before_min > low ? before_min : low);
        //after_max = (after_max < high ? after_max : high);
    }
    pthread_mutex_unlock(&m_lock);
}

/* ext API for player */
std::tr1::shared_ptr<Frame> VideoBuffer::GetFrame(int frame)
{
    std::tr1::shared_ptr<Frame> res;

    pthread_mutex_lock(&m_lock);
    {
        std::map<int, std::tr1::shared_ptr<Frame> >::iterator it = m_buffered_frames.find(frame);
        if(it == m_buffered_frames.end()) {
            res = std::tr1::shared_ptr<Frame>();
        } else {
            res = it->second;
        }
    }
    pthread_mutex_unlock(&m_lock);

    return res;
}

int VideoBuffer::GetLowerBound()
{
    int before_min;

    pthread_mutex_lock(&m_lock);
    {
        if(m_buffered_frames.begin() != m_buffered_frames.end()) {
            before_min = m_buffered_frames.begin()->first - 1;
        } else {
            before_min = -1;
        }
    }
    pthread_mutex_unlock(&m_lock);

    return before_min;
}

int VideoBuffer::GetUpperBound()
{
    int after_max;

    pthread_mutex_lock(&m_lock);
    {
        if(m_buffered_frames.begin() != m_buffered_frames.end()) {
            after_max = m_buffered_frames.rbegin()->first + 1;
        } else {
            after_max = 0;
        }
    }
    pthread_mutex_unlock(&m_lock);

    return after_max;
}

void VideoBuffer::Reset()
{
    pthread_mutex_lock(&m_lock);
    {
        m_buffered_frames.clear();
        m_last_frame = -1;
    }
    pthread_mutex_unlock(&m_lock);
}

// Currently unused?
bool VideoBuffer::HasFrame(int number)
{
    bool ret;

    pthread_mutex_lock(&m_lock);
    {
        ret = m_buffered_frames.find(number) != m_buffered_frames.end();
    }
    pthread_mutex_unlock(&m_lock);

    return ret;
}

int VideoBuffer::GetLastReceivedFrame()
{
    return m_last_frame;
}

void VideoBuffer::reinitializeDecompress(codec_t codec, codec_t compress)
{
    m_decompress.reintializeDecompress(codec, compress);
}
