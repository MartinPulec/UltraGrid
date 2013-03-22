#include "config.h"
#include "config_unix.h"

#include "../include/VideoBuffer.h"

#include <iostream>

using namespace std::tr1;
using namespace std;


#define DEMUX_TO_DECOMPRESS_INT 15

VideoBuffer::VideoBuffer() :
    m_decompress(this)
{
    //ctor
    SetDefaultValues();
}

void VideoBuffer::SetDefaultValues()
{
    m_last_frame_received = -1;

    m_dec_req_first = 0;
}

VideoBuffer::~VideoBuffer()
{
    //dtor
    m_decompress.waitFree();
}

void VideoBuffer::put_received_frame(std::tr1::shared_ptr<Frame> data)
{
    std::lock_guard<std::mutex> lock(m_lock);

    int seq_num = data->video_desc.seq_num;
//#ifdef DEBUG
    std::clog << "Buffer: Received frame " << seq_num << std::endl;
//#endif

    m_received_frames.insert(std::pair<int, std::tr1::shared_ptr<Frame> >(seq_num, data));
    m_last_frame_received = seq_num;

    if(seq_num >= m_dec_req_first && seq_num < m_dec_req_first + DEMUX_TO_DECOMPRESS_INT) {
        m_decompress.push(data);
        m_decompress_enqueued[seq_num] = true;
    }
    //Observable::notifyObservers();
}

void VideoBuffer::put_decompressed_frame(std::tr1::shared_ptr<Frame> data)
{
    std::lock_guard<std::mutex> lock(m_decompressed_frames_lock);
    int seq_num = data->video_desc.seq_num;

    m_decompressed_frames.insert(std::pair<int, std::tr1::shared_ptr<Frame> >(seq_num,
                                                                              data));
    m_frame_decompressed.notify_one();

    std::clog << "Decompressed: " << seq_num << endl;
}


void VideoBuffer::DropFrames(int low, int high)
{
    std::lock_guard<std::mutex> lock(m_lock);

    map<int, shared_frame>::iterator low_it, high_it;

    low_it = m_received_frames.lower_bound (low);
    high_it = m_received_frames.upper_bound (high);

    //received_frames.erase(received_frames.begin(), low_it);
    //received_frames.erase(high_it, received_frames.end());
    m_received_frames.erase(low_it, high_it);

    //before_min = (before_min > low ? before_min : low);
    //after_max = (after_max < high ? after_max : high);
}

void VideoBuffer::DecompressFrames()
{
    // should hold lock from caller
    std::lock_guard<std::mutex> lock(m_lock, std::adopt_lock_t());

    for (int i = m_dec_req_first; i < m_dec_req_first + DEMUX_TO_DECOMPRESS_INT; ++i)
    {
        std::map<int, bool>::iterator it_state = m_decompress_enqueued.find(i);
        if(it_state == m_decompress_enqueued.end()) { // not ENQUEUED nor DECOMPRESSED
            frame_map::iterator it_received_frame = m_received_frames.find(i);
            if(it_received_frame != m_received_frames.end())
            {
                std::clog << "Want Decompressed: " << i<< endl;
                m_decompress.push(it_received_frame->second);
                m_decompress_enqueued[i] = true;
            }
        }

    }
}

void VideoBuffer::DiscardOldDecompressedFrames()
{
    // should hold master lock from caller
    std::lock_guard<std::mutex> lock(m_lock, std::adopt_lock_t());
    std::lock_guard<std::mutex> lock_decompressed(m_decompressed_frames_lock);

    // [0, m_dec_req_first)
    m_decompress_enqueued.erase(m_decompress_enqueued.begin(), m_decompress_enqueued.lower_bound(m_dec_req_first));
    m_decompressed_frames.erase(m_decompressed_frames.begin(),
                                m_decompressed_frames.lower_bound(m_dec_req_first));
    // [m_dec_req_first, end)
    m_decompress_enqueued.erase(m_decompress_enqueued.upper_bound(m_dec_req_first + DEMUX_TO_DECOMPRESS_INT),
                             m_decompress_enqueued.end());
    m_decompressed_frames.erase(m_decompressed_frames.upper_bound(m_dec_req_first + DEMUX_TO_DECOMPRESS_INT),
                                m_decompressed_frames.end());

}

/* ext API for player */
std::tr1::shared_ptr<Frame> VideoBuffer::GetFrame(int frame)
{
    std::unique_lock<std::mutex> lock(m_lock);
    std::tr1::shared_ptr<Frame> res;

    m_dec_req_first = frame;
    DecompressFrames();
    DiscardOldDecompressedFrames();
    /* check if we have frames
     * if not, try to enqueue more
     */
    std::unique_lock<std::mutex> lock_decompressed(m_decompressed_frames_lock);
    std::map<int, bool>::iterator it_state = m_decompress_enqueued.find(frame);
    if(it_state != m_decompress_enqueued.end()) { // not enqueued
        while(m_decompressed_frames.find(frame) == m_decompressed_frames.end()) {
            m_frame_decompressed.wait(lock_decompressed);
        }
    } else {
        return std::tr1::shared_ptr<Frame>();
    }

    std::map<int, std::tr1::shared_ptr<Frame> >::iterator it = m_decompressed_frames.find(frame);
    assert(it != m_received_frames.end());

    return it->second;
}

int VideoBuffer::GetLowerBound()
{
    std::lock_guard<std::mutex> lock(m_lock);
    int before_min;

    if(m_received_frames.begin() != m_received_frames.end()) {
        before_min = m_received_frames.begin()->first - 1;
    } else {
        before_min = -1;
    }

    return before_min;
}

int VideoBuffer::GetUpperBound()
{
    int after_max;
    std::lock_guard<std::mutex> lock(m_lock);

    if(m_received_frames.begin() != m_received_frames.end()) {
        after_max = m_received_frames.rbegin()->first + 1;
    } else {
        after_max = 0;
    }

    return after_max;
}

void VideoBuffer::Reset()
{
    std::lock_guard<std::mutex> lock(m_lock);

    m_decompress.waitFree();

    std::lock_guard<std::mutex> decompress_lock(m_decompressed_frames_lock);

    m_received_frames.clear();
    m_decompressed_frames.clear();
    m_decompress_enqueued.clear();
    SetDefaultValues();
}

// Currently unused?
bool VideoBuffer::HasFrame(int number)
{
    bool ret;
    std::lock_guard<std::mutex> lock(m_lock);

    ret = m_received_frames.find(number) != m_received_frames.end();

    return ret;
}

int VideoBuffer::GetLastReceivedFrame()
{
    return m_last_frame_received;
}

void VideoBuffer::reinitializeDecompress(codec_t compress)
{
    std::lock_guard<std::mutex> lock(m_lock);

    m_decompress.reintializeDecompress(compress);
}
