#include "../include/VideoBufferOnFlyManager.h"

#include <iostream>

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif

#include "tv.h"

VideoBufferOnFlyManager::VideoBufferOnFlyManager(ClientManager & connection, VideoBuffer & buffer) :
    connection_(connection), buffer_(buffer),
    lastRequestedFrame_(-1)
{
    //ctor
    platform_spin_init(&this->lock_);

    this->lastRequestTime_.tv_sec = 0;
    this->lastRequestTime_.tv_usec = 0;
    this->lastFrameReceivedTime_.tv_sec = 0;
    this->lastFrameReceivedTime_.tv_usec = 0;

    this->buffer_.registerObserver(this);
}

VideoBufferOnFlyManager::~VideoBufferOnFlyManager()
{
    //dtor
    this->buffer_.unregisterObserver(this);

    platform_spin_destroy(&this->lock_);
}

void VideoBufferOnFlyManager::Notify()
{
    platform_spin_lock(&this->lock_);
    if(buffer_.HasFrame(this->lastRequestedFrame_)) {
        gettimeofday(&this->lastFrameReceivedTime_, NULL);
    }
    platform_spin_unlock(&this->lock_);
}

int VideoBufferOnFlyManager::Clamp(int frame_nr, int total_frames)
{
    if(frame_nr < 0)
        return 0;
    if(frame_nr >= total_frames)
        return total_frames - 1;
    return frame_nr;
}

void VideoBufferOnFlyManager::RequestAdditionalBuffers(int current_frame, int total_frames, int play_direction)
{
    platform_spin_lock(&this->lock_);

    this->lastRequestedFrame_ = current_frame;
    this->lastFrameReceivedTime_.tv_sec = 0;
    this->lastFrameReceivedTime_.tv_usec = 0;

    if(!this->buffer_.HasFrame(Clamp(current_frame, total_frames))) {
            this->connection_.pause(current_frame - play_direction * 2, 5, true);
            goto request_sent;
    }

    for(int i = 1; i < 5; ++i) {
        if(!this->buffer_.HasFrame(Clamp(current_frame + i, total_frames))) {
            //require 5 frames
            //playdirection is either 1 or -1, so if 0, use current frame + i
            // otherwise, current_frame + i * 4 is used
            int startpos = current_frame + i + (play_direction - 1) / 2 * -4;

            this->connection_.pause(startpos, 5, true);
            goto request_sent;
        }
        if(!this->buffer_.HasFrame(Clamp(current_frame - i, total_frames))) {
            int startpos = current_frame - i + (play_direction - 1) / 2 * -4;

            //require 5 frames
            this->connection_.pause(startpos - 4, 5, true);
            goto request_sent;
        }
    }

request_not_sent:
    platform_spin_unlock(&this->lock_);
    return;

request_sent:

    gettimeofday(&this->lastRequestTime_, NULL);

    platform_spin_unlock(&this->lock_);
}

bool VideoBufferOnFlyManager::LastRequestIsDue(double fps)
{
    struct timeval current_time;
    bool ret;

    platform_spin_unlock(&this->lock_);

    gettimeofday(&current_time, NULL);

    if(tv_gt(this->lastFrameReceivedTime_, this->lastRequestTime_) ||
       tv_diff_usec(current_time, this->lastRequestTime_) / 1000 > (connection_.GetRTTMs() + 1000   )) {

       std::cerr <<    tv_gt(this->lastFrameReceivedTime_, this->lastRequestTime_) << " " <<
       tv_diff_usec(current_time, this->lastRequestTime_) / 1000 << " " << connection_.GetRTTMs() * 2 << std::endl;
        ret = true;
    } else {
        ret = false;
    }
    platform_spin_unlock(&this->lock_);

    return ret;
}
