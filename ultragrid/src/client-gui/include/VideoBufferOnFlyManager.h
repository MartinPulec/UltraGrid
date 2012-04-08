#ifndef VIDEOBUFFERONFLYMANAGER_H
#define VIDEOBUFFERONFLYMANAGER_H

#include <pthread.h>
#include <sys/time.h>

#include "../include/ClientManager.h"
#include "../include/VideoBuffer.h"
#include "../include/Observer.h"

class VideoBufferOnFlyManager: public Observer
{
    public:
        VideoBufferOnFlyManager(ClientManager & connection, VideoBuffer & buffer);
        virtual ~VideoBufferOnFlyManager();
        void Notify();
        void RequestAdditionalBuffers(int current_frame, int total_frames, int play_direction);
        bool LastRequestIsDue(double fps);

    protected:
    private:
        ClientManager &connection_;
        VideoBuffer & buffer_;

        pthread_spinlock_t lock_;

        struct timeval lastRequestTime_;
        struct timeval lastFrameReceivedTime_;

        int lastRequestedFrame_;

        int Clamp(int frame_nr, int total_frames);
};

#endif // VIDEOBUFFERONFLYMANAGER_H
