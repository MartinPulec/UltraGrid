#ifndef VIDEOBUFFER_H
#define VIDEOBUFFER_H

#include <map>
#include <tr1/memory>

#include <pthread.h>

#include "../include/Observable.h"


class GLView;

typedef std::tr1::shared_ptr<char> shared_frame;

class VideoBuffer: public Observable
{
    public:
        VideoBuffer();
        virtual ~VideoBuffer();

        void SetGLView(GLView *view);

        /* ext API for receiver */
        std::tr1::shared_ptr<char> getframe();
        void reconfigure(int width, int height, int codec, int data_len);
        void putframe(std::tr1::shared_ptr<char> data, unsigned int frames);

        /* ext API for player */
        std::tr1::shared_ptr<char> GetFrame(int frame);
        int GetUpperBound();
        int GetLowerBound();
        bool HasFrame(int number);
        void DropFrames(int low, int high);

        void Reset();

    protected:
    private:
        std::map<int, std::tr1::shared_ptr<char> > buffered_frames;
        GLView *view;

        pthread_mutex_t lock;

        int data_len;

        void DropUnusedFrames();
};

#endif // VIDEOBUFFER_H
