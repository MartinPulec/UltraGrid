#include "Decompress.h"

#include <string.h>
#include <wx/log.h>

#include "video_decompress.h"

#include "VideoBuffer.h"

using namespace std::tr1;

////////////////////////////////////////////////////////////
// DecompressThread
////////////////////////////////////////////////////////////
DecompressThread::DecompressThread(Decompress *decompress_) :
    wxThread(wxTHREAD_JOINABLE), parent(decompress_), cv(lock)
{
}

DecompressThread::~DecompressThread()
{
}

wxThread::ExitCode DecompressThread::Entry()
{
    while(1) {
        std::tr1::shared_ptr<Frame> frame = decompress_pop(parent->decompress);

        if(!frame) {
            // process poisoned pill - exit thread
            break;
        }

        parent->pushDecompressedFrame(frame);
    }
}

////////////////////////////////////////////////////////////
// Decompress
////////////////////////////////////////////////////////////
Decompress::Decompress(VideoBuffer *b) :
    buffer(b), thread(0), decompress(0)
{
}

Decompress::~Decompress()
{
    // send a poisoned pill
    push(std::tr1::shared_ptr<Frame>());
    thread->Wait();
}

void Decompress::push(std::tr1::shared_ptr<Frame> frame)
{
    decompress_push(decompress, frame);
}

void Decompress::pushDecompressedFrame(std::tr1::shared_ptr<Frame> frame)
{
    buffer->putframe(frame);
}

void Decompress::reintializeDecompress(codec_t in_codec, codec_t out_codec)
{
    if(thread) {
        // send a poisoned pill
        push(std::tr1::shared_ptr<Frame>());
        thread->Wait();
        thread = 0;
        decompress_done(decompress);
    }

    unsigned int decoder_index;

    switch(in_codec) {
        case JPEG:
            decoder_index = JPEG_MAGIC;
            break;
        default:
            decoder_index = NULL_MAGIC;
            break;
    }

    decompress = decompress_init(decoder_index, out_codec);
    assert(decompress);

    thread = new DecompressThread(this);

    if (thread->Create() != wxTHREAD_NO_ERROR )
    {
        wxLogError(_T("Can't create the thread!"));
        delete thread;
        throw;
    } else {
        if (thread->Run() != wxTHREAD_NO_ERROR) {
            wxLogError(_T("Can't create the thread!"));
            delete thread;
            throw;
        }
    }
}
