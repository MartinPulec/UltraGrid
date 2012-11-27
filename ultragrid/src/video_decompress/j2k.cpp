/*
 * FILE:    video_decompress/dxt_glsl.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2011 CESNET z.s.p.o.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "video_decompress/j2k.h"

#include <iostream>
#include <pthread.h>
#include <queue>
#include <stdlib.h>

#include "debug.h"
#include "demo_dec/demo_dec.h"
#include "video_codec.h"

using namespace std;
using namespace std::tr1;

struct j2k_decompress_data {
        j2k_decompress_data(shared_ptr<Frame> frame_, shared_ptr<char> decompressed_) :
                frame(frame_),
                decompressed(decompressed_)
        {
        }

        shared_ptr<Frame> frame;
        shared_ptr<char> decompressed;
};

class state_j2k_decompress {
        public:
                state_j2k_decompress(codec_t out_codec_) : out_codec(out_codec_) {
                        state = demo_dec_create(NULL, 0);
                        if(!state) {
                                throw;
                        }
                }

                virtual ~state_j2k_decompress() {
                        demo_dec_destroy(state);
                }

                struct demo_dec *state;
                codec_t out_codec;
};

void * j2k_decompress_init(codec_t out_codec)
{
        class state_j2k_decompress *s;

        out_codec = XPD10;
        s = new state_j2k_decompress(out_codec);

        return s;
}

int j2k_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
}

void j2k_push(void *state, std::tr1::shared_ptr<Frame> frame)
{
        class state_j2k_decompress *s = 
                (class state_j2k_decompress *) state;

        if(!frame) {
                // pass poisoned pill
                demo_dec_stop(s->state);
                return;
        }

        size_t new_length = vc_get_linesize(frame->video_desc.width, s->out_codec) *
                frame->video_desc.height;
        cerr << "NEW lenght " << new_length << endl;
        shared_ptr<char> decompressed(std::tr1::shared_ptr<char> (new char[new_length], CharPtrDeleter()));

        struct j2k_decompress_data *new_item;
        new_item = new j2k_decompress_data(frame, decompressed);

        demo_dec_submit(s->state, (void *) new_item,
                        decompressed.get(),
                        frame->video.get(),
                        frame->video_len);
}

std::tr1::shared_ptr<Frame> j2k_pop(void *state)
{
        struct j2k_decompress_data *item;
        class state_j2k_decompress *s = 
                (class state_j2k_decompress *) state;
        std::tr1::shared_ptr<Frame> ret;

        int err = demo_dec_wait(
                 s->state,
                 (void **) &item,
                 NULL,
                 NULL
                );

        if(err == 0) {
                ret = item->frame;
                ret->video_desc.color_spec = s->out_codec;
                ret->video_len = vc_get_linesize(ret->video_desc.width, s->out_codec) *
                        ret->video_desc.height;
                ret->video = item->decompressed;

                delete item;
        } else {
                if(err == 2) cerr << "Error decoding J2K" << endl;
                ret = std::tr1::shared_ptr<Frame>();
        }

        return ret;
}

void j2k_decompress_done(void *state)
{
        class state_j2k_decompress *s = 
                (class state_j2k_decompress *) state;

        delete s;
}

