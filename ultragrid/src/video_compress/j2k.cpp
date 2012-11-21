/*
 * FILE:    dxt_glsl_compress.c
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
#endif /* HAVE_CONFIG_H */

#include "j2k.h"

#include <pthread.h>
#include <queue>
#include <stdlib.h>

#include "debug.h"
#include "demo_enc.h"
#include "host.h"
#include "video_codec.h"
#include "video_compress.h"


#define MAGIC 0x45bb3321

using namespace std;

bool j2k_reconfigure(struct j2k_video_compress *state, struct video_desc video_description);

struct j2k_video_compress {
        struct demo_enc *j2k_encoder;

        uint32_t magic;
};

bool j2k_reconfigure(struct j2k_video_compress *s, struct video_desc video_description)
{
        if(s->j2k_encoder) {
                demo_enc_destroy(s->j2k_encoder);
        }

        s->j2k_encoder = demo_enc_create(NULL, 0, video_description.width,
                        video_description.height,
                        4,
                        1.0f);
}

void * j2k_compress_init(char * opts)
{
        struct j2k_video_compress *s;
        
        s = new j2k_video_compress;
        s->magic = MAGIC;

        s->j2k_encoder = NULL;

        return s;
}

void j2k_push(void *arg, struct video_frame * tx)
{
        struct j2k_video_compress *s = (struct j2k_video_compress *) arg;

        assert(s->magic == MAGIC);

        demo_enc_submit(s->j2k_encoder, (void *) tx,
                        tx->tiles[0].data, tx->tiles[0].data_len,
                        tx->tiles[0].data, tx->tiles[0].data_len,
                        0.7,
                        0);
}

struct video_frame * j2k_pop(void *arg)
{
        struct j2k_video_compress *s = (struct j2k_video_compress *) arg;
        struct video_frame *res;

        assert(s->magic == MAGIC);

        int size = demo_enc_wait(s->j2k_encoder,
                        (void **) &res,
                        NULL,
                        NULL);

        res->tiles[0].data_len = size;
        res->color_spec = J2K;

        assert(size > 0);

        return res;
}

void j2k_compress_done(void *arg)
{
        struct j2k_video_compress *s = (struct j2k_video_compress *) arg;

        assert(s->magic == MAGIC);

        if(s->j2k_encoder) {
                demo_enc_destroy(s->j2k_encoder);
        }

        delete s;
}

