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

#include "none.h"

#include <pthread.h>
#include <queue>
#include <stdlib.h>

#include "debug.h"
#include "host.h"
#include "video_codec.h"
#include "video_compress.h"

#define MAX_QUEUE_LEN 5

#define MAGIC 0x45bb3321

using namespace std;

struct none_video_compress {
        uint32_t magic;
        pthread_mutex_t lock;
        pthread_cond_t in_cv;
        pthread_cond_t out_cv;
        queue<struct video_frame *> dummy_queue;
};

void * none_compress_init(char * opts)
{
        UNUSED(opts);

        struct none_video_compress *s;
        
        s = new none_video_compress;
        s->magic = MAGIC;
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->in_cv, NULL);
        pthread_cond_init(&s->out_cv, NULL);

        return s;
}

void none_push(void *arg, struct video_frame * tx)
{
        struct none_video_compress *s = (struct none_video_compress *) arg;

        assert(s->magic == MAGIC);

        pthread_mutex_lock(&s->lock);
        while(s->dummy_queue.size() > MAX_QUEUE_LEN) {
                pthread_cond_wait(&s->in_cv, &s->lock);
        }

        s->dummy_queue.push(tx);
        pthread_cond_signal(&s->out_cv);
        pthread_mutex_unlock(&s->lock);
}

struct video_frame * none_pop(void *arg)
{
        struct none_video_compress *s = (struct none_video_compress *) arg;
        struct video_frame *res;

        assert(s->magic == MAGIC);

        pthread_mutex_lock(&s->lock);
        while(s->dummy_queue.empty()) {
                pthread_cond_wait(&s->out_cv, &s->lock);
        }
        res = s->dummy_queue.front();
        s->dummy_queue.pop();
        pthread_cond_signal(&s->in_cv);
        pthread_mutex_unlock(&s->lock);

        return res;
}

void none_compress_done(void *arg)
{
        struct none_video_compress *s = (struct none_video_compress *) arg;

        assert(s->magic == MAGIC);

        pthread_cond_destroy(&s->in_cv);
        pthread_cond_destroy(&s->out_cv);
        pthread_mutex_destroy(&s->lock);
        delete s;
}

