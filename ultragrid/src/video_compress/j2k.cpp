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

#include <iostream>
#include <pthread.h>
#include <queue>
#include <stdlib.h>

#include "debug.h"
#include "defs.h"
#include "demo_enc.h"
#include "host.h"
#include "messaging.h"
#include "video_codec.h"
#include "video_compress.h"

#define MAX_QUEUE_LEN 10

#define MAGIC 0x45bb3321

using namespace std;

bool j2k_reconfigure(struct j2k_video_compress *state, struct video_desc video_description);

struct j2k_video_compress: public observer {
        j2k_video_compress() : downscaled(0) {
                message_manager.register_observer(this);
        }

        struct demo_enc *j2k_encoder;
        struct video_desc saved_desc;

        bool initialized;
        bool should_exit;
        pthread_mutex_t lock;
        pthread_cond_t in_cv;
        pthread_cond_t out_cv;
        size_t counter;
        string path;

        uint32_t magic;

        volatile int downscaled;

        void notify(message *msg) {
                if(dynamic_cast<text_message *>(msg)) {
                        text_message *text_msg = dynamic_cast<text_message *>(msg);
                        cerr << text_msg->text << endl;

                        if(strncasecmp(text_msg->text.c_str(), "J2K ", 4) == 0) {
                                const char *data = text_msg->text.c_str() + 4;
                                const char *token = "HDDownscalling ";
                                if(strncasecmp(data, token, strlen(token)) == 0) {
                                        downscaled = atoi(data + strlen(token));
                                }
                        }
                }
        }
};

bool j2k_reconfigure(struct j2k_video_compress *s, struct video_desc video_description)
{
        if(s->j2k_encoder) {
                demo_enc_destroy(s->j2k_encoder);
        }

        s->j2k_encoder = demo_enc_create(NULL, 0, video_description.width,
                        video_description.height,
                        4,
                        1.3f);

        s->initialized = true;
        pthread_cond_signal(&s->out_cv);

        return s->j2k_encoder != NULL;
}

void * j2k_compress_init(char * opts)
{
        struct j2k_video_compress *s;
        
        s = new j2k_video_compress;
        s->magic = MAGIC;

        memset(&s->saved_desc, 0, sizeof(s->saved_desc));
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->out_cv, NULL);
        pthread_cond_init(&s->in_cv, NULL);
        s->initialized = false;
        s->should_exit = false;
        s->counter = 0;

        s->j2k_encoder = NULL;

        return s;
}

void j2k_push(void *arg, struct video_frame * tx, double requested_quality)
{
        struct j2k_video_compress *s = (struct j2k_video_compress *) arg;

        assert(s->magic == MAGIC);

        pthread_mutex_lock(&s->lock);

        if(!tx) {
                s->should_exit = true;
                if(!s->initialized) {
                        pthread_cond_signal(&s->out_cv);
                } else {
                        demo_enc_stop(s->j2k_encoder);
                }
        } else {
                while(s->counter > MAX_QUEUE_LEN) {
                        pthread_cond_wait(&s->in_cv, &s->lock);
                }

                if(tx->tiles[0].width != s->saved_desc.width ||
                                tx->tiles[0].height != s->saved_desc.height) {
                        s->saved_desc.width = tx->tiles[0].width;
                        s->saved_desc.height = tx->tiles[0].height;
                        bool ret = j2k_reconfigure(s, s->saved_desc);
                        assert(ret);
                }

                int bw = J2K_MAX_FRAME_MB * 1000 * 1000 * requested_quality;
                if(bw == 0) {
                        bw = 1;
                }

                float quality = 0.7;

                int subsample_factor = s->downscaled;

                if(subsample_factor == 1) {
                        quality = 0.9;
                } else if (subsample_factor == 2) {
                        quality = 1.1;
                } else if (subsample_factor == 3) {
                        quality = 1.3;
                } else if (subsample_factor == 4) {
                        quality = 1.3;
                }

                tx->tiles[0].width /= 1<<subsample_factor;
                tx->tiles[0].height /= 1<<subsample_factor;

                demo_enc_submit(s->j2k_encoder, (void *) tx,
                                tx->tiles[0].data, tx->tiles[0].data_len,
                                tx->tiles[0].data, bw,
                                quality,
                                subsample_factor,
                                video_directory // defined in main.c
                                );

                s->counter += 1;
        }
        pthread_mutex_unlock(&s->lock);
}

struct video_frame * j2k_pop(void *arg)
{
        struct j2k_video_compress *s = (struct j2k_video_compress *) arg;
        struct video_frame *res;

        assert(s->magic == MAGIC);

        pthread_mutex_lock(&s->lock);
        while(!s->initialized) {
                pthread_cond_wait(&s->out_cv, &s->lock);
        }

        if(s->should_exit) {
                pthread_mutex_unlock(&s->lock);
                return NULL;
        }
        pthread_mutex_unlock(&s->lock);

        int size = demo_enc_wait(s->j2k_encoder,
                        (void **) &res,
                        NULL,
                        NULL);

        pthread_mutex_lock(&s->lock);
        s->counter -= 1;
        pthread_cond_signal(&s->in_cv);
        pthread_mutex_unlock(&s->lock);

        if(size == 0) {
                return NULL;
        } else if(size == -1) {
                fprintf(stderr, "[J2K] Error encoding.\n");
                return NULL;
        }

        res->tiles[0].data_len = size;
        res->color_spec = J2K;

#if 0
        fprintf(stderr, "%d\n", res->frames);
        if(res->frames == 30) {
                int fd = open("frame30.j2k", O_CREAT|O_WRONLY, 0666);
                assert(fd > 0);
                size_t bytes = 0;
                do {
                        bytes += write(fd, res->tiles[0].data + bytes, size - bytes);
                } while (bytes < size);
                close(fd);
        }
#endif

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

        pthread_cond_destroy(&s->in_cv);
        pthread_cond_destroy(&s->out_cv);
        pthread_mutex_destroy(&s->lock);

        delete s;
}

