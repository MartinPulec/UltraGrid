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

extern "C" {
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif
#include "debug.h"
#include "host.h"
#include "video_compress/dxt_glsl.h"
#include "dxt_compress/dxt_encoder.h"
#include "compat/platform_semaphore.h"
#include "video.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#ifdef HAVE_MACOSX
#include <GL/glew.h>
#include "mac_gl_common.h"
#else
#include <GL/glew.h>
#include "x11_common.h"
#include "glx_common.h"
#endif
}

#include "watermark.h"
#include "client-gui/src/cesnet-logo-2.c"
#include "fint-logo.c"

#if defined HAVE_MACOSX && OS_VERSION_MAJOR < 11
#define glGenFramebuffers glGenFramebuffersEXT
#define glBindFramebuffer glBindFramebufferEXT
#define GL_FRAMEBUFFER GL_FRAMEBUFFER_EXT
#define glFramebufferTexture2D glFramebufferTexture2DEXT
#define glDeleteFramebuffers glDeleteFramebuffersEXT
#define GL_FRAMEBUFFER_COMPLETE GL_FRAMEBUFFER_COMPLETE_EXT
#define glCheckFramebufferStatus glCheckFramebufferStatusEXT
#endif

#define STRINGIFY(A) #A

struct state_watermark {
        struct gl_context *context;

        GLuint fbo_id;

        GLuint tex_processed;

        struct video_frame *out;
        struct tile *tile;

        GLuint g_vao;

        GLuint logo[2];

        bool configured;
};


static void configure(struct state_watermark *s, struct video_frame *tx) {
        glEnable(GL_BLEND);
        glEnable(GL_TEXTURE_2D);

        glGenFramebuffers(1, &s->fbo_id);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);


        glGenTextures(1, &s->tex_processed);
        glBindTexture(GL_TEXTURE_2D, s->tex_processed);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tx->tiles[0].width, tx->tiles[0].height,
                        0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

        s->out = vf_alloc(1);
        s->out->color_spec = RGB;
        s->out->interlacing = tx->interlacing;
        s->out->fps = tx->fps;

        s->tile = vf_get_tile(s->out, 0);

        s->tile->width  = tx->tiles[0].width;
        s->tile->height = tx->tiles[0].height;
        s->tile->storage = OPENGL_TEXTURE;
        s->tile->texture = s->tex_processed;

        //glViewport(0, 0, s->tile->width, s->tile->height);
        glBindTexture(GL_TEXTURE_2D, 0);

        s->configured = true;
}

struct state_watermark * watermark_init(struct gl_context *context)
{
        struct state_watermark *s;
        
        s = (struct state_watermark *) malloc(sizeof(struct state_watermark));
        assert (s != NULL);
        s->context = context;

        glewInit();

        glGenTextures(sizeof(s->logo) / sizeof(GLuint), s->logo);
        glBindTexture(GL_TEXTURE_2D, s->logo[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGBA, cesnet_logo.width, cesnet_logo.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, cesnet_logo.pixel_data); 

        glBindTexture(GL_TEXTURE_2D, 0);

        glGenTextures(1, &s->logo[1]);
        glBindTexture(GL_TEXTURE_2D, s->logo[1]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGBA, fint_logo.width, fint_logo.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, fint_logo.pixel_data); 

        glBindTexture(GL_TEXTURE_2D, 0);

        s->configured = false;

        return s;
}

static void watermark_loga(struct state_watermark *s)
{
        int i;
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        GLdouble right_b_x = 0.98, right_b_y = 0.98,
                 right_t_x = 0.98, right_t_y = 0.75;

        for(i = sizeof(s->logo) / sizeof(GLuint) - 1; i >= 0; --i) {
                GLdouble left_b_x, left_b_y,
                         left_t_x, left_t_y;

                int width = 0;
                int height = 0;
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);


                left_b_y = right_b_y;
                left_t_y = right_t_y;

                left_t_x = right_t_x - (right_b_y - right_t_y) / height * width / (double) s->tile->width * s->tile->height;
                left_b_x = left_t_x;

                glBindTexture(GL_TEXTURE_2D, s->logo[i]);
                glBegin(GL_QUADS);
                glTexCoord2f(0.0, 0.0); glVertex2f(left_t_x, left_t_y);
                glTexCoord2f(1.0, 0.0); glVertex2f(right_t_x, right_t_y);
                glTexCoord2f(1.0, 1.0); glVertex2f(right_b_x, right_b_y);
                glTexCoord2f(0.0, 1.0); glVertex2f(left_b_x ,left_b_y);
                glEnd();

                right_b_x = right_t_x = left_b_x - 0.05;
        }
        
        glBlendFunc(GL_SRC_ALPHA, GL_ZERO);
}

struct video_frame * add_watermark(struct state_watermark *s, struct video_frame * tx)
{
        if(tx->tiles[0].storage != OPENGL_TEXTURE) {
                return tx;
        }

        glUseProgram(0);

        if(!s->configured) {
                configure(s, tx);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo_id);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->tex_processed, 0);

        assert(GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER));

        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport( 0, 0, s->tile->width, s->tile->height);

        glClearColor(1.0,0.0,0.0,1.0);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glBindTexture(GL_TEXTURE_2D, tx->tiles[0].texture);

        glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);


#if 0
        if(encoder->legacy) {
#endif
        glBegin(GL_QUADS);
                glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
                glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
                glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
                glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
        glEnd();

        watermark_loga(s);


        //glClear(GL_COLOR_BUFFER_BIT);

#if 0
        } else {
#if ! defined HAVE_MACOSX || OS_VERSION_MAJOR >= 11
                // Compress
                glBindVertexArray(s->vao);
                //glDrawElements(GL_TRIANGLE_STRIP, sizeof(m_quad.indices) / sizeof(m_quad.indices[0]), GL_UNSIGNED_SHORT, BUFFER_OFFSET(0));
                glDrawArrays(GL_TRIANGLES, 0, 6);
                glBindVertexArray(0);
#endif
        }
#endif

        glPopAttrib();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glUseProgram(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);


        s->out->frames = tx->frames;
        s->tile->texture = s->tex_processed;

        return s->out;
}

void watermark_done(struct state_watermark *s)
{
}

