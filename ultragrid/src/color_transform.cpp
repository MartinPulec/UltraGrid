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
#include "mac_gl_common.h"
#else
#include <GL/glew.h>
#include "x11_common.h"
#include "glx_common.h"
#endif
}

#include "color_transform.h"

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

static char * copy = STRINGIFY(
uniform sampler2D image;
void main()
{
        vec4 col = texture2D(image, gl_TexCoord[0].st);
        gl_FragColor = col;
});

// source code for a shader unit (xsedmik)
static char * lut3D_matrix = STRINGIFY(
uniform sampler2D image;
uniform float imageWidth;
uniform mat4 matrix;

void main()
{
        vec4 col = texture2D(image, gl_TexCoord[0].st);
        gl_FragColor = matrix * col;
});

static char * vert = STRINGIFY(
void main() {
        gl_TexCoord[0] = gl_MultiTexCoord0;
        gl_Position = ftransform();
});


struct state_color_transform {
        struct gl_context *context;

        GLuint fbo_id;

        GLuint tex_input, tex_processed;

        struct video_frame *out;
        struct tile *tile;

        unsigned int configured:1;

        GLuint g_vao;

        GLuint program;

        struct lut_list *luts_to_apply;
};

static void build_programs(struct state_color_transform *s)
{
        char            *log;
        const GLchar    *VProgram, *FProgram;
        GLuint     VSHandle, FSHandle;

        if(s->luts_to_apply) {
                assert(s->luts_to_apply->next  == 0); // not yet implemented

                assert(s->luts_to_apply->type == LUT_3D_MATRIX); //only implemented by now

                if(s->luts_to_apply->type == LUT_3D_MATRIX) {
                        FProgram = (const GLchar *) lut3D_matrix;
                }
        } else {
                FProgram = (const GLchar *) copy;
        }

        VProgram = (const GLchar *) vert;
        /* Set up program objects. */
        s->program =glCreateProgram();
       VSHandle=glCreateShader(GL_VERTEX_SHADER);
        FSHandle=glCreateShader(GL_FRAGMENT_SHADER);

        /* Compile Shader */
        glShaderSource(FSHandle,1, &FProgram,NULL);
        glCompileShader(FSHandle);

        /* Print compile log */
        log=(char*)calloc(32768,sizeof(char));
        glGetShaderInfoLog(FSHandle,32768,NULL,log);
        printf("Compile Log: %s\n", log);

        glShaderSource(VSHandle,1, &VProgram,NULL);
        glCompileShader(VSHandle);
        memset(log, 0, 32768);
        glGetShaderInfoLog(VSHandle,32768,NULL,log);
        printf("Compile Log: %s\n", log);

        /* Attach and link our program */
        glAttachShader(s->program, FSHandle);
        glAttachShader(s->program, VSHandle);
        glLinkProgram(s->program);

        /* Print link log. */
        memset(log, 0, 32768);
        glGetProgramInfoLog(s->program, 32768, NULL, log);
        printf("Link Log: %s\n", log);
        free(log);

        if(s->luts_to_apply) {
                assert(s->luts_to_apply->next  == 0); // not yet implemented

                assert(s->luts_to_apply->type == LUT_3D_MATRIX); //only implemented by now

                if(s->luts_to_apply->type == LUT_3D_MATRIX) {
                        GLfloat matrix[16];

                        for(int x = 0; x < 4; ++x) {
                                for(int y = 0; y < 4; ++y) {
                                        if(x == 3 || y == 3) {
                                                matrix[x + y * 4] = 0.0;
                                        } else {
                                                matrix[x + y * 4] = ((double *)s->luts_to_apply->lut)[x + y * 3];
                                        }
                                }
                        }


                        glUseProgram(s->program);
                        glUniformMatrix4fv(glGetUniformLocationARB(s->program,"matrix"), 1, /*transpose*/ GL_TRUE, (const GLfloat *) &matrix);
                        glUseProgram(0);
                }
        }

        glUseProgram(s->program);
        glUniform1iARB(glGetUniformLocationARB(s->program,"image"),0);
        glUseProgram(0);
}


static void configure(struct state_color_transform *s, struct video_frame *tx) {
        glGenFramebuffers(1, &s->fbo_id);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glGenTextures(1, &s->tex_input);
        glBindTexture(GL_TEXTURE_2D, s->tex_input);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        switch(tx->color_spec) {
                case RGB:
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tx->tiles[0].width, tx->tiles[0].height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
                        break;
                case RGBA:
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tx->tiles[0].width, tx->tiles[0].height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                        break;
                case RGB16:
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tx->tiles[0].width, tx->tiles[0].height, 0, GL_RGB, GL_UNSIGNED_SHORT, NULL);
                        break;
                default:
                        fprintf(stderr, "%s:%d: Unsupported video codec %d.\n", __FILE__, __LINE__, tx->color_spec);
                        abort();
        }

        glGenTextures(1, &s->tex_processed);
        glBindTexture(GL_TEXTURE_2D, s->tex_processed);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tx->tiles[0].width, tx->tiles[0].height,
                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        s->out = vf_alloc(1);
        s->out->color_spec = RGBA;
        s->out->interlacing = tx->interlacing;
        s->out->fps = tx->fps;
        s->out->colorspace = RGB_709_D65;

        s->tile = vf_get_tile(s->out, 0);

        s->tile->width  = tx->tiles[0].width;
        s->tile->height = tx->tiles[0].height;
        s->tile->storage = OPENGL_TEXTURE;
        s->tile->texture = s->tex_processed;

        glViewport(0, 0, s->tile->width, s->tile->height);

        s->luts_to_apply = tx->luts_to_apply;

        build_programs(s);

        s->configured = TRUE;
}

struct state_color_transform * color_transform_init(struct gl_context *context)
{
        struct state_color_transform *s;
        
        s = (struct state_color_transform *) malloc(sizeof(struct state_color_transform));
        assert (s != NULL);
        s->context = context;

        s->configured = FALSE;

        return s;
}

struct video_frame * color_transform_transform(struct state_color_transform *s, struct video_frame * tx)
{
        GLenum format;
        GLenum type;

        if(!s->configured) {
                configure(s, tx);
        }

        switch(tx->color_spec) {
                case RGB:
                        format = GL_RGB;
                        type = GL_UNSIGNED_BYTE;
                        break;
                case RGBA:
                        format = GL_RGBA;
                        type = GL_UNSIGNED_BYTE;
                        break;
                case RGB16:
                        format = GL_RGB;
                        type = GL_UNSIGNED_SHORT;
                        break;
                default:
                        fprintf(stderr, "%s:%d: Unsupported video codec %d.\n", __FILE__, __LINE__, tx->color_spec);
                        abort();
        }

        glUseProgram(s->program);

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

        glBindTexture(GL_TEXTURE_2D, s->tex_input);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tx->tiles[0].width, tx->tiles[0].height,  format, type, tx->tiles[0].data);

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
        glBindTexture(GL_TEXTURE_2D, 0);


        glPopAttrib();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glUseProgram(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        return s->out;
}

void color_transform_done(struct state_color_transform *s)
{
}

