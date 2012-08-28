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
#include "config_win32.h"
#endif // HAVE_CONFIG_H
#include "debug.h"
#include "host.h"
#include "video_compress/dxt_glsl.h"
#include "dxt_compress/dxt_encoder.h"
#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>

#include <GL/glew.h>

#ifdef HAVE_MACOSX
#include "mac_gl_common.h"
#else
#include "x11_common.h"
#include "glx_common.h"
#endif

#include "gl_context.h"

static const char fp_display_rgba_to_yuv422_legacy[] = 
"#define GL_legacy 1\n"
    "#if GL_legacy\n"
    "#define TEXCOORD gl_TexCoord[0]\n"
    "#else\n"
    "#define TEXCOORD TEX0\n"
    "#define texture2D texture\n"
    "#endif\n"
    "\n"
    "#if GL_legacy\n"
    "#define colorOut gl_FragColor\n"
    "#else\n"
    "out vec4 colorOut;\n"
    "#endif\n"
    "\n"
    "#if ! GL_legacy\n"
    "in vec4 TEX0;\n"
    "#endif\n"
    "\n"
    "uniform sampler2D image;\n"
    "uniform float imageWidth; // is original image width, it means twice as wide as ours\n"
    "\n"
    "void main()\n"
    "{\n"
    "        vec4 rgba1, rgba2;\n"
    "        vec4 yuv1, yuv2;\n"
    "        vec2 coor1, coor2;\n"
    "        float U, V;\n"
    "\n"
    "        coor1 = TEXCOORD.xy - vec2(1.0 / (imageWidth * 2.0), 0.0);\n"
    "        coor2 = TEXCOORD.xy + vec2(1.0 / (imageWidth * 2.0), 0.0);\n"
    "\n"
    "        rgba1  = texture2D(image, coor1);\n"
    "        rgba2  = texture2D(image, coor2);\n"
    "        \n"
    "        yuv1.x = 1.0/16.0 + (rgba1.r * 0.2126 + rgba1.g * 0.7152 + rgba1.b * 0.0722) * 0.8588; // Y\n"
    "        yuv1.y = 0.5 + (-rgba1.r * 0.1145 - rgba1.g * 0.3854 + rgba1.b * 0.5) * 0.8784;\n"
    "        yuv1.z = 0.5 + (rgba1.r * 0.5 - rgba1.g * 0.4541 - rgba1.b * 0.0458) * 0.8784;\n"
    "        \n"
    "        yuv2.x = 1.0/16.0 + (rgba2.r * 0.2126 + rgba2.g * 0.7152 + rgba2.b * 0.0722) * 0.8588; // Y\n"
    "        yuv2.y = 0.5 + (-rgba2.r * 0.1145 - rgba2.g * 0.3854 + rgba2.b * 0.5) * 0.8784;\n"
    "        yuv2.z = 0.5 + (rgba2.r * 0.5 - rgba2.g * 0.4541 - rgba2.b * 0.0458) * 0.8784;\n"
    "        \n"
    "        U = mix(yuv1.y, yuv2.y, 0.5);\n"
    "        V = mix(yuv1.z, yuv2.z, 0.5);\n"
    "        \n"
    "        colorOut = vec4(U,yuv1.x, V, yuv2.x);\n"
    "}\n"
;


struct none_video_compress {
        struct video_frame *out;
        struct tile *tile;
        unsigned int configured:1;

        struct gl_context *context;

        GLuint program_rgba_to_yuv422;

        GLuint fbo_rgba;
        GLuint texture_rgba;
        GLuint fbo;
        GLuint texture;
};

void * none_compress_init(char * opts, struct gl_context *context)
{
        struct none_video_compress *s;
        
        s = (struct none_video_compress *) malloc(sizeof(struct none_video_compress));
        s->out = vf_alloc(1);
        s->tile = vf_get_tile(s->out, 0);

        glewInit();
        s->context = context;


        const GLchar *VProgram, *FProgram;
        char            *log;
        GLuint     VSHandle,FSHandle,PHandle;

        int len;
        GLsizei gllen;

        
        FProgram = (const GLchar *) fp_display_rgba_to_yuv422_legacy;
        /* Set up program objects. */
        s->program_rgba_to_yuv422 = glCreateProgram();
        FSHandle=glCreateShader(GL_FRAGMENT_SHADER);
        
        /* Compile Shader */
        len = strlen(FProgram);
        glShaderSource(FSHandle, 1, &FProgram, &len);
        glCompileShader(FSHandle);
        
        /* Print compile log */
        log = calloc(32768,sizeof(char));
        glGetShaderInfoLog(FSHandle, 32768, &gllen, log);
        printf("Compile Log: %s\n", log);
#if 0
        glShaderSource(VSHandle,1, &VProgram,NULL);
        glCompileShaderARB(VSHandle);
        memset(log, 0, 32768);
        glGetInfoLogARB(VSHandle,32768, &gllen,log);
        printf("Compile Log: %s\n", log);

        /* Attach and link our program */
        glAttachObjectARB(PHandle,VSHandle);
#endif
        glAttachShader(PHandle, FSHandle);
        glLinkProgram(PHandle);
        
        /* Print link log. */
        memset(log, 0, 32768);
        glGetInfoLogARB(PHandle,32768,NULL,log);
        printf("Link Log: %s\n", log);
        free(log);


        s->configured = FALSE;

        return s;
}

int none_configure_with(struct none_video_compress *s, struct video_frame *tx)
{
        s->out->color_spec = UYVY;
        s->out->interlacing = tx->interlacing;
        s->out->fps = tx->fps;

        s->tile->width = tx->tiles[0].width;
        s->tile->height = tx->tiles[0].height;
        s->tile->data_len = 2 * s->tile->width * s->tile->height; /* RGBA */
        s->tile->data = malloc(s->tile->data_len);

        glEnable(GL_TEXTURE_2D);

        glGenFramebuffers(1, &s->fbo);
        glGenFramebuffers(1, &s->fbo_rgba);

        glGenTextures(1, &s->texture_rgba); 
        glBindTexture(GL_TEXTURE_2D, s->texture); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGBA, s->tile->width, s->tile->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0); 

        glGenTextures(1, &s->texture); 
        glBindTexture(GL_TEXTURE_2D, s->texture); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGBA, s->tile->width / 2, s->tile->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0); 

        glBindTexture(GL_TEXTURE_2D, 0);

        glClearColor(1.0,1.0,0,1.0);

        s->configured = TRUE;
}

struct video_frame * none_compress(void *arg, struct video_frame * tx)
{
        struct none_video_compress *s = (struct none_video_compress *) arg;

        assert(tx->tiles[0].storage == OPENGL_TEXTURE);

        GLuint tex_source = tx->tiles[0].texture;
        
        if(!s->configured) {
                int ret;
                ret = none_configure_with(s, tx);
                if(!ret)
                        return NULL;
        }

        glUseProgram(0);

        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->texture, 0);

        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport( 0, 0, s->tile->width, s->tile->height);
        
        glMatrixMode( GL_PROJECTION );
        glPushMatrix();
        glLoadIdentity( );
        glOrtho(-1,1,-1,1,10,-10);
        
        glMatrixMode( GL_MODELVIEW );
        glPushMatrix();
        glLoadIdentity( );


        glBindTexture(GL_TEXTURE_2D, tex_source);

        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        
        glBegin(GL_QUADS);
                glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
                glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
                glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
                glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
        glEnd();
        //glClear(GL_COLOR_BUFFER_BIT);


        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->texture, 0);

        glBindTexture(GL_TEXTURE_2D, s->texture_rgba); /* to texturing unit 0 */

        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport( 0, 0, s->tile->width / 2, s->tile->height);

        glUseProgram(s->program_rgba_to_yuv422);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
        glEnd();

        // Read back
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, s->tile->width / 2, s->tile->height, GL_RGBA, GL_UNSIGNED_BYTE, s->tile->data);

        glPopAttrib();
        glMatrixMode( GL_PROJECTION );
        glPopMatrix();
        glMatrixMode( GL_MODELVIEW );
        glPopMatrix();

        glPopAttrib();
        glMatrixMode( GL_PROJECTION );
        glPopMatrix();
        glMatrixMode( GL_MODELVIEW );
        glPopMatrix();

        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        s->out->frames = tx->frames;

        return s->out;
}

void none_compress_done(void *arg)
{
}
