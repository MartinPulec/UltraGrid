/*
 * FILE:    screen.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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


#include "host.h"
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "debug.h"
#include "video_codec.h"
#include "video_capture.h"

#include "tv.h"

#include "video_capture/screen.h"
#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "video_display.h"
#include "video.h"

#include <pthread.h>

#ifdef HAVE_MACOSX
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <Carbon/Carbon.h>
#else
#include <GL/glew.h>
#include <GL/glx.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include "x11_common.h"
#include "gl_context.h"
#endif

#define GL 1
typedef void glXBindTexImageEXTProc_t (Display *dpy, GLXDrawable drawable, int buffer, const int *attrib_list);
glXBindTexImageEXTProc_t (*glXBindTexImageEXTProc) = NULL;

/* prototypes of functions defined in this module */
static void show_help(void);
#ifdef HAVE_LINUX
static void *grab_thread(void *args);
#endif // HAVE_LINUX

static volatile bool should_exit = false;

static void show_help()
{
        printf("Screen capture\n");
        printf("Usage\n");
        printf("\t-t screen[:fps=<fps>]\n");
        printf("\t\t<fps> - preferred grabbing fps (otherwise unlimited)\n");
}

/* defined in main.c */
extern int uv_argc;
extern char **uv_argv;

static struct vidcap_screen_state *state;

struct vidcap_screen_state {
        struct video_frame       *frame; 
        struct tile       *tile; 
        int frames;
        struct       timeval t, t0;
#ifdef HAVE_MACOSX
        CGDirectDisplayID display;
#else
        Display *dpy;
        Window root;

        char *buffer[2];
        int buffer_net;
        pthread_mutex_t lock;
        pthread_cond_t boss_cv;
        volatile bool boss_waiting;
        pthread_cond_t worker_cv;
        volatile bool worker_waiting;
        volatile bool process_item;

        volatile bool should_exit_worker;

        pthread_t worker_id;

        Pixmap pixmap;
        GLXPixmap glxpixmap;
        struct gl_context context;
        GLuint tex;
        GLuint tex_out;
        GLuint fbo;
        double top, bottom;
#endif

        struct timeval prev_time;

        double fps;

};

pthread_once_t initialized = PTHREAD_ONCE_INIT;

static void initialize() {
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);


#ifndef HAVE_MACOSX
        XWindowAttributes wa;

        x11_enter_thread();

        x11_lock();

        s->dpy = x11_acquire_display();

        x11_unlock();

        s->root = DefaultRootWindow(s->dpy);

        XGetWindowAttributes(s->dpy, DefaultRootWindow(s->dpy), &wa);
        s->tile->width = wa.width;
        s->tile->height = wa.height;

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->boss_cv, NULL);
        pthread_cond_init(&s->worker_cv, NULL);
        s->buffer_net = 1;

        s->worker_waiting = false;
        s->boss_waiting = false;
        s->process_item = true; // start it

        s->should_exit_worker = false;


        if(!init_gl_context(&s->context, GL_CONTEXT_ANY)) {
                abort();
        }

        GLenum err = glewInit();
        if (GLEW_OK != err)
        {
                /* Problem: glewInit failed, something is seriously wrong. */
                fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
                abort();
        }

        glEnable(GL_TEXTURE_2D);

        glGenTextures(1, &state->tex);
        glBindTexture(GL_TEXTURE_2D, state->tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, state->tile->width, state->tile->height,
                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glGenTextures(1, &state->tex_out);
        glBindTexture(GL_TEXTURE_2D, state->tex_out);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, state->tile->width, state->tile->height,
                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glBindTexture(GL_TEXTURE_2D, 0);
        glGenFramebuffersEXT(1, &state->fbo);

        VisualID visualid = XVisualIDFromVisual (wa.visual);
        int nfbconfigs;
        int i;
        XVisualInfo *visinfo;
        int value;
        GLXFBConfig *fbconfigs = glXGetFBConfigs (s->dpy, DefaultScreen(s->dpy), &nfbconfigs);
        for (i = 0; i < nfbconfigs; i++)
        {
                visinfo = glXGetVisualFromFBConfig (s->dpy, fbconfigs[i]);
                if (!visinfo || visinfo->visualid != visualid)
                        continue;

                glXGetFBConfigAttrib (s->dpy, fbconfigs[i], GLX_DRAWABLE_TYPE, &value);
                if (!(value & GLX_PIXMAP_BIT))
                        continue;

                glXGetFBConfigAttrib (s->dpy, fbconfigs[i],
                                GLX_BIND_TO_TEXTURE_TARGETS_EXT,
                                &value);
                if (!(value & GLX_TEXTURE_2D_BIT_EXT))
                        continue;

                glXGetFBConfigAttrib (s->dpy, fbconfigs[i],
                                GLX_BIND_TO_TEXTURE_RGBA_EXT,
                                &value);
                if (value == FALSE)
                {
                        glXGetFBConfigAttrib (s->dpy, fbconfigs[i],
                                        GLX_BIND_TO_TEXTURE_RGB_EXT,
                                        &value);
                        if (value == FALSE)
                                continue;
                }

                glXGetFBConfigAttrib (s->dpy, fbconfigs[i],
                                GLX_Y_INVERTED_EXT,
                                &value);
                if (value == TRUE)
                {
                        s->top = 0.0f;
                        s->bottom = 1.0f;
                }
                else
                {
                        s->top = 1.0f;
                        s->bottom = 0.0f;
                }

                break;
        }

        if (i == nfbconfigs) {
                abort();
        }

        s->pixmap = XCompositeNameWindowPixmap (s->dpy, s->root);
        const int pixmapAttribs[] = { GLX_TEXTURE_TARGET_EXT, GLX_TEXTURE_2D_EXT,
                GLX_TEXTURE_FORMAT_EXT, GLX_TEXTURE_FORMAT_RGBA_EXT,
                None };
        s->glxpixmap = glXCreatePixmap (s->dpy, fbconfigs[i], s->pixmap, pixmapAttribs);

        glXBindTexImageEXTProc = glXGetProcAddressARB( (const GLubyte *) "glXBindTexImageEXT");
        assert(glXBindTexImageEXTProc != NULL);

        glBindTexture (GL_TEXTURE_2D, s->tex);

        glXBindTexImageEXTProc (s->dpy, s->glxpixmap, GLX_FRONT_LEFT_EXT, NULL);
        abort();

#else
        s->display = CGMainDisplayID();
        CGImageRef image = CGDisplayCreateImage(s->display);

        s->tile->width = CGImageGetWidth(image);
        s->tile->height = CGImageGetHeight(image);
        CFRelease(image);
#endif

        s->frame->color_spec = RGBA;
        if(s->fps > 0.0) {
                s->frame->fps = s->fps;
        } else {
                s->frame->fps = 30;
        }
        s->frame->interlacing = PROGRESSIVE;
        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) * s->tile->height;

#ifndef HAVE_MACOSX
#ifndef GL
        s->buffer[0] = (char *) malloc(s->tile->data_len);
        s->buffer[1] = (char *) malloc(s->tile->data_len);

        pthread_create(&s->worker_id, NULL, grab_thread, s);
#endif // ! GL
#else
        s->tile->data = (char *) malloc(s->tile->data_len);
#endif

        return;

        goto error; // dummy use (otherwise compiler would complain about unreachable code (Mac)
error:
        fprintf(stderr, "[Screen cap.] Initialization failed!\n");
        exit_uv(128);
}


#ifdef HAVE_LINUX
static void *grab_thread(void *args)
{
        struct vidcap_screen_state *s = args;

        while(!s->should_exit_worker) {
                pthread_mutex_lock(&s->lock);
                while(!s->process_item) {
                        s->worker_waiting = true;
                        pthread_cond_wait(&s->worker_cv, &s->lock);
                        s->worker_waiting = false;
                }

                XImage *image = XGetImage(s->dpy,s->root, 0,0, s->tile->width, s->tile->height, AllPlanes, ZPixmap);

                /*
                 * The more correct way is to use X pixel accessor (XGetPixel) as in previous version
                 * Unfortunatelly, this approach is damn slow. Current approach might be incorrect in
                 * some configurations, but seems to work currently. To be corrected if there is an
                 * opposite case.
                 */
                vc_copylineRGBA((unsigned char *) s->buffer[(s->buffer_net + 1) % 2],
                                (unsigned char *) &image->data[0], s->tile->data_len, 16, 8, 0);

                XDestroyImage(image);

                s->process_item = false;

                if(s->boss_waiting)
                        pthread_cond_signal(&s->boss_cv);

                pthread_mutex_unlock(&s->lock);
        }

        return NULL;
}
#endif // HAVE_LINUX

struct vidcap_type * vidcap_screen_probe(void)
{
        struct vidcap_type*		vt;

        vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id          = VIDCAP_SCREEN_ID;
                vt->name        = "screen";
                vt->description = "Grabbing screen";
        }
        return vt;
}

void * vidcap_screen_init(char *init_fmt, unsigned int flags)
{
        struct vidcap_screen_state *s;

        printf("vidcap_screen_init\n");

        UNUSED(flags);


        state = s = (struct vidcap_screen_state *) malloc(sizeof(struct vidcap_screen_state));
        if(s == NULL) {
                printf("Unable to allocate screen capture state\n");
                return NULL;
        }

        gettimeofday(&s->t0, NULL);

        s->fps = 0.0;

        s->frame = NULL;
        s->tile = NULL;

#ifdef HAVE_LINUX
        s->worker_id = 0;
        s->buffer[0] = NULL;
        s->buffer[1] = NULL;
#endif

        s->prev_time.tv_sec = 
                s->prev_time.tv_usec = 0;


        s->frames = 0;

        if(init_fmt) {
                if (strcmp(init_fmt, "help") == 0) {
                        show_help();
                        return NULL;
                } else if (strncasecmp(init_fmt, "fps=", strlen("fps=")) == 0) {
                        s->fps = atoi(init_fmt + strlen("fps="));
                }
        }

        return s;
}

void vidcap_screen_finish(void *state)
{
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        assert(s != NULL);
        should_exit = true;
#ifdef HAVE_LINUX
        pthread_mutex_lock(&s->lock);
        if(s->boss_waiting) {
                pthread_cond_signal(&s->boss_cv);
        }

        s->should_exit_worker = true;
        if(s->worker_waiting) {
                s->process_item = true; // get out of loop
                pthread_cond_signal(&s->worker_cv);
        }

        pthread_mutex_unlock(&s->lock);
#endif // HAVE_LINUX
}

void vidcap_screen_done(void *state)
{
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        assert(s != NULL);
#ifdef HAVE_LINUX
        if(s->worker_id) {
                pthread_join(s->worker_id, NULL);
        }

        free(s->buffer[0]);
        free(s->buffer[1]);
#endif

        if(s->tile) {
#ifdef HAVE_MACOS_X
                free(s->tile->data);
#endif
        }
        vf_free(s->frame);
        free(s);
}

struct video_frame * vidcap_screen_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        pthread_once(&initialized, initialize);

        *audio = NULL;

#ifndef HAVE_MACOSX
#ifdef GL
#else // GL
        pthread_mutex_lock(&s->lock);

        if(should_exit) {
                pthread_mutex_unlock(&s->lock);
                return NULL;
        }

        while(s->process_item) {
                s->boss_waiting = true;
                pthread_cond_wait(&s->boss_cv, &s->lock);
                s->boss_waiting = false;
        }
        
        s->buffer_net = (s->buffer_net + 1) % 2;
        s->tile->data = s->buffer[s->buffer_net];

        s->process_item = true;
        if(s->worker_waiting)
                pthread_cond_signal(&s->worker_cv);
        pthread_mutex_unlock(&s->lock);
#endif // GL
#else
        CGImageRef image = CGDisplayCreateImage(s->display);
        CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(image));
        const unsigned char *pixels = CFDataGetBytePtr(data);

        int linesize = s->tile->width * 4;
        int y;
        unsigned char *dst = (unsigned char *) s->tile->data;
        const unsigned char *src = (const unsigned char *) pixels;
        for(y = 0; y < (int) s->tile->height; ++y) {
                vc_copylineRGBA (dst, src, linesize, 16, 8, 0);
                src += linesize;
                dst += linesize;
        }

        CFRelease(data);
        CFRelease(image);

#endif

        if(s->fps > 0.0) {
                struct timeval cur_time;

                gettimeofday(&cur_time, NULL);
                while(tv_diff_usec(cur_time, s->prev_time) < 1000000.0 / s->frame->fps) {
                        gettimeofday(&cur_time, NULL);
                }
                s->prev_time = cur_time;
        }

        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);        
        if (seconds >= 5) {
                float fps  = s->frames / seconds;
                fprintf(stderr, "[screen capture] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->t0 = s->t;
                s->frames = 0;
        }

        s->frames++;

        return s->frame;
}

