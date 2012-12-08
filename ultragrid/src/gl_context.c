#include <stdio.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif

#ifdef HAVE_MACOSX
#include "mac_gl_common.h"
#else
#include <GL/glew.h>
#include "x11_common.h"
#include "glx_common.h"
#endif

#include "gl_context.h"

/*
 * TODO: remove this and reenable the latter one
 */
void init_gl_context(struct gl_context *context) {
#ifndef HAVE_MACOSX
        x11_enter_thread();
        context->legacy = TRUE;
        if(context) {
        }
#else
        context->context = NULL;
        context->context = mac_gl_init(MAC_GL_PROFILE_LEGACY);
        context->legacy = TRUE;
#endif
}

/*
 * TODO: reenable after there are OpenGL 3.x renderers
 */
#if 0
void init_gl_context(struct gl_context *context) {
#ifndef HAVE_MACOSX
        x11_enter_thread();
        printf("Trying OpenGL 3.1 first.\n");
        context->context = glx_init(MK_OPENGL_VERSION(3,1));
        context->legacy = FALSE;
        if(!context->context) {
                fprintf(stderr, "[RTDXT] OpenGL 3.1 profile failed to initialize, falling back to legacy profile.\n");
                context->context = glx_init(OPENGL_VERSION_UNSPECIFIED);
                context->legacy = TRUE;
        }
        glx_validate(context->context);
#else
        context->context = NULL;
        if(get_mac_kernel_version_major() >= 11) {
                printf("[RTDXT] Mac 10.7 or latter detected. Trying OpenGL 3.2 Core profile first.\n");
                context->context = mac_gl_init(MAC_GL_PROFILE_3_2);
                if(!context->context) {
                        fprintf(stderr, "[RTDXT] OpenGL 3.2 Core profile failed to initialize, falling back to legacy profile.\n");
                } else {
                        context->legacy = FALSE;
                }
        }

        if(!context->context) {
                context->context = mac_gl_init(MAC_GL_PROFILE_LEGACY);
                context->legacy = TRUE;
        }
#endif
}
#endif


void destroy_gl_context(struct gl_context *context) {
#ifdef HAVE_MACOSX
        mac_gl_free(context->context);
#else
#endif
}

