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

#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "dxt_glsl_compress.h"
#include "dxt_compress/dxt_encoder.h"
#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include "x11_common.h"

/*
 * GLX context creation overtaken from:
 * http://www.opengl.org/wiki/Tutorial:_OpenGL_3.0_Context_Creation_%28GLX%29
 */

struct video_compress {
        struct dxt_encoder *encoder;

        struct video_frame *out;
        decoder_t decoder;
        char *decoded;
        unsigned int configured:1;
        unsigned int interlaced_input:1;
        codec_t color_spec;
};

static void configure_with(struct video_compress *s, struct video_frame *frame);
static int isExtensionSupported(const char *extList, const char *extension);
static int ctxErrorHandler( Display *dpy, XErrorEvent *ev );

#define GLX_CONTEXT_MAJOR_VERSION_ARB       0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB       0x2092
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
 
// Helper to check for extension string presence.  Adapted from:
//   http://www.opengl.org/resources/features/OGLextensions/
static int isExtensionSupported(const char *extList, const char *extension)
{
  const char *start;
  const char *where, *terminator;
 
  /* Extension names should not have spaces. */
  where = strchr(extension, ' ');
  if ( where || *extension == '\0' )
    return FALSE;
 
  /* It takes a bit of care to be fool-proof about parsing the
     OpenGL extensions string. Don't be fooled by sub-strings,
     etc. */
  for ( start = extList; ; ) {
    where = strstr( start, extension );
 
    if ( !where )
      break;
 
    terminator = where + strlen( extension );
 
    if ( where == start || *(where - 1) == ' ' )
      if ( *terminator == ' ' || *terminator == '\0' )
        return TRUE;
 
    start = terminator;
  }
 
  return FALSE;
}
 
static int ctxErrorOccurred = FALSE;
static int ctxErrorHandler( Display *dpy, XErrorEvent *ev )
{
    ctxErrorOccurred = TRUE;
    return 0;
}

void glx_init()
{
        Display *display = XOpenDisplay(0);
 
  if ( !display )
  {
    printf( "Failed to open X display\n" );
    exit(1);
  }
 
  // Get a matching FB config
  static int visual_attribs[] =
    {
      GLX_X_RENDERABLE    , True,
      GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
      GLX_RENDER_TYPE     , GLX_RGBA_BIT,
      GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
      GLX_RED_SIZE        , 8,
      GLX_GREEN_SIZE      , 8,
      GLX_BLUE_SIZE       , 8,
      GLX_ALPHA_SIZE      , 8,
      GLX_DEPTH_SIZE      , 24,
      GLX_STENCIL_SIZE    , 8,
      GLX_DOUBLEBUFFER    , True,
      //GLX_SAMPLE_BUFFERS  , 1,
      //GLX_SAMPLES         , 4,
      None
    };
 
  int glx_major, glx_minor;
 
  // FBConfigs were added in GLX version 1.3.
  if ( !glXQueryVersion( display, &glx_major, &glx_minor ) || 
       ( ( glx_major == 1 ) && ( glx_minor < 3 ) ) || ( glx_major < 1 ) )
  {
    printf( "Invalid GLX version" );
    exit(1);
  }
 
  printf( "Getting matching framebuffer configs\n" );
  int fbcount;
  GLXFBConfig *fbc = glXChooseFBConfig( display, DefaultScreen( display ), 
                                        visual_attribs, &fbcount );
  if ( !fbc )
  {
    printf( "Failed to retrieve a framebuffer config\n" );
    exit(1);
  }
  printf( "Found %d matching FB configs.\n", fbcount );
 
  // Pick the FB config/visual with the most samples per pixel
  printf( "Getting XVisualInfos\n" );
  int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;
 
  int i;
  for ( i = 0; i < fbcount; i++ )
  {
    XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[i] );
    if ( vi )
    {
      int samp_buf, samples;
      glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
      glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLES       , &samples  );
 
      printf( "  Matching fbconfig %d, visual ID 0x%2x: SAMPLE_BUFFERS = %d,"
              " SAMPLES = %d\n", 
              i, vi -> visualid, samp_buf, samples );
 
      if ( best_fbc < 0 || samp_buf && samples > best_num_samp )
        best_fbc = i, best_num_samp = samples;
      if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
        worst_fbc = i, worst_num_samp = samples;
    }
    XFree( vi );
  }
 
  GLXFBConfig bestFbc = fbc[ best_fbc ];
 
  // Be sure to free the FBConfig list allocated by glXChooseFBConfig()
  XFree( fbc );
 
  // Get a visual
  XVisualInfo *vi = glXGetVisualFromFBConfig( display, bestFbc );
  printf( "Chosen visual ID = 0x%x\n", vi->visualid );
 
  printf( "Creating colormap\n" );
  XSetWindowAttributes swa;
  Colormap cmap;
  swa.colormap = cmap = XCreateColormap( display,
                                         RootWindow( display, vi->screen ), 
                                         vi->visual, AllocNone );
  swa.background_pixmap = None ;
  swa.border_pixel      = 0;
  swa.event_mask        = StructureNotifyMask;
 
  printf( "Creating window\n" );
  Window win = XCreateWindow( display, RootWindow( display, vi->screen ), 
                              0, 0, 100, 100, 0, vi->depth, InputOutput, 
                              vi->visual, 
                              CWBorderPixel|CWColormap|CWEventMask, &swa );
  if ( !win )
  {
    printf( "Failed to create window.\n" );
    exit(1);
  }
 
  // Done with the visual info data
  XFree( vi );
 
 /* We don't need this for UG */
  /*XStoreName( display, win, "GL 3.0 Window" );
 
  printf( "Mapping window\n" );
  XMapWindow( display, win );*/
 
  // Get the default screen's GLX extension list
  const char *glxExts = glXQueryExtensionsString( display,
                                                  DefaultScreen( display ) );
 
  // NOTE: It is not necessary to create or make current to a context before
  // calling glXGetProcAddressARB
  glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
  glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
           glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );
 
  GLXContext ctx = 0;
 
  // Install an X error handler so the application won't exit if GL 3.0
  // context allocation fails.
  //
  // Note this error handler is global.  All display connections in all threads
  // of a process use the same error handler, so be sure to guard against other
  // threads issuing X commands while this code is running.
  ctxErrorOccurred = FALSE;
  int (*oldHandler)(Display*, XErrorEvent*) =
      XSetErrorHandler(&ctxErrorHandler);
 
  // Check for the GLX_ARB_create_context extension string and the function.
  // If either is not present, use GLX 1.3 context creation method.
  if ( !isExtensionSupported( glxExts, "GLX_ARB_create_context" ) ||
       !glXCreateContextAttribsARB )
  {
    printf( "glXCreateContextAttribsARB() not found"
            " ... using old-style GLX context\n" );
    ctx = glXCreateNewContext( display, bestFbc, GLX_RGBA_TYPE, 0, True );
  }
 
  // If it does, try to get a GL 3.0 context!
  else
  {
    int context_attribs[] =
      {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 0,
        //GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
        None
      };
 
    printf( "Creating context\n" );
    ctx = glXCreateContextAttribsARB( display, bestFbc, 0,
                                      True, context_attribs );
 
    // Sync to ensure any errors generated are processed.
    XSync( display, False );
    if ( !ctxErrorOccurred && ctx )
      printf( "Created GL 3.0 context\n" );
    else
    {
      // Couldn't create GL 3.0 context.  Fall back to old-style 2.x context.
      // When a context version below 3.0 is requested, implementations will
      // return the newest context version compatible with OpenGL versions less
      // than version 3.0.
      // GLX_CONTEXT_MAJOR_VERSION_ARB = 1
      context_attribs[1] = 1;
      // GLX_CONTEXT_MINOR_VERSION_ARB = 0
      context_attribs[3] = 0;
 
      ctxErrorOccurred = FALSE;
 
      printf( "Failed to create GL 3.0 context"
              " ... using old-style GLX context\n" );
      ctx = glXCreateContextAttribsARB( display, bestFbc, 0, 
                                        True, context_attribs );
    }
  }
 
  // Sync to ensure any errors generated are processed.
  XSync( display, False );
 
  // Restore the original error handler
  XSetErrorHandler( oldHandler );
 
  if ( ctxErrorOccurred || !ctx )
  {
    printf( "Failed to create an OpenGL context\n" );
    exit(1);
  }
 
  // Verifying that context is a direct context
  if ( ! glXIsDirect ( display, ctx ) )
  {
    printf( "Indirect GLX rendering context obtained\n" );
  }
  else
  {
    printf( "Direct GLX rendering context obtained\n" );
  }
 
  printf( "Making context current\n" );
  glXMakeCurrent( display, win, ctx );

  glewInit();
}

static void configure_with(struct video_compress *s, struct video_frame *frame)
{
        int i;
        int x, y;
	int h_align = 0;
        enum dxt_format format;
        
        assert(tile_get(frame, 0, 0)->width % 4 == 0 && tile_get(frame, 0, 0)->height % 4 == 0);
        s->out = vf_alloc(frame->grid_width, frame->grid_height);
        
        for (x = 0; x < frame->grid_width; ++x) {
                for (y = 0; y < frame->grid_height; ++y) {
                        if (tile_get(frame, x, y)->width != tile_get(frame, 0, 0)->width ||
                                        tile_get(frame, x, y)->width != tile_get(frame, 0, 0)->width)
                                error_with_code_msg(128,"[RTDXT] Requested to compress tiles of different size!");
                                
                        tile_get(s->out, x, y)->width = tile_get(frame, 0, 0)->width;
                        tile_get(s->out, x, y)->height = tile_get(frame, 0, 0)->height;
                }
        }
        
        glx_init();
        
        s->out->aux = frame->aux;
        s->out->fps = frame->fps;
        s->out->color_spec = s->color_spec;

        switch (frame->color_spec) {
                case RGB:
                        s->decoder = (decoder_t) memcpy;
                        format = DXT_FORMAT_RGB;
                        break;
                case RGBA:
                        s->decoder = (decoder_t) memcpy;
                        format = DXT_FORMAT_RGBA;
                        break;
                case R10k:
                        s->decoder = (decoder_t) vc_copyliner10k;
                        format = DXT_FORMAT_RGBA;
                        break;
                case UYVY:
                case Vuy2:
                case DVS8:
                        s->decoder = (decoder_t) memcpy;
                        format = DXT_FORMAT_YUV422;
                        break;
                case v210:
                        s->decoder = (decoder_t) vc_copylinev210;
                        format = DXT_FORMAT_YUV422;
                        break;
                case DVS10:
                        s->decoder = (decoder_t) vc_copylineDVS10;
                        format = DXT_FORMAT_YUV422;
                        break;
                case DXT1:
                case DXT5:
                        fprintf(stderr, "Input frame is already comperssed!");
                        exit(128);
                default:
                        fprintf(stderr, "Unknown codec: %d\n", frame->color_spec);
                        exit(128);
        }
        
        

        /* We will deinterlace the output frame */
        if(s->out->aux & AUX_INTERLACED)
                s->interlaced_input = TRUE;
        else
                s->interlaced_input = FALSE;
        s->out->aux &= ~AUX_INTERLACED;
        
        if(s->out->color_spec == DXT1) {
                s->encoder = dxt_encoder_create(DXT_TYPE_DXT1, s->out->tiles[0].width, s->out->tiles[0].height, format);
                s->out->aux |= AUX_RGB;
                s->out->tiles[0].data_len = s->out->tiles[0].width * s->out->tiles[0].height / 2;
        } else if(s->out->color_spec == DXT5){
                s->encoder = dxt_encoder_create(DXT_TYPE_DXT5_YCOCG, s->out->tiles[0].width, s->out->tiles[0].height, format);
                s->out->aux |= AUX_YUV; /* YCoCg */
                s->out->tiles[0].data_len = s->out->tiles[0].width * s->out->tiles[0].height;
        }
        
        for (x = 0; x < frame->grid_width; ++x) {
                for (y = 0; y < frame->grid_height; ++y) {
                        tile_get(s->out, x, y)->linesize = s->out->tiles[0].width;
                        switch(format) { 
                                case DXT_FORMAT_RGBA:
                                        tile_get(s->out, x, y)->linesize *= 4;
                                        break;
                                case DXT_FORMAT_RGB:
                                        tile_get(s->out, x, y)->linesize *= 3;
                                        break;
                                case DXT_FORMAT_YUV422:
                                        tile_get(s->out, x, y)->linesize *= 2;
                                        break;
                        }
                        tile_get(s->out, x, y)->data_len = s->out->tiles[0].data_len;
                        tile_get(s->out, x, y)->data = (char *) malloc(s->out->tiles[0].data_len);
                }
        }
        
        if(!s->encoder) {
                fprintf(stderr, "[DXT GLSL] Failed to create encoder.\n");
                exit(128);
        }
        
        s->decoded = malloc(4 * s->out->tiles[0].width * s->out->tiles[0].height);
        
        s->configured = TRUE;
}

struct video_compress * dxt_glsl_init(char * opts)
{
        struct video_compress *s;
        
        s = (struct video_compress *) malloc(sizeof(struct video_compress));
        s->out = NULL;
        s->decoded = NULL;
        
        x11_enter_thread();
        
        if(opts && strcmp(opts, "help") == 0) {
                printf("DXT GLSL comperssion usage:\n");
                printf("\t-cg:DXT1\n");
                printf("\t\tcompress with DXT1\n");
                printf("\t-cg:DXT5\n");
                printf("\t\tcompress with DXT5 YCoCg\n");
                return NULL;
        }
        
        if(opts) {
                if(strcasecmp(opts, "DXT5") == 0) {
                        s->color_spec = DXT5;
                } else if(strcasecmp(opts, "DXT1") == 0) {
                        s->color_spec = DXT1;
                } else {
                        fprintf(stderr, "Unknown compression : %s\n", opts);
                        return NULL;
                }
        } else {
                s->color_spec = DXT1;
        }
                
        s->configured = FALSE;

        return s;
}

struct video_frame * dxt_glsl_compress(void *arg, struct video_frame * tx)
{
        struct video_compress *s = (struct video_compress *) arg;
        int i;
        unsigned char *line1, *line2;
        
        int x, y;
        
        if(!s->configured)
                configure_with(s, tx);

        for (x = 0; x < tx->grid_width;  ++x) {
                for (y = 0; y < tx->grid_height;  ++y) {
                        struct tile *in_tile = tile_get(tx, x, y);
                        struct tile *out_tile = tile_get(s->out, x, y);
                        
                        line1 = (unsigned char *) in_tile->data;
                        line2 = (unsigned char *) s->decoded;
                        
                        for (i = 0; i < (int) in_tile->height; ++i) {
                                s->decoder(line2, line1, out_tile->linesize,
                                                0, 8, 16);
                                line1 += vc_get_linesize(in_tile->width, tx->color_spec);
                                line2 += out_tile->linesize;
                        }
                        
                        if(s->interlaced_input)
                                vc_deinterlace((unsigned char *) s->decoded, out_tile->linesize,
                                                out_tile->height);
                        
                        dxt_encoder_compress(s->encoder,
                                        (unsigned char *) s->decoded,
                                        (unsigned char *) out_tile->data);
                }
        }
        
        return s->out;
}

void dxt_glsl_exit(void *arg)
{
        struct video_compress *s = (struct video_compress *) arg;
        
        free(s->out->tiles[0].data);
        vf_free(s->out);
        free(s);
}
