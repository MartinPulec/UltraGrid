#include <wx/sizer.h>
#include <wx/dcclient.h>

#include "../include/GLView.h"


extern "C" {
#include "config.h"
#include "video.h"
};

#define GL_GLEXT_PROTOTYPES 1

#ifdef HAVE_MACOSX
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#else /* HAVE_MACOSX */
#include <GL/gl.h>
#include <GL/glext.h>
#include "x11_common.h"

#endif /* HAVE_MACOSX */

#define STRINGIFY(A) #A

// source code for a shader unit (xsedmik)
static const char * yuv422_to_rgb_fp = STRINGIFY(
uniform sampler2D image;
uniform float imageWidth;
void main()
{
        vec4 yuv;
        yuv.rgba  = texture2D(image, gl_TexCoord[0].xy).grba;
        if(gl_TexCoord[0].x * imageWidth / 2.0 - floor(gl_TexCoord[0].x * imageWidth / 2.0) > 0.5)
                yuv.r = yuv.a;
        yuv.r = 1.1643 * (yuv.r - 0.0625);
        yuv.g = yuv.g - 0.5;
        yuv.b = yuv.b - 0.5;
        gl_FragColor.r = yuv.r + 1.7926 * yuv.b;
        gl_FragColor.g = yuv.r - 0.2132 * yuv.g - 0.5328 * yuv.b;
        gl_FragColor.b = yuv.r + 2.1124 * yuv.g;
});

static const char * yuv422_to_rgb_vp = STRINGIFY(
void main() {
        gl_TexCoord[0] = gl_MultiTexCoord0;
        gl_Position = ftransform();
});

/* DXT YUV (FastDXT) related */
static const char * frag = STRINGIFY(
        uniform sampler2D yuvtex;

        void main(void) {
        vec4 col = texture2D(yuvtex, gl_TexCoord[0].st);

        float Y = 1.1643 * (col[0] - 0.0625);
        float U = col[1] - 0.5;
        float V = col[2] - 0.5;

        float R = Y + 1.7926 * U;
        float G = Y - 0.2132 * U - 0.5328 * V;
        float B = Y + 2.1124 * V;

        gl_FragColor=vec4(R,G,B,1.0);
}
);

static const char * vert = STRINGIFY(
void main() {
        gl_TexCoord[0] = gl_MultiTexCoord0;
        gl_Position = ftransform();}
);

static const char fp_display_dxt5ycocg[] = STRINGIFY(
uniform sampler2D _image;
void main()
{
        vec4 _rgba;
        float _scale;
        float _Co;
        float _Cg;
        float _R;
        float _G;
        float _B;
        _rgba = texture2D(_image, gl_TexCoord[0].xy);
        _scale = 1.00000000E+00/(3.18750000E+01*_rgba.z + 1.00000000E+00);
        _Co = (_rgba.x - 5.01960814E-01)*_scale;
        _Cg = (_rgba.y - 5.01960814E-01)*_scale;
        _R = (_rgba.w + _Co) - _Cg;
        _G = _rgba.w + _Cg;
        _B = (_rgba.w - _Co) - _Cg;
        _rgba = vec4(_R, _G, _B, 1.00000000E+00);
        gl_FragColor = _rgba;
        return;
} // main end
);



DEFINE_EVENT_TYPE(wxEVT_RECONF)
DEFINE_EVENT_TYPE(wxEVT_PUTF)
DEFINE_EVENT_TYPE(wxEVT_UPDATE_TIMER)

BEGIN_EVENT_TABLE(GLView, wxGLCanvas)
  EVT_COMMAND  (wxID_ANY, wxEVT_RECONF, GLView::Reconf)
  EVT_COMMAND  (wxID_ANY, wxEVT_PUTF, GLView::Putf)
  EVT_WINDOW_CREATE(GLView::PostInit)
  EVT_SIZE(GLView::Resized)
END_EVENT_TABLE()

GLView::GLView(wxFrame *p, wxWindowID id, const wxPoint &pos, const wxSize &size, long style, const wxString &name, int *attribList) :
    wxGLCanvas(p, id, pos, size, style, name, attribList),
    parent(p),
    init(false)
{
}

void GLView::PostInit(wxWindowCreateEvent&)
{
    context = new wxGLContext(this);
    wxGLCanvas::SetCurrent(*context);
    SetCurrent();
    //wxPaintDC(this);

    glewInit();

    glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
    glEnable( GL_TEXTURE_2D );

    glGenTextures(1, &texture_display);
    glBindTexture(GL_TEXTURE_2D, texture_display);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &texture_uyvy);
    glBindTexture(GL_TEXTURE_2D, texture_uyvy);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glsl_arb_init();
    dxt_arb_init();
    dxt5_arb_init();
    init = true;
}

void GLView::glsl_arb_init()
{
        char 		*log;
        const GLchar	*VProgram, *FProgram;

        FProgram = (const GLchar *) yuv422_to_rgb_fp;
	VProgram = (const GLchar *) yuv422_to_rgb_vp;
        /* Set up program objects. */
        PHandle=glCreateProgram();
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
        glAttachShader(PHandle,FSHandle);
        glAttachShader(PHandle,VSHandle);
        glLinkProgram(PHandle);

        /* Print link log. */
        memset(log, 0, 32768);
        glGetProgramInfoLog(PHandle,32768,NULL,log);
        printf("Link Log: %s\n", log);
        free(log);

        // Create fbo
        glGenFramebuffersEXT(1, &fbo_id);
}

void GLView::dxt_arb_init()
{
    char *log;
    const GLchar *FProgram, *VProgram;

    FProgram = (const GLchar *) frag;
    VProgram = (const GLchar  *) vert;
    /* Set up program objects. */
    PHandle_dxt=glCreateProgram();
    FSHandle_dxt=glCreateShader(GL_FRAGMENT_SHADER);
    VSHandle_dxt=glCreateShader(GL_VERTEX_SHADER);

    /* Compile Shader */
    glShaderSource(FSHandle_dxt,1,&FProgram,NULL);
    glCompileShader(FSHandle_dxt);
    glShaderSource(VSHandle_dxt,1,&VProgram,NULL);
    glCompileShader(VSHandle_dxt);

    /* Print compile log */
    log=(char*)calloc(32768,sizeof(char));
    glGetShaderInfoLog(FSHandle_dxt,32768,NULL,log);
    printf("Compile Log: %s\n", log);
    free(log);
    log=(char*)calloc(32768,sizeof(char));
    glGetShaderInfoLog(VSHandle_dxt,32768,NULL,log);
    printf("Compile Log: %s\n", log);
    free(log);

    /* Attach and link our program */
    glAttachShader(PHandle_dxt,FSHandle_dxt);
    glAttachShader(PHandle_dxt,VSHandle_dxt);
    glLinkProgram(PHandle_dxt);

    /* Print link log. */
    log=(char*)calloc(32768,sizeof(char));
    glGetProgramInfoLog(PHandle_dxt,32768,NULL,log);
    printf("Link Log: %s\n", log);
    free(log);
}

void GLView::dxt5_arb_init()
{
        char *log;
        const GLchar *FProgram;

        FProgram = (const GLchar *) fp_display_dxt5ycocg;

        /* Set up program objects. */
        PHandle_dxt5=glCreateProgram();
        FSHandle_dxt5=glCreateShader(GL_FRAGMENT_SHADER);

        /* Compile Shader */
        glShaderSource(FSHandle_dxt5, 1, &FProgram, NULL);
        glCompileShader(FSHandle_dxt5);

        /* Print compile log */
        log=(char*)calloc(32768, sizeof(char));
        glGetShaderInfoLog(FSHandle_dxt5, 32768, NULL, log);
        printf("Compile Log: %s\n", log);
        free(log);

        /* Attach and link our program */
        glAttachShader(PHandle_dxt5, FSHandle_dxt5);
        glLinkProgram(PHandle_dxt5);

        /* Print link log. */
        log=(char*)calloc(32768, sizeof(char));
        glGetShaderInfoLog(PHandle_dxt5, 32768, NULL, log);
        printf("Link Log: %s\n", log);
        free(log);
}



GLView::~GLView()
{
    //dtor
}

void GLView::reconfigure(int width, int height, int codec)
{
    wxCommandEvent event(wxEVT_RECONF, GetId());

    this->width = width;
    this->height = height;
    this->codec = codec;
    this->dxt_height = (height + 3) / 4 * 4;
    this->aspect = (double) width / height;
    // we deffer the event (this method is called from "wrong" thread
    wxPostEvent(this, event);
    //wxEvent wxEVT_RECONFIGURE = wxNewEventType();
}

void GLView::Reconf(wxCommandEvent& event)
{
    //PostInit();
    wxGLCanvas::SetCurrent(*context);
    //this->SetSize(wxSize(width, height));

//this->SetSizeHints(wxSize(width, height), wxSize(width, height));

    // compute increment
    sizeIncrement.setFirst(width - GetSize().x);
    sizeIncrement.setSecond(height - GetSize().y);
    event.SetClientObject(&sizeIncrement);

    //this->Show(false);
    //sizer->Fit(this);
    //sizer->SetSizeHints(this);

    glUseProgram(0);

    if(codec == DXT1 || codec == DXT1_YUV) {
        glBindTexture(GL_TEXTURE_2D,texture_display);
        glCompressedTexImage2D(GL_TEXTURE_2D, 0,
                GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                (width + 3) / 4 * 4, dxt_height, 0,
                ((width + 3) / 4 * 4* dxt_height)/2,
                NULL);
        if(codec == DXT1_YUV) {
            glBindTexture(GL_TEXTURE_2D,texture_display);
            glUseProgramObjectARB(PHandle_dxt);
        }
    } else if (codec == UYVY) {
        glActiveTexture(GL_TEXTURE0 + 2);
        glBindTexture(GL_TEXTURE_2D,texture_uyvy);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                        width / 2, height, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE,
                        NULL);
        glUseProgram(PHandle);
        glUniform1i(glGetUniformLocation(PHandle, "image"), 2);
        glUniform1f(glGetUniformLocation(PHandle, "imageWidth"),
                (GLfloat) width);
        glUseProgram(0);
        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D,texture_display);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                        width, height, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE,
                        NULL);
    } else if (codec == RGBA) {
        glBindTexture(GL_TEXTURE_2D,texture_display);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                        width, height, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE,
                        NULL);
    } else if (codec == RGB) {
        glBindTexture(GL_TEXTURE_2D,texture_display);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                        width, height, 0,
                        GL_RGB, GL_UNSIGNED_BYTE,
                            NULL);
    } else if (codec == DXT5) {
        glUseProgram(PHandle_dxt5);

        glBindTexture(GL_TEXTURE_2D,texture_display);
        glCompressedTexImage2D(GL_TEXTURE_2D, 0,
        GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
        (width + 3) / 4 * 4, dxt_height, 0,
        (width + 3) / 4 * 4 * dxt_height,
        NULL);
    }

    resize();

    //wxPostEvent(parent, event);
}

void GLView::Resized(wxSizeEvent& evt)
{
    resize();
}

void GLView::resize()
{
    if(!init) return;
    glViewport( 0, 0, ( GLint )GetSize().x, ( GLint )GetSize().y);
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity( );

	double screen_ratio;
	double x = 1.0,
	   y = 1.0;

	// debug_msg("Resized to: %dx%d\n", width, height);

	screen_ratio = (double) GetSize().x / GetSize().y;
	if(screen_ratio > aspect) {
	    x = (double) GetSize().y * aspect / GetSize().x;
	} else {
	    y = (double) GetSize().x/ (GetSize().y * aspect);
	}
	glScalef(x, y, 1);

	glOrtho(-1,1,-1/aspect,1/aspect,10,-10);

	glMatrixMode( GL_MODELVIEW );

	glLoadIdentity( );

    glClear(GL_COLOR_BUFFER_BIT);
    float bottom = 1.0f - (dxt_height - height) / (float) dxt_height * 2;

        //gl_check_error();
        glBegin(GL_QUADS);
          /* Front Face */
          /* Bottom Left Of The Texture and Quad */
          glTexCoord2f( 0.0f, bottom ); glVertex2f( -1.0f, -1/aspect);
          /* Bottom Right Of The Texture and Quad */
          glTexCoord2f( 1.0f, bottom ); glVertex2f(  1.0f, -1/aspect);
          /* Top Right Of The Texture and Quad */
          glTexCoord2f( 1.0f, 0.0f ); glVertex2f(  1.0f,  1/aspect);
          /* Top Left Of The Texture and Quad */
          glTexCoord2f( 0.0f, 0.0f ); glVertex2f( -1.0f,  1/aspect);
        glEnd( );

        SwapBuffers();
}


void GLView::putframe(char *data, unsigned int frames)
{
    this->data = data;
    this->frames = frames;
    wxCommandEvent event(wxEVT_PUTF, GetId());
    wxPostEvent(this, event);

    wxCommandEvent event_timer(wxEVT_UPDATE_TIMER, GetId());
    wxPostEvent(parent, event_timer);
}

unsigned int GLView::GetFrameSeq()
{
    return this->frames;
}

void GLView::Putf(wxCommandEvent&)
{
        wxGLCanvas::SetCurrent(*context);
        switch(codec) {
        case DXT1:
            glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            (width + 3) / 4 * 4, dxt_height,
                            GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                            ((width + 3) / 4 * 4 * dxt_height)/2,
                            data);
            break;
        case DXT1_YUV:
            dxt_bind_texture();
            break;
        case UYVY:
            gl_bind_texture();
            break;
        case RGBA:
            glBindTexture(GL_TEXTURE_2D, texture_display);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            width, height,
                            GL_RGBA, GL_UNSIGNED_BYTE,
                            data);
            break;
        case RGB:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            width, height,
                            GL_RGB, GL_UNSIGNED_BYTE,
                            data);
            break;
        case DXT5:
            glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            (width + 3) / 4 * 4, dxt_height,
                            GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                            (width + 3) / 4 * 4 * dxt_height,
                            data);
            break;
        default:
            //fprintf(stderr, "[GL] Fatal error - received unsupported codec.\n");
            //exit_uv(128);
            return;
    }

    {
        float bottom;

        /* Clear the screen */
        glClear(GL_COLOR_BUFFER_BIT);

        glLoadIdentity( );
        glTranslatef( 0.0f, 0.0f, -1.35f );

        /* Reflect that we may have taller texture than reasonable data
         * if we use DXT and source height was not divisible by 4
         * In normal case, there would be 1.0 */
        bottom = 1.0f - (dxt_height - height) / (float) dxt_height * 2;

        //gl_check_error();
        glBegin(GL_QUADS);
          /* Front Face */
          /* Bottom Left Of The Texture and Quad */
          glTexCoord2f( 0.0f, bottom ); glVertex2f( -1.0f, -1/aspect);
          /* Bottom Right Of The Texture and Quad */
          glTexCoord2f( 1.0f, bottom ); glVertex2f(  1.0f, -1/aspect);
          /* Top Right Of The Texture and Quad */
          glTexCoord2f( 1.0f, 0.0f ); glVertex2f(  1.0f,  1/aspect);
          /* Top Left Of The Texture and Quad */
          glTexCoord2f( 0.0f, 0.0f ); glVertex2f( -1.0f,  1/aspect);
        glEnd( );

        SwapBuffers();

        //gl_check_error();
    }
}

void GLView::gl_bind_texture()
{
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_id);
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture_display, 0);
        //assert(GL_FRAMEBUFFER_COMPLETE_EXT == glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT));
        glActiveTexture(GL_TEXTURE0 + 2);
        glBindTexture(GL_TEXTURE_2D, texture_uyvy);

        glMatrixMode( GL_PROJECTION );
        glPushMatrix();
	glLoadIdentity( );
	glOrtho(-1,1,-1/aspect,1/aspect,10,-10);

	glMatrixMode( GL_MODELVIEW );
        glPushMatrix();
	glLoadIdentity( );

        glPushAttrib(GL_VIEWPORT_BIT);

        glViewport( 0, 0, width, height);

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width / 2, height,  GL_RGBA, GL_UNSIGNED_BYTE, data);
        glUseProgramObjectARB(PHandle);

        glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

        float aspect = (double) width / height;
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0/aspect);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0/aspect);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0/aspect);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0/aspect);
        glEnd();

        glPopAttrib();


        glMatrixMode( GL_PROJECTION );
        glPopMatrix();
        glMatrixMode( GL_MODELVIEW );
        glPopMatrix();

        glUseProgramObjectARB(0);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D, texture_display);
}

void GLView::dxt_bind_texture()
{
        static int i=0;

        //TODO: does OpenGL use different stuff here?
        glActiveTexture(GL_TEXTURE0);
        i=glGetUniformLocationARB(PHandle,"yuvtex");
        glUniform1iARB(i,0);
        glBindTexture(GL_TEXTURE_2D,texture_display);
	glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
			width, height,
			GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
			(width * height/16)*8,
			data);
}
