#include <wx/sizer.h>
#include <wx/dcclient.h>

#include <iostream>

#include "../include/GLView.h"


extern "C" {
#include "config.h"
#include "config_unix.h"
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

#include "cesnet-logo-2.c"

static char fp_display_rgba_to_yuv422_legacy[] =
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



DEFINE_EVENT_TYPE(wxEVT_RECONF)
DEFINE_EVENT_TYPE(wxEVT_PUTF)
DEFINE_EVENT_TYPE(wxEVT_UPDATE_TIMER)
DEFINE_EVENT_TYPE(wxEVT_TOGGLE_FULLSCREEN)
DEFINE_EVENT_TYPE(wxEVT_TOGGLE_PAUSE)

BEGIN_EVENT_TABLE(GLView, wxGLCanvas)
  EVT_PAINT(GLView::OnPaint)
  EVT_WINDOW_CREATE(GLView::PostInit)

  EVT_COMMAND  (wxID_ANY, wxEVT_RECONF, GLView::Reconf)
  EVT_COMMAND  (wxID_ANY, wxEVT_PUTF, GLView::Putf)

  EVT_SIZE(GLView::Resized)

  EVT_LEFT_DCLICK(GLView::DClick)
  EVT_LEFT_DOWN(GLView::Click)
  EVT_MOTION(GLView::Mouse)
  EVT_LEFT_UP(GLView::Mouse)
  EVT_KEY_DOWN(GLView::KeyDown)

  EVT_MOUSEWHEEL(GLView::Wheel)
END_EVENT_TABLE()

const int lightness_count = 3;
const int channel_count = 3;

extern "C" void gl_check_error(void);

GLView::GLView(wxFrame *p, wxWindowID id, const wxPoint &pos, const wxSize &size, long style, const wxString &name, int *attribList) :
#if 0 //def HAVE_MACOSX
    wxGLCanvas(p, id, pos, size, style, name, attribList),
#else
    wxGLCanvas(p, id, attribList, pos, size, style, name),
#endif
    parent(p),
    init(false),
    Frame(std::tr1::shared_ptr<char>()),
    useHWDisplay(false)
{
    vpXMultiplier = vpYMultiplier = 1.0;
    xoffset = yoffset = 0.0;
}

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

const char * filter_luma = STRINGIFY(
        uniform sampler2D image;

        void main(void) {
        vec4 col = texture2D(image, gl_TexCoord[0].st);

        // ITU-T BT.709
        float luma = col.r * 0.2126 + col.g * 0.7152 + col.b * 0.0722;

        gl_FragColor=vec4(luma, luma, luma, 1.0);
}
);

const char * filter_mono = STRINGIFY(
        uniform sampler2D image;

        void main(void) {
        vec4 col = texture2D(image, gl_TexCoord[0].st);

        // ITU-T BT.709
        float luma = col.r * 0.2126 + col.g * 0.7152 + col.b * 0.0722;
        float mono = step(0.5, luma);

        gl_FragColor=vec4(mono, mono, mono, 1.0);
}
);

const char * filter_red = STRINGIFY(
        uniform sampler2D image;

        void main(void) {
        vec4 col = texture2D(image, gl_TexCoord[0].st);
        gl_FragColor=vec4(col.r, 0.0, 0.0, 1.0);
}
);

const char * filter_green = STRINGIFY(
        uniform sampler2D image;

        void main(void) {
        vec4 col = texture2D(image, gl_TexCoord[0].st);
        gl_FragColor=vec4(0.0, col.g, 0.0, 1.0);
}
);

const char * filter_blue = STRINGIFY(
        uniform sampler2D image;

        void main(void) {
        vec4 col = texture2D(image, gl_TexCoord[0].st);
        gl_FragColor=vec4(0.0, 0.0, col.b, 1.0);
}
);

const char * filter_hide_red = STRINGIFY(
        uniform sampler2D image;

        void main(void) {
        vec4 col = texture2D(image, gl_TexCoord[0].st);
        gl_FragColor=vec4(0.0, col.g, col.b, 1.0);
}
);

const char * filter_hide_green = STRINGIFY(
        uniform sampler2D image;

        void main(void) {
        vec4 col = texture2D(image, gl_TexCoord[0].st);
        gl_FragColor=vec4(col.r, 0.0, col.b, 1.0);
}
);

const char * filter_hide_blue = STRINGIFY(
        uniform sampler2D image;

        void main(void) {
        vec4 col = texture2D(image, gl_TexCoord[0].st);
        gl_FragColor=vec4(col.r, col.g, 0.0, 1.0);
}
);


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


void GLView::PostInit(wxWindowCreateEvent&)
{
    context = new wxGLContext(this);
    wxGLCanvas::SetCurrent(*context);
    //SetCurrent();

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

    glGenTextures(1, &texture_final);
    glBindTexture(GL_TEXTURE_2D, texture_final);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glsl_arb_init();
    dxt_arb_init();
    dxt5_arb_init();
    prepare_filters();
    init_device_shaders();

    glGenFramebuffersEXT(1, &fbo_uncompressed);
    glGenTextures(1, &texture_uncompressed);
    glBindTexture(GL_TEXTURE_2D, texture_uncompressed);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

#ifdef USE_PBO
    glGenBuffersARB(1, &pbo);
#endif

    CurrentFilterIdx = 0;
    CurrentFilter = Filters[CurrentFilterIdx];

    init = true;

    LoadSplashScreen();
}

void GLView::prepare_filters()
{
    for(int i = 0; i < sizeof(filters) / sizeof(const char *); ++i) {
        if(filters[i] == 0) {
            Filters[i] = 0;
            continue;
        }

        char 		*log;
        const GLchar	*VProgram, *FProgram;
        GLuint     VSHandle, FSHandle;

        FProgram = (const GLchar *) filters[i];
        VProgram = (const GLchar *) vert;
        /* Set up program objects. */
        this->Filters[i] =glCreateProgram();
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
        glAttachShader(Filters[i], FSHandle);
        glAttachShader(Filters[i], VSHandle);
        glLinkProgram(Filters[i]);

        /* Print link log. */
        memset(log, 0, 32768);
        glGetProgramInfoLog(Filters[i], 32768, NULL, log);
        printf("Link Log: %s\n", log);
        free(log);

        glUseProgram(Filters[i]);
        glUniform1iARB(glGetUniformLocationARB(PHandle,"image"),0);
        glUseProgram(Filters[i]);
    }
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
        glGetProgramInfoLog(PHandle_dxt5, 32768, NULL, log);
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

    ResetDefaults();

    this->width = data_width = width;
    this->height = data_height = height;
    this->codec = data_codec = codec;
    this->dxt_height = (height + 3) / 4 * 4;
    this->aspect = (double) width / height;
    // we deffer the event (this method is called from "wrong" thread
    wxPostEvent(this, event);
    //wxEvent wxEVT_RECONFIGURE = wxNewEventType();
}

void GLView::Reconf(wxCommandEvent& event)
{
    this->data = NULL;
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

    glBindTexture(GL_TEXTURE_2D,texture_final);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                        width, height, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE,
                        NULL);

    glBindTexture(GL_TEXTURE_2D,texture_uncompressed);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                        width, height, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE,
                        NULL);


    glGenFramebuffersEXT(1, &fbo_display_id);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, texture_device);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGBA, width / 2, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glBindTexture(GL_TEXTURE_2D, 0);

#ifdef USE_PBO
     glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pbo);
     glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, width * 2 /* UYVY */, 0, GL_STREAM_READ_ARB);
     glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
#endif

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
	scaleX = scaleY = 1.0;

	// debug_msg("Resized to: %dx%d\n", width, height);

	screen_ratio = (double) GetSize().x / GetSize().y;
	if(screen_ratio > aspect) {
	    scaleX = (double) GetSize().y * aspect / GetSize().x;
	} else {
	    scaleY = (double) GetSize().x/ (GetSize().y * aspect);
	}
	glScalef(scaleX  * vpXMultiplier, scaleY  * vpYMultiplier, 1);

	glOrtho(-1 + xoffset, 1 + xoffset,-1/aspect - yoffset, 1/aspect - yoffset,10,-10);

	glMatrixMode( GL_MODELVIEW );

	glLoadIdentity( );

    glClear(GL_COLOR_BUFFER_BIT);

#if 0

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
#endif

    Refresh();
    //SwapBuffers();
}

void GLView::LoadSplashScreen()
{
    wxCommandEvent unused;
    this->width = cesnet_logo.width;
    this->height = cesnet_logo.height;
    this->codec = RGBA;
    this->dxt_height = (height + 3) / 4 * 4;
    this->aspect = (double) cesnet_logo.width / cesnet_logo.height;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );

    ResetDefaults();

    Reconf(unused);
    this->data = (char *) cesnet_logo.pixel_data;
    Render();
}


void GLView::putframe(std::tr1::shared_ptr<char> data, bool outputToHWDisplay)
{
    if(data_width != width || data_height != height || data_codec != codec) {
        width = data_width;
        height = data_height;
        codec = data_codec;
        this->dxt_height = (height + 3) / 4 * 4;
        this->aspect = (double) width / height;

        wxCommandEvent unused;
        Reconf(unused);
    }

    this->data = data.get();
    this->Frame = data;
    this->useHWDisplay = outputToHWDisplay;

    Render();
}

void GLView::Putf(wxCommandEvent&)
{
}

void GLView::RenderScrollbars()
{
    glMatrixMode( GL_PROJECTION );
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadIdentity();

    glUseProgram(0);

    glDisable(GL_TEXTURE_2D);

    if(vpXMultiplier > 1.0) {
        float linewidth = 1.92 / vpXMultiplier;
        float line_x_left = (-1 + (xoffset - -(vpXMultiplier - 1) /vpXMultiplier)) * 0.96;
        float line_x_right = line_x_left + linewidth;

        glBegin(GL_QUADS);
       glColor4f(0.3,0.3,0.3,1.0);
            glVertex3f(line_x_left, -0.98, 1);
            glVertex3f(line_x_right, -0.98, 1);
            glVertex3f(line_x_right, -0.96, 1);
            glVertex3f(line_x_left, -0.96, 1);
        glEnd();
        glColor4f(1.0, 1.0, 1.0, 1.0);
    }
    if(vpYMultiplier > 1.0) {
        float linewidth = 1.92 / vpYMultiplier;
        float line_y_top = 0.96 - ( (yoffset - -(vpYMultiplier - 1) /vpYMultiplier / aspect) ) * aspect * 0.96;
        float line_y_bottom = line_y_top - linewidth;

        glBegin(GL_QUADS);
       glColor4f(0.3,0.3,0.3,1.0);
            glVertex3f(0.96,line_y_top, 1);
            glVertex3f(0.96 + 0.02 / aspect,line_y_top, 1);
            glVertex3f(0.96 + 0.02 / aspect,line_y_bottom, 1);
            glVertex3f(0.96,line_y_bottom, 1);
        glEnd();
       glColor3f(1,1,1);
    }

    glMatrixMode( GL_PROJECTION );
    glPopMatrix();
    glMatrixMode( GL_MODELVIEW );
    glPopMatrix();

    glEnable(GL_TEXTURE_2D);
}

void GLView::Render()
{
    if (!data) {
        return;
    }

    struct video_frame *frame = NULL;
    if(useHWDisplay) {
        frame = display_get_frame(this->hw_display);
    }

    wxGLCanvas::SetCurrent(*context);

    glBindTexture(GL_TEXTURE_2D, texture_display);

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
            glUseProgram(PHandle_dxt5);
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
        {
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_uncompressed);
            glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture_uncompressed, 0);

            /* Clear the screen */
            glClear(GL_COLOR_BUFFER_BIT);

            glMatrixMode( GL_PROJECTION );
            glPushMatrix();
            glLoadIdentity( );
            glOrtho(-1,1,-1,1,10,-10);
            glMatrixMode( GL_MODELVIEW );
            glPushMatrix();
            glLoadIdentity( );
            glPushAttrib(GL_VIEWPORT_BIT);
            glViewport( 0, 0, width, height);

            glBindTexture(GL_TEXTURE_2D, texture_display);

            glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
            glBegin(GL_QUADS);
            glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
            glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
            glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
            glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
            glEnd( );

            glPopAttrib();
            glMatrixMode( GL_PROJECTION );
            glPopMatrix();
            glMatrixMode( GL_MODELVIEW );
            glPopMatrix();


            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
            glBindTexture(GL_TEXTURE_2D, texture_uncompressed);

            glUseProgram(0);
        }

        /* Clear the screen */
        glClear(GL_COLOR_BUFFER_BIT);

        float bottom;

        glLoadIdentity( );
        glTranslatef( 0.0f, 0.0f, -1.35f );

        /* Reflect that we may have taller texture than reasonable data
         * if we use DXT and source height was not divisible by 4
         * In normal case, there would be 1.0 */
        if(codec == DXT1 || codec == DXT5 || codec == DXT1_YUV) {
            bottom = 1.0f - (dxt_height - height) / (float) dxt_height * 2;
        } else {
            bottom = 1.0f;
        }

        if(CurrentFilter) {

            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_display_id);
            glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture_final, 0);


            glMatrixMode( GL_PROJECTION );
            glPushMatrix();
            glLoadIdentity( );
            glOrtho(-1,1,-1/aspect,1/aspect,10,-10);
            glMatrixMode( GL_MODELVIEW );
            glPushMatrix();
            glLoadIdentity( );
            glPushAttrib(GL_VIEWPORT_BIT);
            glViewport( 0, 0, width, height);

            glUseProgram(CurrentFilter);

            glBegin(GL_QUADS);
            glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0/aspect);
            glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0/aspect);
            glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0/aspect);
            glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0/aspect);
            glEnd( );

            glPopAttrib();
            glMatrixMode( GL_PROJECTION );
            glPopMatrix();
            glMatrixMode( GL_MODELVIEW );
            glPopMatrix();


            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
            glBindTexture(GL_TEXTURE_2D, texture_final);

            glUseProgram(0);
        } else {
        }

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

        RenderScrollbars();

        if(frame) {
            glDisable(GL_BLEND);
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo_device);
            glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture_device, 0);

            glPushAttrib(GL_VIEWPORT_BIT);
            glViewport( 0, 0, width / 2, height);

            glMatrixMode( GL_PROJECTION );
            glPushMatrix();
            glLoadIdentity( );

            glScalef(vpXMultiplier, vpYMultiplier, 1);

            glOrtho(-1 + xoffset, 1 + xoffset,-1 + yoffset * aspect, 1 +  yoffset * aspect,10,-10);
            //glOrtho(-1,1,-1,1,10,-10);

            glMatrixMode( GL_MODELVIEW );
            glPushMatrix();
            glLoadIdentity( );

            glUseProgram(program_rgba_to_yuv422);
            glUniform1f(glGetUniformLocation(program_rgba_to_yuv422, "imageWidth"),
                        (GLfloat) width * vpXMultiplier);

            glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

            glBegin(GL_QUADS);
                glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
                glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
                glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
                glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
            glEnd();

            glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
#ifdef USE_PBO
            // glReadPixels() should return immediately.
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pbo);
            glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
            glReadPixels(0, 0, width / 2, height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, 0);

            // map the PBO to process its data by CPU
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pbo);
            ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB,
                            GL_READ_ONLY_ARB);
            if(ptr)
            {
                    memcpy(frame->tiles[0].data, ptr, decoder->width * decoder->height * 2);
                    glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);
            }

            // back to conventional pixel operation
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
#else
            glReadPixels(0, 0, width / 2, height, GL_RGBA, GL_UNSIGNED_BYTE, frame->tiles[0].data);
#endif
            glMatrixMode( GL_PROJECTION );
            glPopMatrix();
            glMatrixMode( GL_MODELVIEW );
            glPopMatrix();

            glPopAttrib();
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

            glEnable(GL_BLEND);
            //memcpy(frame->tiles[0].data, data, frame->tiles[0].data_len);
            display_put_frame(this->hw_display, (char *) frame);
        }

        glUseProgram(0);

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

void GLView::DClick(wxMouseEvent& evt)
{
    /*wxCommandEvent evt_pause(wxEVT_TOGGLE_PAUSE, GetId());
    evt_pause.SetExtraLong(MOUSE_CLICKED_MAGIC);
    wxPostEvent(parent, evt_pause);*/

    wxCommandEvent event_fullscreen(wxEVT_TOGGLE_FULLSCREEN, GetId());
    wxPostEvent(parent, event_fullscreen);
}

void GLView::Click(wxMouseEvent& evt)
{
    /*SetFocus();
    wxCommandEvent evt_pause(wxEVT_TOGGLE_PAUSE, GetId());
    evt_pause.SetExtraLong(MOUSE_CLICKED_MAGIC);
    wxPostEvent(parent, evt_pause);
    */
    evt.Skip(); // DOC: The handler of this event should normally call event.Skip() to allow the default processing to take place as otherwise the window under mouse wouldn't get the focus.
}

void GLView::Mouse(wxMouseEvent& evt)
{
    wxPostEvent(parent, evt);
}

void GLView::OnPaint( wxPaintEvent& WXUNUSED(event) )
{
    if(this->init) {
        wxPaintDC(this);
        Render();
    }
}

void GLView::KeyDown(wxKeyEvent& evt)
{
    wxPostEvent(parent, evt);
}

void GLView::ToggleLightness()
{
    if(CurrentFilterIdx >= lightness_count)
        CurrentFilterIdx = 0;
    else
        CurrentFilterIdx = (CurrentFilterIdx + 1) % lightness_count;
    CurrentFilter = Filters[CurrentFilterIdx];
}

void GLView::DefaultLightness()
{
    CurrentFilterIdx = 0;
    CurrentFilter = Filters[CurrentFilterIdx];
}

void GLView::ShowOnlyChannel(int val)
{
    switch(val) {
        case 'R':
            CurrentFilterIdx = lightness_count + 0;
            break;
        case 'G':
            CurrentFilterIdx = lightness_count + 1;
            break;
        case 'B':
            CurrentFilterIdx = lightness_count + 2;
            break;
    }
    CurrentFilter = Filters[CurrentFilterIdx];

    Render();
}

void GLView::HideChannel(int val)
{
    switch(val) {
        case 'R':
            CurrentFilterIdx = lightness_count + channel_count + 0;
            break;
        case 'G':
            CurrentFilterIdx = lightness_count + channel_count + 1;
            break;
        case 'B':
            CurrentFilterIdx = lightness_count + channel_count + 2;
            break;
    }
    CurrentFilter = Filters[CurrentFilterIdx];

    Render();
}

void GLView::Zoom(double ratio)
{
    vpXMultiplier *= (1 + ratio);
    vpYMultiplier *= (1 + ratio);
    Recompute();
    resize();
}

void GLView::Go(double x, double y)
{
    xoffset += x;
    yoffset += y;
    Recompute();
    resize();
}

void GLView::ResetDefaults()
{
    xoffset = yoffset = 0;
    vpXMultiplier = vpYMultiplier = 1.0;

    CurrentFilterIdx = 0;
    CurrentFilter = Filters[CurrentFilterIdx];
}

void GLView::Recompute()
{
    if(vpXMultiplier < 1.0 || vpYMultiplier < 1.0) {
        vpXMultiplier = vpYMultiplier = 1.0;
    }

    if(xoffset < - (vpXMultiplier - 1) / vpXMultiplier)
        xoffset = - (vpXMultiplier - 1) /vpXMultiplier;

    if(xoffset > + (vpXMultiplier - 1) / vpXMultiplier)
        xoffset = + (vpXMultiplier - 1) / vpXMultiplier;

    if(yoffset < - (vpYMultiplier - 1) / vpYMultiplier / aspect)
        yoffset = - (vpYMultiplier - 1) /vpYMultiplier / aspect;

    if(yoffset > (vpYMultiplier - 1) / vpYMultiplier / aspect)
        yoffset = (vpYMultiplier - 1) /vpYMultiplier / aspect;
}

void GLView::Wheel(wxMouseEvent& evt)
{
    wxPostEvent(parent, evt);
}

void GLView::GoPixels(int xdelta, int ydelta)
{
    xoffset += (double) xdelta / GetSize().x / vpXMultiplier;
    yoffset += (double) ydelta / GetSize().y / vpYMultiplier;
    Recompute();
    resize();
}

void GLView::setHWDisplay(struct display* hw_display)
{
    this->hw_display = hw_display;
}

void GLView::init_device_shaders()
{
    glGenFramebuffersEXT(1, &fbo_device);

    glGenTextures(1, &texture_device);

    GLsizei log_len;
    program_rgba_to_yuv422 = glCreateProgram();
    GLuint     VSHandle,FSHandle,PHandle;
    FSHandle = glCreateShader(GL_FRAGMENT_SHADER);

    char *shader_program = strdup(fp_display_rgba_to_yuv422_legacy);
    glShaderSource(FSHandle, 1,  (const GLchar **) &shader_program, NULL);
    glCompileShader(FSHandle);

    char *log = (char *) calloc(32768,sizeof(char));
    glGetShaderInfoLog(FSHandle, 32768, &log_len, log);
    printf("Compile Log: %s\n", log);

    glAttachShader(program_rgba_to_yuv422, FSHandle);
    glLinkProgram(program_rgba_to_yuv422);

    memset(log, 0, 32768);
    glGetProgramInfoLog(program_rgba_to_yuv422, 32768, &log_len, log);
    printf("Link Log: %s\n", log);

    glUseProgram(program_rgba_to_yuv422);
    glUniform1i(glGetUniformLocation(program_rgba_to_yuv422, "image"), 0);
    glUseProgram(0);

    free(log);
    free(shader_program);
}
