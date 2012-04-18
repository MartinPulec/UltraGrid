#ifndef GLVIEW_H
#define GLVIEW_H
#include <GL/glew.h>
#include <wx/glcanvas.h>
#include <wx/frame.h>

#include <wx/timer.h>

#include <pthread.h>

#include <tr1/memory>

#include "../include/ClientDataIntPair.h"

#define MOUSE_CLICKED_MAGIC 0x1b3ff678

BEGIN_DECLARE_EVENT_TYPES()
DECLARE_EVENT_TYPE(wxEVT_RECONF, -1)
DECLARE_EVENT_TYPE(wxEVT_PUTF, -1)
DECLARE_EVENT_TYPE(wxEVT_UPDATE_TIMER, -1)
DECLARE_EVENT_TYPE(wxEVT_TOGGLE_FULLSCREEN, -1)
DECLARE_EVENT_TYPE(wxEVT_REPAINT, -1)
DECLARE_EVENT_TYPE(wxEVT_TOGGLE_PAUSE, -1)
END_DECLARE_EVENT_TYPES()

class wxFlexGridSizer;

extern const char *filter_mono, *filter_luma;
extern const char *filter_red, *filter_green, *filter_blue;
extern const char *filter_hide_red, *filter_hide_green, *filter_hide_blue;
static const char *filters[] = {0, filter_mono, filter_luma,
                                filter_red, filter_green, filter_blue,
                                filter_hide_red, filter_hide_green, filter_hide_blue};

class GLView : public wxGLCanvas
{
    public:
        GLView(wxFrame *parent, wxWindowID id=wxID_ANY, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=0, const wxString &name=wxT("GLCanvas"), int *attribList=NULL);
        virtual ~GLView();
        /*
         * @returns buffer size
         */
        void reconfigure(int width, int height, int codec);
        void putframe(std::tr1::shared_ptr<char> data);
        void PostInit(wxWindowCreateEvent&);

        void OnPaint( wxPaintEvent& WXUNUSED(event) );
        void Render();
        void LoadSplashScreen();
        void KeyDown(wxKeyEvent& evt);

        void ToggleLightness();
        void DefaultLightness();

        void ShowOnlyChannel(int val);
        void HideChannel(int val);

        void Zoom(double ratio);

        void Go(double x, double y);
        void GoPixels(int x, int y);

        void ResetDefaults();

    protected:
        DECLARE_EVENT_TABLE()

    private:
        void Reconf(wxCommandEvent&);
        void Putf(wxCommandEvent&);
        void Resized(wxSizeEvent& evt);
        void DClick(wxMouseEvent& evt);
        void Click(wxMouseEvent& evt);
        void Wheel(wxMouseEvent& evt);
        void Mouse(wxMouseEvent&);

        void Recompute();

        void resize();

        ClientDataIntPair sizeIncrement;
        wxGLContext *context;
        wxFrame* parent;
        int width, height, codec;
        int data_width, data_height, data_codec;
        int dxt_height; /* ceiled to multiples of 4 */
        double aspect;
        char *data;

        double scaleX, scaleY;

        GLuint     VSHandle,FSHandle,PHandle;
        /* TODO: make same shaders process YUVs for DXT as for
        * uncompressed data */
        GLuint VSHandle_dxt, FSHandle_dxt, PHandle_dxt;
        GLuint FSHandle_dxt5, PHandle_dxt5;

        GLuint Filters[sizeof(filters)/sizeof(const char *)];
        GLuint CurrentFilter;
        GLuint CurrentFilterIdx;

        // Framebuffer
        GLuint fbo_id;
        GLuint fbo_display_id;

        GLuint		texture_display;
        GLuint		texture_uyvy;
        GLuint		texture_final;

        std::tr1::shared_ptr<char> Frame;

        double vpXMultiplier, vpYMultiplier;
        double xoffset, yoffset;

        bool init;

        void prepare_filters();

        void glsl_arb_init();
        void dxt_arb_init();
        void dxt5_arb_init();

        void gl_bind_texture();
        void dxt_bind_texture();
};

#endif // GLVIEW_H
