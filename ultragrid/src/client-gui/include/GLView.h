#ifndef GLVIEW_H
#define GLVIEW_H
#include <GL/glew.h>
#include <wx/glcanvas.h>
#include <wx/frame.h>

#include "../include/ClientDataIntPair.h"

BEGIN_DECLARE_EVENT_TYPES()
DECLARE_EVENT_TYPE(wxEVT_RECONF, -1)
DECLARE_EVENT_TYPE(wxEVT_PUTF, -1)
DECLARE_EVENT_TYPE(wxEVT_UPDATE_TIMER, -1)
END_DECLARE_EVENT_TYPES()

class wxFlexGridSizer;

class GLView : public wxGLCanvas
{
    public:
        GLView(wxFrame *parent, wxWindowID id=wxID_ANY, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=0, const wxString &name=wxT("GLCanvas"), int *attribList=NULL);
        virtual ~GLView();
        void reconfigure(int width, int height, int codec);
        void putframe(char *data, unsigned int frames);
        void PostInit(wxWindowCreateEvent&);
        unsigned int GetFrameSeq();

        DECLARE_EVENT_TABLE()
    protected:
    private:
        void Reconf(wxCommandEvent&);
        void Putf(wxCommandEvent&);
        void Resized(wxSizeEvent& evt);

        void resize();

        ClientDataIntPair sizeIncrement;
        wxGLContext *context;
        wxFrame* parent;
        int width, height, codec;
        int dxt_height; /* ceiled to multiples of 4 */
        double aspect;
        char *data;
        unsigned int frames;

        GLuint     VSHandle,FSHandle,PHandle;
        /* TODO: make same shaders process YUVs for DXT as for
        * uncompressed data */
        GLuint VSHandle_dxt,FSHandle_dxt,PHandle_dxt;
        GLuint FSHandle_dxt5, PHandle_dxt5;

        // Framebuffer
        GLuint fbo_id;
        GLuint		texture_display;
        GLuint		texture_uyvy;

        bool init;

            void glsl_arb_init();
        void dxt_arb_init();
        void dxt5_arb_init();

        void gl_bind_texture();
        void dxt_bind_texture();
};

#endif // GLVIEW_H
