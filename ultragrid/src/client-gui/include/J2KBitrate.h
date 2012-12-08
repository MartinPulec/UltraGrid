#ifndef J2KBITRATE_H
#define J2KBITRATE_H

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <wx/stattext.h>

class client_guiFrame;

class J2KBitrate : public wxStaticText
{
    public:
        J2KBitrate(wxWindow* parent, wxWindowID id, const wxString& label, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = 0, const wxString& name = wxT("staticText"));
        virtual ~J2KBitrate();
    protected:
    private:
        void Update();
        void QualityChanged(wxScrollEvent& evt);
        void FPSChanged(wxCommandEvent& evt);

        client_guiFrame *m_parent;

        friend class client_guiFrame;
};

#endif // J2KBITRATE_H
