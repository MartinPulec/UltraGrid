#include "J2KBitrate.h"

#include "client_guiMain.h"
#include "defs.h"

J2KBitrate::J2KBitrate(wxWindow* parent, wxWindowID id, const wxString& label, const wxPoint& pos, const wxSize& size, long style, const wxString& name) :
    wxStaticText(parent, id, label, pos, size, style, name),
    m_parent(dynamic_cast<client_guiFrame *>(parent))
{
}

J2KBitrate::~J2KBitrate()
{
    //dtor
}

void J2KBitrate::Update()
{
    float quality = m_parent->J2KQualitySlider->GetValue() / 1000.0;

    double fps;
    if(m_parent->fps->GetValue().ToDouble(&fps)) {
        SetLabel(wxString::Format(wxT("%.0f Mbps"), fps * quality * 8 * J2K_MAX_FRAME_MB));
    } else {
        SetLabel(wxT("N/A"));
    }
}

void J2KBitrate::QualityChanged(wxScrollEvent& evt)
{
    Update();
}

void J2KBitrate::FPSChanged(wxCommandEvent& evt)
{
    Update();
}
