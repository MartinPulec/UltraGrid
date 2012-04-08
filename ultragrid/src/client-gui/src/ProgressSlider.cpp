#include "../include/ProgressSlider.h"
#include <iostream>

DEFINE_EVENT_TYPE(wxEVT_SCROLLED)

BEGIN_EVENT_TABLE(ProgressSlider,wxSlider)
    //(*EventTable(client_guiFrame)
    //*)
#ifdef __WXMAC__
    EVT_SCROLL_THUMBRELEASE(ProgressSlider::SliderMoved)
#else
    EVT_SCROLL_CHANGED(ProgressSlider::SliderMoved)
#endif

END_EVENT_TABLE()


ProgressSlider::ProgressSlider(wxWindow *p, wxWindowID id, int value, int minValue, int maxValue, const wxPoint &pos, const wxSize &size,
                                    long style, const wxValidator &validator, const wxString &name)
    : wxSlider(p, id, value, minValue, maxValue, pos, size, style, validator, name),
    parent(p)
{
    //ctor
}

ProgressSlider::~ProgressSlider()
{
    //dtor
}


void ProgressSlider::SliderMoved(wxScrollEvent&)
{
    wxCommandEvent event_scrolled(wxEVT_SCROLLED, GetId());
    wxPostEvent(parent, event_scrolled);
}
