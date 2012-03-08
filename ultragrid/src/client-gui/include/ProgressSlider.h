#ifndef PROGRESSSLIDER_H
#define PROGRESSSLIDER_H

#include <wx/slider.h>


BEGIN_DECLARE_EVENT_TYPES()
DECLARE_EVENT_TYPE(wxEVT_SCROLLED, -1)
END_DECLARE_EVENT_TYPES()


class ProgressSlider : public wxSlider
{
    public:
        ProgressSlider(wxWindow *parent, wxWindowID id, int value, int minValue, int maxValue, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize,
                                    long style=wxSL_HORIZONTAL, const wxValidator &validator=wxDefaultValidator, const wxString &name=wxSliderNameStr);
        virtual ~ProgressSlider();


    protected:
    private:
        wxWindow *parent;
        void SliderMoved(wxScrollEvent&);
        DECLARE_EVENT_TABLE()
};

#endif // PROGRESSSLIDER_H
