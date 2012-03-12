#ifndef CUSTOMGRIDBAGSIZER_H
#define CUSTOMGRIDBAGSIZER_H

#include <wx/gbsizer.h>


class CustomGridBagSizer : public wxGridBagSizer
{
    public:
        CustomGridBagSizer(int rows, int cols, int vgap, int hgap);
        wxSizerItem* Add(wxWindow* window, int proportion = 0,int flag = 0, int border = 0, wxObject* userData = NULL);
        wxSizerItem* Add(wxSizer* sizer, int proportion = 0,int flag = 0, int border = 0, wxObject* userData = NULL);
        virtual ~CustomGridBagSizer();

    protected:

    private:
        int rows, cols;

        int current;
};

#endif // CUSTOMGRIDBAGSIZER_H
