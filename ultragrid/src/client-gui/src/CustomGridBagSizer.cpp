#include "../include/CustomGridBagSizer.h"

CustomGridBagSizer::CustomGridBagSizer(int rows, int cols, int vgap, int hgap)
    : wxGridBagSizer(vgap, hgap)
{
    this->rows = rows;
    this->cols = cols;
    current = 0;
}

CustomGridBagSizer::~CustomGridBagSizer()
{
    //dtor
}

wxSizerItem* CustomGridBagSizer::Add(wxWindow* window, int proportion,int flag, int border, wxObject* userData)
{
    wxGBPosition pos(current % rows, current / rows);
    wxGridBagSizer::Add(window, pos, wxDefaultSpan, flag, border, userData);
    current++;
}

wxSizerItem* CustomGridBagSizer::Add(wxSizer* sizer, int proportion,int flag, int border, wxObject* userData)
{
    wxGBPosition pos(current % rows, current / rows);
    wxGridBagSizer::Add(sizer, pos, wxDefaultSpan, flag, border, userData);
    current++;
}
