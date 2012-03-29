#include "../include/Utils.h"

Utils::Utils()
{
    //ctor
}

Utils::~Utils()
{
    //dtor
}


wxString Utils::FromCDouble(double value, int precision)
{
    wxString ret;
    ret << value;
    ret.Replace(wxT(","), wxT("."));

    return ret;
}
