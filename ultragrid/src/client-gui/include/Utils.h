#ifndef UTILS_H
#define UTILS_H

#include <wx/string.h>

class Utils
{
    public:
        Utils();
        virtual ~Utils();

        static wxString FromCDouble(double value, int precision);
    protected:
    private:
};

#endif // UTILS_H
