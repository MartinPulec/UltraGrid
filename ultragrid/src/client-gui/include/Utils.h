#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <netinet/in.h>
#include <wx/string.h>

class Utils
{
    public:
        Utils();
        virtual ~Utils();

        static wxString FromCDouble(double value, int precision);
        static bool boolFromString(std::string);

        static int conn_nonb(struct sockaddr_in sa, int sock, int timeout);
    protected:
    private:
};

#endif // UTILS_H
