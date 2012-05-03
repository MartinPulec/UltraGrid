#ifndef CLIENTDATACSTR_H
#define CLIENTDATACSTR_H

#include <wx/clntdata.h>

class ClientDataCStr: public wxClientData
{
    public:
        ClientDataCStr(const char *data);
        virtual ~ClientDataCStr();

        char *get();
    protected:
    private:
        char *cstring;
};

#endif // CLIENTDATACSTR_H
