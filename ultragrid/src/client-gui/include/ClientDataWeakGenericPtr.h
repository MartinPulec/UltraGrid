#ifndef CLIENTDATAWEAKGENERICPTR_H
#define CLIENTDATAWEAKGENERICPTR_H

#include <wx/clntdata.h>

class ClientDataWeakGenericPtr : public wxClientData
{
    public:
        ClientDataWeakGenericPtr(void *ptr);
        void *get();
    protected:
    private:
        void *ptr;
};

#endif // CLIENTDATAWEAKGENERICPTR_H
