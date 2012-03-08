#ifndef CLIENTDATAINTPAIR_H
#define CLIENTDATAINTPAIR_H

#include <wx/clntdata.h>


class ClientDataIntPair : public wxClientData
{
    public:
        ClientDataIntPair(int a, int b);
        ClientDataIntPair();
        virtual ~ClientDataIntPair();

        int first();
        int second();
        void setFirst(int val);
        void setSecond(int val);
    protected:
    private:
        int first_int;
        int second_int;
};

#endif // CLIENTDATAINTPAIR_H
