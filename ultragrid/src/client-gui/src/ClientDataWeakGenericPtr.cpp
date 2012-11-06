#include "ClientDataWeakGenericPtr.h"

ClientDataWeakGenericPtr::ClientDataWeakGenericPtr(void *p) :
    ptr(p)
{
    //ctor
}

void *ClientDataWeakGenericPtr::get()
{
    return this->ptr;
}
