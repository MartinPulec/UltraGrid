#include "../include/ClientDataCStr.h"

#include <string.h>
#include <stdlib.h>

ClientDataCStr::ClientDataCStr(const char *data)
{
    this->cstring = strdup(data);
}

ClientDataCStr::~ClientDataCStr()
{
    free(this->cstring);
}

char *ClientDataCStr::get()
{
    return this->cstring;
}
