#include "../include/ConnectionClosedException.h"

ConnectionClosedException::ConnectionClosedException()
{
    //ctor
}

ConnectionClosedException::~ConnectionClosedException() throw()
{
    //dtor
}

const char* ConnectionClosedException::what() const throw()
{
    return "Connection was closed";
}
