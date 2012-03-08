#include "../include/ClientDataIntPair.h"

ClientDataIntPair::ClientDataIntPair(int a, int b) :
    first_int(a), second_int(b)
{
    //ctor
}

ClientDataIntPair::ClientDataIntPair()
{
    //ctor
}

ClientDataIntPair::~ClientDataIntPair()
{
    //dtor
}

int ClientDataIntPair::first()
{
    return first_int;
}

int ClientDataIntPair::second()
{
    return second_int;
}

void ClientDataIntPair::setFirst(int val)
{
    first_int = val;
}

void ClientDataIntPair::setSecond(int val)
{
    second_int = val;
}
