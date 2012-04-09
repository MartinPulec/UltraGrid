#include "../include/Observable.h"

#include "../include/Observer.h"

Observable::Observable()
{
    //ctor
}

Observable::~Observable()
{
    //dtor
}

void Observable::registerObserver(Observer * observer)
{
    observers_.insert(observers_.begin(), observer);
}

void Observable::unregisterObserver(Observer * observer)
{
    observers_.remove(observer);
}

void Observable::notifyObservers()
{
    for(std::list <Observer *>::iterator it = observers_.begin(); it != observers_.end(); ++ it)
    {
        (*it)->NotifyObserver(this);
    }

}
