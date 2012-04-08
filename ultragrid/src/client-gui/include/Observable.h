#ifndef OBSERVABLE_H
#define OBSERVABLE_H

#include <list>

class Observer;

class Observable
{
    public:
        Observable();
        virtual ~Observable();

        void registerObserver(Observer * observer);
        void unregisterObserver(Observer * observer);
        void notifyObservers();
    protected:
    private:
        std::list <Observer *> observers_;
};

#endif // OBSERVABLE_H
