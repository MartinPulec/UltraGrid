#ifndef OBSERVER_H
#define OBSERVER_H

class Observable;

class Observer
{
    public:
        virtual void NotifyObserver(Observable *object) = 0;
    protected:
    private:
};

#endif // OBSERVER_H
