#ifndef CONNECTIONCLOSEDEXCEPTION_H
#define CONNECTIONCLOSEDEXCEPTION_H

#include <exception>


class ConnectionClosedException : public std::exception
{
    public:
        ConnectionClosedException();
        virtual ~ConnectionClosedException() throw();
        virtual const char* what() const throw();
    protected:
    private:
};

#endif // CONNECTIONCLOSEDEXCEPTION_H
