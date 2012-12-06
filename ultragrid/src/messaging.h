#ifndef _MESSAGING_H
#define _MESSAGING_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <list>
#include <string>

class message {
        public:
                virtual ~message() {
                }
};

struct text_message: public message
{
        std::string text;
};

class observer {
        public:
                virtual void notify(message *) = 0;
};

class message_manager {
        public:
                message_manager() {
                        pthread_mutex_init(&lock, NULL);
                }

                virtual ~message_manager() {
                        pthread_mutex_destroy(&lock);
                }

                void register_observer(observer *o) {
                        pthread_mutex_lock(&lock);
                        observers.push_back(o);
                        pthread_mutex_unlock(&lock);
                }

                void broadcast(class message *message) {
                        pthread_mutex_lock(&lock);
                        for(std::list<observer *>::iterator it = observers.begin();
                                        it != observers.end();
                                        ++it) {
                                (*it)->notify(message);
                        }

                        pthread_mutex_unlock(&lock);
                }

        private:
                pthread_mutex_t lock;
                std::list<observer *> observers;
};

extern class message_manager message_manager;

#endif
