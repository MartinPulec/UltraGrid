#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <cuda_runtime.h>
#include <iostream>
#include <map>

#include "cuda_memory_pool.h"

using namespace std;

class cuda_memory_pool {
        public:
                cuda_memory_pool() {
                        pthread_mutex_init(&mutex, NULL);
                }

                virtual ~cuda_memory_pool() {
                        pthread_mutex_destroy(&mutex);
                        for(multimap<int, void *>::iterator it = available_memory.begin();
                                        it != available_memory.end();
                                        ++it) {
                                cudaFreeHost(it->second);
                        }
                }

                void *alloc(size_t size) {
                        void *buffer = NULL;

                        pthread_mutex_lock(&mutex);
                        multimap<int, void *>::iterator it = available_memory.lower_bound(size);
                        if(it != available_memory.end()) {
                                buffer = it->second;
                                available_memory.erase(it);
                        }
                        pthread_mutex_unlock(&mutex);

                        if(buffer) {
                                return buffer;
                        }

                        cudaError_t res = cudaHostAlloc(&buffer, size,
                                        cudaHostAllocPortable);
                        if(res == cudaSuccess) {
                                return buffer;
                        } else {
                                return NULL;
                        }
                }

                void free(void *ptr, size_t size)
                {
                        pthread_mutex_lock(&mutex);
                        available_memory.insert(std::pair<int,void *>(size,ptr));
                        pthread_mutex_unlock(&mutex);
                }
        private:
                multimap<int, void *> available_memory;
                pthread_mutex_t      mutex;
};

static struct cuda_memory_pool pool;

void * cuda_pool_alloc(size_t size)
{
        return pool.alloc(size);
}

void cuda_pool_dispose(void *ptr, size_t size)
{
        pool.free(ptr, size);
}

