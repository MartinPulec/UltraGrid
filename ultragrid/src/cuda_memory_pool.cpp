#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <cuda_runtime.h>
#include <stack>
#include <map>

#include "cuda_memory_pool.h"

using namespace std;

class cuda_memory_pool {
        public:
                cuda_memory_pool() {
                        pthread_spin_init(&spin, PTHREAD_PROCESS_PRIVATE);
                }

                virtual ~cuda_memory_pool() {
                        pthread_spin_destroy(&spin);
                }

                void *alloc(size_t size) {
                        void *buffer = NULL;

                        pthread_spin_lock(&spin);
                        if(available_memory.find(size) != available_memory.end()) {
                                if(!available_memory[size].empty()) {
                                        buffer = available_memory[size].top();
                                        available_memory[size].pop();
                                }
                        }
                        pthread_spin_unlock(&spin);

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
                        pthread_spin_lock(&spin);
                        if(available_memory.find(size) == available_memory.end()) {
                                available_memory[size] = stack<void *>();
                        }
                        available_memory[size].push(ptr);
                        pthread_spin_unlock(&spin);
                }
        private:
                map<int, stack<void *> > available_memory;
                pthread_spinlock_t      spin;
};

static struct cuda_memory_pool pool;

void * cuda_alloc(size_t size)
{
        return pool.alloc(size);
}

void cuda_free(void *ptr, size_t size)
{
        pool.free(ptr, size);
}

