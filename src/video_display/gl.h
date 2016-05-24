#include <cuda_wrapper.h>
#include <utils/video_frame_pool.h>

struct cuda_buffer_data_allocator {
        void *allocate(size_t size) {
                void *ptr;
                if (CUDA_WRAPPER_SUCCESS != cuda_wrapper_malloc_host(&ptr,
                                        size)) {
                        return NULL;
                }
                return ptr;
        }
        void deallocate(void *ptr) {
                cuda_wrapper_free(ptr);
        }
};

extern video_frame_pool<cuda_buffer_data_allocator> shared_pool;

