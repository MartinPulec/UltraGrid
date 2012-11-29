#ifdef __cplusplus
extern "C" {
#endif


void cuda_memory_pool_init();
void cuda_memory_pool_destroy();

void * cuda_alloc(size_t size);
void cuda_free(void *ptr, size_t size);

#ifdef __cplusplus
}
#endif

