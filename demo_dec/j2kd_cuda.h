///
/// @file    j2kd_cuda.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Wrappers for common cuda calls.
///



#ifndef J2KD_CUDA_H
#define J2KD_CUDA_H

#include <cuda_runtime_api.h>
#include "j2kd_type.h"
#include "j2kd_error.h"


namespace cuj2kd {




/// Checks cuda return value, possibly throwing an exception, or nothing else.
/// @param code  cuda call return code
/// @param what  what call caused the error
inline void checkCudaCall(const cudaError_t & code, const char * const what) {
    if(cudaSuccess != code) {
        throw Error(J2KD_ERROR_CUDA, "CUDA call '%s' returned '%s'.",
                    what, cudaGetErrorString(code));
    }
}


/// Synchronizes with given stream and checks for errors.
/// @param what  what call caused the error
/// @param stream  cuda stream to run the memcpy in
inline void checkKernelCall(const char * what, const cudaStream_t & stream) {
    checkCudaCall(cudaStreamSynchronize(stream), what);
}


/// Starts asynchronous memcpy to or from GPU and checks the call result.
/// @param src     source buffer pointer
/// @param dest    destination buffer pointer
/// @param size    byte count
/// @param toGPU   true if source is in main system memory and dest is on GPU

inline void asyncMemcpy(const void * const src, void * const dest,
                        const size_t size, const bool toGPU,
                        const cudaStream_t & stream) {
    if(size) {
        const cudaMemcpyKind kind = toGPU ? cudaMemcpyHostToDevice
                                          : cudaMemcpyDeviceToHost;
        const char * what = toGPU ? "async memcpy H>D" : "async memcpy D>H";
        checkCudaCall(cudaMemcpyAsync(dest, src, size, kind, stream), what);
    }
}


/// Synchronization with some CUDA stream.
/// @param stream  ID of the stream
inline void streamSync(const cudaStream_t & stream) {
    checkCudaCall(cudaStreamSynchronize(stream), "cuda stream sync");
}


/// Starts synchronous memcpy to or from GPU and checks the call result.
/// @param src     source buffer pointer
/// @param dest    destination buffer pointer
/// @param size    byte count
/// @param toGPU   true if source is in main system memory and dest is on GPU
/// @param stream  cuda stream to run the memcpy in
inline void syncMemcpy(const void * const src, void * const dest,
                       const size_t size, const bool toGPU,
                       const cudaStream_t & stream) {
    if(size) {
        asyncMemcpy(src, dest, size, toGPU, stream);
        streamSync(stream);
    }
}


/// Starts synchronous memcpy to GPU sumbol and checks the call result.
/// @param src     source buffer pointer
/// @param dest    destination symbol
/// @param size    byte count
/// @param stream  cuda stream to run the memcpy in
template <typename SYMBOL_T>
inline void syncMemcpyToSymbol(const void * const src,
                               SYMBOL_T & symbol,
                               const size_t size,
                               const cudaStream_t & stream) {
    if(size) {
        cudaMemcpyToSymbolAsync(symbol, src, size, 0, cudaMemcpyHostToDevice, stream);
        streamSync(stream);
    }
}


/// Allocate GPU buffer.
/// @param size  size of the buffer
inline void * mallocGPU(const size_t size) {
    void * ptr;
    checkCudaCall(cudaMalloc(&ptr, size), "GPU malloc");
    return ptr;
}


/// Free GPU buffer.
/// @param ptr  pointer to GPU buffer
inline void freeGPU(void * const ptr) {
    checkCudaCall(cudaFree(ptr), "GPU free");
}


/// Allocate page-locked CPU buffer.
/// @param size  size of the buffer
inline void * mallocCPU(const size_t size) {
    void * ptr;
    checkCudaCall(cudaMallocHost(&ptr, size), "CPU malloc");
    return ptr;
}


/// Free CPU page-locked buffer.
/// @param ptr  pointer to CPU buffer
inline void freeCPU(void * const ptr) {
    checkCudaCall(cudaFreeHost(ptr), "CPU free");
}



} // end of namespace cuj2kd


#endif // J2KD_CUDA_H
