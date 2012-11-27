///
/// @file    j2kd_buffer.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Automatically resizing buffer.
///

#ifndef J2KD_BUFFER_H
#define J2KD_BUFFER_H

// #include "j2kd_type.h"
#include "j2kd_util.h"
#include "j2kd_error.h"
#include "j2kd_cuda.h"



namespace cuj2kd {



/// Single buffer, with its current usage and total capacity.
template <typename T>
class Buffer {
protected:
    T * items;          ///< pointer to buffer in main system memory

private:    
    size_t capacity;    ///< total number of items in the buffer
    size_t usage;       ///< number of items currently used in buffers
    
protected:
    /// Prepares at least given number of items.
    void resize(const size_t count) {
        if(capacity < count) {
            // allocate new buffer
            capacity = count;
            T * const newPtr = (T*)mallocCPU(count * sizeof(T));
            
            // copy existing items and release old buffer
            if(items) {
                if(usage) {
                    memcpy(newPtr, items, sizeof(T) * usage);
                }
                freeCPU(items);
            }
            items = newPtr;
        }
    }
    
public:
    /// Creates new buffer with some initial capacity.
    Buffer(size_t n = 16) : items(0), capacity(0), usage(0) { resize(n); }
    
    /// Destroys the buffer, freeing the item buffer.
    ~Buffer() { if(items) freeCPU(items); }
    
    /// Marks all items as free.
    void clear() { usage = 0; }
    
    /// Reserves one item at the end of the buffer, returning its index,
    /// possibly extending the buffer.
    size_t reserveIdx() {
        if(usage >= capacity) { resize(capacity * 2); }
        return usage++;
    }
    
    /// Reserves more items at the end of the buffer, returning index 
    /// of first of them.
    /// @param count number of items to be reserved (must be positive)
    size_t reserveMore(const size_t count) {
        // TODO: reimplement
        size_t index = 0;
        for(size_t i = count; i--; ) { index = reserveIdx(); }
        return index + 1 - count;
    }
    
    /// Reserves one item at the end of the buffer, returning pointer to it,
    /// possibly extending the buffer.
    T * reservePtr() {
        const size_t idx = reserveIdx();  // this may change ptr
        return items + idx;
    }
    
    /// Provides access to items in the buffer.
    T & operator [] (const int index) const { return items[index]; }
    
    /// Gets current usage count.
    size_t count() const { return usage; }
}; // end of class Buffer



/// Adds second GPU buffer to the CPU buffer.
template <typename T>
class BufferPair : public Buffer<T> {
private:
    T * ptrGPU;      ///< pointer to GPU copy of the buffer
    size_t sizeGPU;  ///< size of the GPU copy (in bytes)
    
    /// Resizes the GPU buffer to be able to contain all items of CPU buffer.
    void resizeGPU() {
        const size_t requiredSize = max((size_t)1, this->count() * sizeof(T));
        if(requiredSize > sizeGPU) {
            if(ptrGPU) {
                freeGPU(ptrGPU);
                ptrGPU = 0;
                sizeGPU = 0;
            }
            ptrGPU = (T*)mallocGPU(requiredSize);
            sizeGPU = requiredSize;
        }
    }
    
public:
    /// Creates a new pair of buffers with some initial capacity.
    BufferPair(const size_t n = 16) : Buffer<T>(n), ptrGPU(0), sizeGPU(0) { }
    
    /// Releases resources associated wiht the buffer.
    ~BufferPair() { if(ptrGPU) freeGPU(ptrGPU); }
    
    /// Copies current contents of the buffer to GPU buffer 
    /// and gets the GPU pointer.
    T * copyToGPUAsync(const cudaStream_t & stream) {
        // resize the GPU buffer to fit all items.
        resizeGPU();
        asyncMemcpy(this->items, ptrGPU, this->count() * sizeof(T), true, stream);
        return ptrGPU;
    }
    
    /// Gets GPU pointer.
    T * getPtrGPU() const { return ptrGPU; }
}; // end of class BufferPair



/// Input and output buffer pair in GPU memory.
template <typename T>
class IOBufferGPU {
private:
    T * ptrs[2];          // pointers to two buffers
    size_t capacities[2]; // capacities of two buffers
    int inputBufferIdx;   // current index of input buffer (0 or 1)
    
    /// Resizes given buffer to be big enough for given number of bytes.
    void resize(const int bufferIdx, const size_t byteCount) {
        if(capacities[bufferIdx] < byteCount) {
            if(ptrs[bufferIdx]) {
                freeGPU(ptrs[bufferIdx]);
                ptrs[bufferIdx] = 0;
                capacities[bufferIdx] = 0;
            }
            ptrs[bufferIdx] = (T*)mallocGPU(byteCount);
            capacities[bufferIdx] = byteCount;
        }
    }
public:
    /// Initializes pair of GPU buffers.
    IOBufferGPU() {
        ptrs[0] = 0;
        ptrs[1] = 0;
        capacities[0] = 0;
        capacities[1] = 0;
        reset();
    }
    
    /// Releases all resources aassociated with buffers.
    ~IOBufferGPU() {
        if(ptrs[0]) {
            freeGPU(ptrs[0]);
        }
        if(ptrs[1]) {
            freeGPU(ptrs[1]);
        }
    }
    
//     /// Resizes input GPU buffer to fit at least given number of bytes.
//     /// Contents of newly resized input buffer are undefined.
//     void inResize(const size_t byteCount) {
//         resize(inputBufferIdx, byteCount);
//     }

    /// Sets first buffer as input and the other as output.
    void reset() {
        inputBufferIdx = 0;
    }
    
    /// starts asynchrnonous copy of input data from given CPU buffer
    /// into input GPU buffer in specified CUDA stream.
    /// @param inPtr   pointer to input data in main system memory
    /// @param inSize  size of input in bytes
    /// @param stream  CUDA stream for async copy operation
    void inAsyncLoad(const T * const inPtr,
                     const size_t inSize,
                     const cudaStream_t & stream) {
        resize(inputBufferIdx, inSize);
        asyncMemcpy(inPtr, ptrs[inputBufferIdx], inSize, true, stream);
    }
    
    /// Resizes output GPU buffer to fit at least given number of bytes.
    /// Contents of newly resized output buffer are undefined.
    void outResize(const size_t byteCount) {
        resize(inputBufferIdx ^ 1, byteCount);
    }
    
    /// Swaps the two buffer, so that input buffer becomes output buffer
    /// and vice versa. Contents and sizes of both buffes are left unchanged.
    void swap() {
        inputBufferIdx = inputBufferIdx ^ 1;
    }
    
    /// @return pointer to input buffer in GPU memory
    const T * inPtr() const {
        return ptrs[inputBufferIdx];
    }
    
    /// @return pointer to mutable input buffer in GPU memory
    T * mutableInPtr() const {
        return ptrs[inputBufferIdx];
    }
    
    /// @return pointer to output buffer in GPU memory
    T * outPtr() const {
        return ptrs[inputBufferIdx ^ 1];
    }
};



/// GPU buffer with autoresizing and place reservation.
class GPUBuffer {
private:
    size_t capacity;      ///< item capacity of currently allocated buffer
    void * ptr;              ///< pointer to GPU buffer
public:
    /// initializes the buffer.
    GPUBuffer() : capacity(0), ptr(0) {}
    
    /// Releases associated resources.
    ~GPUBuffer() { if(ptr) freeGPU(ptr); }
    
    /// Possibly reallocates buffer to be big enough and returns the pointer.
    void * getPtr() const { return ptr; }
    
    /// Resizes the buffer to have at least given size in bytes.
    /// @param size bytes number of bytes to be reserved.
    /// @return pointer to the buffer in GPU memory
    void * resize(const size_t size) {
        if(capacity < size) {
            if(ptr) {
                freeGPU(ptr);
                ptr = 0;
                capacity = 0;
            }
            ptr = mallocGPU(size);
            capacity = size;
        }
        return ptr;
    }
}; // end of class GPUBuffer



/// CPU pool of some structure instances. Tracks all instances 
/// and automatically allocates/frees them.
template <typename T>
class Pool {
private:
    /// internal type of instance of the object.
    struct Instance : public T {
        Instance * next; /// pointer to next instance or null
    }; // end of struct instance
    
    Instance * free;  ///< chain of free instances or null
    Instance * used;  ///< chain of used instances or null

public:
    /// Initializies pointers to instances.
    Pool() : free(0), used(0) {}
    
    /// Deletes all instances (both used and free).
    ~Pool() { 
        reuse();
        while(free) {
            Instance * const i = free;
            free = free->next;
            delete i;
        }
    }
    
    /// Gets pointer to some unused instance.
    T* get() {
        Instance * i = free;
        if(i) {
            free = free->next;
        } else {
            i = new Instance();
        }
        i->next = used;
        used = i;
        return i;
    }
    
    /// Returns all instances for reuse.
    void reuse() {
        while(used) {
            Instance * i = used;
            used = used->next;
            i->next = free;
            free = i;
        }
    }
}; // end of class Pool



} // end of namespace cuj2kd


#endif // J2KD_BUFFER_H
