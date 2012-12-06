///
/// @file    demo_dec.cpp
/// @author  Martin Jirman (jirman@cesnet.cz)
/// @brief   Multiple GPU decoder wrapper for 2012 demo.
/// 

#include <vector>
#include <queue>
#include <map>
#include <cstdio>
#include <pthread.h>
#include <inttypes.h>
#include <cuda_runtime_api.h>
#include "j2kd_api.h"
#include "demo_dec.h"


/// Decoding work item.
struct work_item {
    long int index;                // frame index (assigned by input queue)
    j2kd_component_format fmt[12]; // formatting info (3 or 3x4 components)
    void * custom_data_ptr;        // custom pointer associated with the image
    void * out_buffer_ptr;         // output buffer pointer
    const void * codestream_ptr;   // codestream pointer
    int codestream_size;           // codestream size
    int status;                    // 0 if OK, 2 for decoding error
    size_t out_buffer_size;        // expected size of the output buffer
    int size_x;                    // output width
    int size_y;                    // output height
};


/// Thread safe input queue, assigns serial numbers to input frames.
class thread_safe_input_queue {
private:
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    volatile bool running;
    volatile long int counter;
    std::queue<work_item*> items;
public:
    /// Initializes input queue.
    thread_safe_input_queue() {
        if(pthread_mutex_init(&mutex, 0)) {
            throw "pthread_mutex_init";
        }
        if(pthread_cond_init(&cond, 0)) {
            throw "pthread_cond_init";
        }
        running = true;
        counter = 0;
    }
    
    /// Destroys the queue (should already be stopped).
    ~thread_safe_input_queue() {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&cond);
    }
    
    /// Puts new item into the queue assinging a number to it.
    void put(work_item * const item) {
        pthread_mutex_lock(&mutex);
        if(running) {
            item->index = counter++;
            items.push(item);
            pthread_cond_signal(&cond);
        }
        pthread_mutex_unlock(&mutex);
    }
    
    /// Waits for next item from the queue or null if stopped.
    work_item * get() {
        pthread_mutex_lock(&mutex);
        while(running && items.empty()) {
            pthread_cond_wait(&cond, &mutex);
        }
        work_item * item = 0;
        if(running) {
            item = items.front();
            items.pop();
        }
        pthread_mutex_unlock(&mutex);
        return item;
    }
    
    /// Gets either next item or null (if queue is empty) as soon as possible.
    work_item * try_get() {
        pthread_mutex_lock(&mutex);
        work_item * item = 0;
        if(running && !items.empty()) {
            item = items.front();
            items.pop();
        }
        pthread_mutex_unlock(&mutex);
        return item;
    }
    
    /// Stops the queue, unblocking all waiting threads.
    void stop() {
        pthread_mutex_lock(&mutex);
        running = false;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mutex);
    }
};


/// Reorders output to get them in input order.
struct thread_safe_output_queue {
private:
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    volatile bool running;
    volatile long int next_index;
    std::map<long int, work_item*> items;
public:
    /// Initializes input queue.
    thread_safe_output_queue() {
        if(pthread_mutex_init(&mutex, 0)) {
            throw "pthread_mutex_init";
        }
        if(pthread_cond_init(&cond, 0)) {
            throw "pthread_cond_init";
        }
        running = true;
        next_index = 0;
    }
    
    /// Destroys the queue (should already be stopped).
    ~thread_safe_output_queue() {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&cond);
    }
    
    /// Enqueues work item if not stopped.
    void put(work_item * const item) {
        pthread_mutex_lock(&mutex);
        if(running) {
            items[item->index] = item;
            pthread_cond_signal(&cond);
        }
        pthread_mutex_unlock(&mutex);
    }
    
    /// Waits for next work item or for queue stopping.
    work_item * get() {
        pthread_mutex_lock(&mutex);
        std::map<long int, work_item*>::iterator pos;
        const long int required_index = next_index++;
        while(running && items.end() == (pos = items.find(required_index))) {
            pthread_cond_wait(&cond, &mutex);
        }
        work_item * item = 0;
        if(running) {
            item = (*pos).second;
            items.erase(pos);
        }
        pthread_mutex_unlock(&mutex);
        return item;
    }
    
    /// Resets next work item index to 0.
    void reset() {
        next_index = 0;
    }
    
    /// Stops the queue, unblocking all waiting threads.
    void stop() {
        pthread_mutex_lock(&mutex);
        running = false;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mutex);
    }
};


/// Parameters of decoding thread.
struct thread_params {
    // index of GPU to be used by the thread
    const int gpu_idx;
    
    // pointer to shared structure with I/O queues
    demo_dec * const dec;
    
    // index of work item to be pushed
    int work_item_idx;
    
    // thread parameters constructor
    thread_params(int gpu_idx, demo_dec * dec, const int work_item_idx)
            : gpu_idx(gpu_idx), dec(dec), work_item_idx(work_item_idx) {}
};


/// Multiple-GPU JPEG 2000 decoder instance type for 2012 demo.
struct demo_dec {
    // array of decoder pointers for each GPU
    std::vector<pthread_t> threads;
    
    // input and output queues
    thread_safe_input_queue input;
    thread_safe_output_queue output;
};


/// Decoder input request callback.
static int on_in_begin(void * custom_callback_ptr,
                       void ** custom_image_ptr_out,
                       const void ** codestream_ptr_out,
                       size_t * codestream_size_out,
                       const j2kd_component_format ** comp_format_ptr_out,
                       int * comp_format_count_out,
                       int should_block) {
    // cast parameter to decoder object
    demo_dec * const dec = (demo_dec*)custom_callback_ptr;
    
    // either wait for next work item or only look if there is one ready
    work_item * const item = should_block
            ? dec->input.get() : dec->input.try_get();
    
    // null item means "should stop" or "have no items right now"
    if(0 == item) {
        // indicate whether there may be more items later
        // (there may be more items if the waiting was blocking)
        return should_block ? 0 : 1;
    }
    
    // set output parameters
    *custom_image_ptr_out = (void*)item;
    *codestream_ptr_out = item->codestream_ptr;
    *codestream_size_out = item->codestream_size;
    *comp_format_ptr_out = item->fmt;
    *comp_format_count_out = 3;
    
    // indicate that there may be more images to be decoded
    return 1;
}


/// Input loading end callback (input buffer it not needed anymore)
static void on_in_end(void * custom_callback_ptr,
                      void * custom_image_ptr,
                      const void * codestream_ptr) {
    // nothing to be done here
}


/// Output buffer request callback.
static void on_out(void * custom_callback_ptr,
                   void * custom_image_ptr,
                   void ** output_ptr_out,
                   size_t * output_capacity_out,
                   int * output_in_device_mem_out) {
    // get work item info
    work_item * const item = (work_item*)custom_image_ptr;
    
    // set output parameters
    *output_ptr_out = item->out_buffer_ptr;
    *output_capacity_out = item->out_buffer_size;
    *output_in_device_mem_out = 0;  // output buffer is in host memory
}


/// "Decoding done" callback.
static void on_end(void * custom_callback_ptr,
                   void * custom_image_ptr,
                   j2kd_status_code status,
                   const j2kd_component_format * comp_format_ptr,
                   void * output_ptr) {
    // get pointer to decoded work item and queues
    demo_dec * const dec = (demo_dec*)custom_callback_ptr;
    work_item * const item = (work_item*)custom_image_ptr;
    
    // set status of the work item and enqueue it
    item->status = status == J2KD_OK ? 0 : 2; // 2 == decoding error, 0 == OK
    dec->output.put(item);
}


/// Decoding thread implementation.
static void * dec_thread_impl(void * data) {
    // get parameters
    const thread_params * const params = (thread_params*) data;
    
    // select CUDA device and create decoder instance
    j2kd_decoder * dec = 0;
    if(cudaSuccess == cudaSetDevice(params->gpu_idx)) {
        dec = j2kd_create(0);
    }
    
    // send initialization result work item to main thread using output queue
    work_item * const item = new work_item;
    item->status = dec ? 0 : -1;
    item->index = params->work_item_idx;
    params->dec->output.put(item);
    
    // start decoding if no error occured
    if(dec) {
        if(J2KD_OK != j2kd_run(dec, on_in_begin, on_in_end, on_out,
                               0, on_end, params->dec)) {
            printf("Decoder ERROR: %s.\n", j2kd_status(dec));
        }
        
        // destroy the decoder instance
        j2kd_destroy(dec);
    }
    delete params;
    
    // return value ignored
    return 0;
}


///  Creates and initializes new instance of JPEG 2000 decoder.
/// @param gpu_indices_ptr   pointer to array of indices of GPUs to be used 
///                          for decoding or null to use all available GPUs
/// @param gpu_indices_count count of GPU indices (unused if pointer is null)
/// @return either pointer to new instance of decoder or null if error occured
demo_dec * demo_dec_create(const int * gpu_indices_ptr, int gpu_indices_count) {
    demo_dec * dec = 0;
    try {
        // create the instance
        dec = new demo_dec;
        if(0 == dec) {
            throw "new demo_dec";
        }
        
        // compose list of all available GPU indices if not provided
        std::vector<int> indices;
        if(0 == gpu_indices_ptr) {
            // get GPU count
            int gpu_count;
            if(cudaSuccess != cudaGetDeviceCount(&gpu_count)) {
                throw "cudaGetDeviceCount";
            }
            
            // add all usable GPUs to the list
            for(int gpu_idx = gpu_count; gpu_idx--;) {
                cudaDeviceProp prop;
                if(cudaSuccess != cudaGetDeviceProperties(&prop, gpu_idx)) {
                    throw "cudaGetDeviceProperties";
                }
                if(prop.major >= 2) {
                    printf("Using device #%d: %s for decoding.\n",
                           gpu_idx, prop.name);
                    indices.push_back(gpu_idx);
                }
            }
            
            // replace the list with newly composed one
            gpu_indices_ptr = &indices[0];
            gpu_indices_count = indices.size();
        }
        
        // check GPU count
        if(gpu_indices_count < 1) {
            throw "no devices found";
        }
        
        // start one thread for each listed GPU
        for(int i = gpu_indices_count; i--;) {
            // parameters for the thread
            const int gpu_idx = gpu_indices_ptr[i];
            thread_params * const params = new thread_params(gpu_idx, dec, i);
            
            // create the thread
            pthread_t thread;
            if(pthread_create(&thread, 0, dec_thread_impl, (void*)params)) {
                delete params;
                throw "pthread_create";
            }
            
            // rememeber thread's ID
            dec->threads.push_back(thread);
        }
        
        // collect responses of all threads (all should be 0 if initialized OK)
        int status = 0;
        for(int remaining = dec->threads.size(); remaining--; ) {
            const work_item * const item = dec->output.get();
            status |= item->status;
            delete item;
        }
        if(status) {
            throw "thread initialization";
        }
        dec->output.reset();
        
        // everything OK
        return dec;
    } catch (const char * message) {
        if(message) {
            printf("Decoder initialization error: %s.\n", message);
        }
        if(dec) {
            demo_dec_stop(dec);
            demo_dec_destroy(dec);
        }
        return 0;  // allocation error
    }
}


/// Releases all resources of the instance. 
/// Effects are undefined if any thread waits for output when this is called.
/// @param dec_ptr pointer to decoder instance
void demo_dec_destroy(demo_dec * dec_ptr) {
    // stop threads
    demo_dec_stop(dec_ptr);
    
    // wait for all decoder threads to stop
    for(int thread_idx = dec_ptr->threads.size(); thread_idx--;) {
        pthread_join(dec_ptr->threads[thread_idx], 0);
    }
    
    // destroy the instance
    delete dec_ptr;
}


/// Submits frame for decoding.
/// @param dec_ptr         pointer to decoder instance
/// @param custom_data_ptr custom pointer associated with frame
/// @param out_buffer_ptr  pointer to ouptut buffer with sufficient capacity
/// @param codestream_ptr  pointer to JPEG 2000 codestream
/// @param codestream_size size of given codestream (in bytes)
void demo_dec_submit(demo_dec * dec_ptr,
                     void * custom_data_ptr,
                     void * out_buffer_ptr,
                     const void * codestream_ptr,
                     int codestream_size) {
    // compose new work item
    work_item * const item = new work_item;
    item->index = -1;  // frame index (assigned by input queue)
    item->status = -1; // set by "decoding end" callback
    item->custom_data_ptr = custom_data_ptr;  // custom pointer for the image
    item->codestream_ptr = codestream_ptr;    // codestream pointer
    item->codestream_size = codestream_size;  // codestream size
    item->out_buffer_ptr = out_buffer_ptr;    // output buffer pointer
    
    // get and check info about the image
    int comp_count;
    if(!demo_dec_image_info(codestream_ptr, codestream_size,
                            &comp_count, &item->size_x, &item->size_y)
            || comp_count != 3) {
        // let the decoding fail if codestream is not correct
        item->size_x = 0;
        item->size_y = 0;
        codestream_size = 0;
    }
    
    // output size (4 bytes per pixel)
    item->out_buffer_size = item->size_x * item->size_y * 4;
    
    // initialize format specification for each of 3 components
    for(int comp_idx = 3; comp_idx--;) {
        item->fmt[comp_idx].component_idx = comp_idx;
        item->fmt[comp_idx].type = J2KD_TYPE_INT32;
        item->fmt[comp_idx].offset = 0;
        item->fmt[comp_idx].stride_x = 1;
        item->fmt[comp_idx].stride_y = item->size_x;
        item->fmt[comp_idx].bit_depth = 10;
        item->fmt[comp_idx].is_signed = 0;
        item->fmt[comp_idx].final_shl = (2 - comp_idx) * 10;
        item->fmt[comp_idx].combine_or = comp_idx;
    }
    
    // submit the work item into the queue
    dec_ptr->input.put(item);
}


/// Unblocks all waiting threads and stops decoding.
/// (Indicated by return value of demo_dec_wait.)
/// @param dec_ptr  pointer to decoder instance
void demo_dec_stop(demo_dec * dec_ptr) {
    dec_ptr->input.stop();
    dec_ptr->output.stop();
}


/// Waits for next decoded image of for decoder deallocation.
/// @param dec_ptr             pointer to decoder instance
/// @param custom_data_ptr_out null or pointer to pointer, where custom data 
///                            pointer associated with the frame is written
/// @param out_buffer_ptr_out  null or pointer to pointer, where provided 
///                            output buffer pointer is written
/// @param codestream_ptr_out  null or pointer to pointer, where provided 
///                            input codestream pointer is written
/// @return 0 if frame decoded correctly,
///         1 if decoder was stopped while waiting (outputs are undefined),
///         2 if error occured when decoding the frame
int demo_dec_wait(demo_dec * dec_ptr,
                  void ** custom_data_ptr_out,
                  void ** out_buffer_ptr_out,
                  const void ** codestream_ptr_out) {
    // Wait for next frame
    work_item * const item = dec_ptr->output.get();
    
    // the decoder was stopped while waiting if item is null
    if(0 == item) {
        return 1;
    }
    
    // fill in output parameters
    if(custom_data_ptr_out) {
        *custom_data_ptr_out = item->custom_data_ptr;
    }
    if(out_buffer_ptr_out) {
        *out_buffer_ptr_out = item->out_buffer_ptr;
    }
    if(codestream_ptr_out) {
        *codestream_ptr_out = item->codestream_ptr;
    }
    
    // remember correct result code according to decoding result 
    // before destroying the work item structure
    const int result = item->status ? 2 : 0;
    delete item;
    return result;
}


/// Gets count of GPU threads of the decoder.
/// @return GPU decoding thread count of the decoder instance
int demo_dec_gpu_count(demo_dec * dec_ptr) {
    return (int)dec_ptr->threads.size();
}


/// Gets basic info about given codestream.
/// @param codestream_ptr   pointer to codestream
/// @param codestream_size  codestream size in bytes
/// @param comp_count_out   pointer to int where color component count is 
///                         written or null
/// @param size_x_out       null or pointer to int where image width is written
/// @param size_y_out       null or pointer to int where image height is written
/// @return 0 if input is definitely NOT valid JPEG 2000 codestream 
///         (outputs are undefined), nonzero if it may be valid
int demo_dec_image_info(const void * codestream_ptr,
                        int codestream_size,
                        int * comp_count_out,
                        int * size_x_out,
                        int * size_y_out) {
    // get info about codestream
    j2kd_image_info i;
    if(J2KD_OK != j2kd_get_image_info(codestream_ptr, codestream_size, &i)) {
        // indicate that the codestream is not a codestream :)
        return 0;
    }
    
    // fill in required output parameters
    if(comp_count_out) {
        *comp_count_out = i.comp_count;
    }
    if(size_x_out) {
        *size_x_out = i.image_end_x - i.image_begin_x;
    }
    if(size_y_out) {
        *size_y_out = i.image_end_y - i.image_begin_y;
    }
    
    // indicate that the codestreammay be a codestream
    return 1;
}

