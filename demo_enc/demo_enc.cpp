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
#include <cuda_runtime_api.h>
#include "cuj2k/j2k_encoder_extended.h"
#include "demo_enc.h"



/// Decoding work item.
struct work_item {
    long int index;                // frame index (assigned by input queue)
    void * custom_data_ptr;        // custom pointer associated with the image
    void * out_buffer_ptr;         // output buffer pointer
    size_t out_buffer_size;        // output buffer size
    int subsampled;                // true if subsampled
    const void * src_ptr;          // source data pointer
    j2k_image_params params;       // encoding parameters for the image
    int out_size;                  // output codestream size or -1 for error
};



/// Parameters of encoding thread.
struct thread_params {
    // index of GPU to be used by the thread
    const int gpu_idx;
    
    // pointer to shared structure with I/O queues
    demo_enc * const enc;
    
    // index of work item to be pushed
    int work_item_idx;
    
    // thread parameters constructor
    thread_params(int gpu_idx, demo_enc * enc, const int work_item_idx)
            : gpu_idx(gpu_idx), enc(enc), work_item_idx(work_item_idx) {}
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



/// Multiple-GPU JPEG 2000 encoder instance type for 2012 demo.
struct demo_enc {
    // array of encoder pointers for each GPU
    std::vector<pthread_t> threads;
    
    // input and output queues
    thread_safe_input_queue input;
    thread_safe_output_queue output;
    
    // JPEG 2000 encoder parameters
    j2k_encoder_params enc_params;
};



/// Called when encoder needs more images.
static int on_input(j2k_encoder * encoder,
                    void * user_callback_data,
                    int should_block) {
    // cast parameter to encoder object
    demo_enc * const enc = (demo_enc*)user_callback_data;
    
    // either wait for next work item or only look if there is one ready
    work_item * const item = should_block
            ? enc->input.get() : enc->input.try_get();
    
    // null item means "should stop" or "have no input yet"
    if(0 == item) {
        return should_block ? 0 : 1;
    }
    
    // submit the image
    if(0 != j2k_encoder_set_input(encoder, item->src_ptr,
                                  J2K_FMT_R10_G10_B10_X2_L,
                                  &item->params, item)) {
        // error => return work item back with error code set
        item->out_size = -1;
        enc->output.put(item);
    }
    
    // indicate that there may be more images to be decoded
    return 1;
}



/// Called when encoder has another frame encoded.
static void on_output(j2k_encoder * encoder,
                      void * user_callback_data,
                      void * user_input_data) {
    // get pointer to decoded work item and queues
    demo_enc * const dec = (demo_enc*)user_callback_data;
    work_item * const item = (work_item*)user_input_data;
    
    // get the image
    size_t out_size = 0;
    if(0 != j2k_encoder_get_output(encoder, item->out_buffer_ptr,
                                   item->out_buffer_size, &out_size)) {
        item->out_size = -1;
    } else {
        item->out_size = out_size;
    }
    
    // set status of the work item and enqueue it
    dec->output.put(item);
}



/// Encoding thread implementation.
static void * enc_thread_impl(void * data) {
    // get parameters
    const thread_params * const params = (thread_params*) data;
    
    // initialization status and decoder instance pointer
    bool error = false;
    j2k_encoder * enc = 0;
    
    // select CUDA device
    if(cudaSuccess != cudaSetDevice(params->gpu_idx)) {
        error = true;
    } else {
        // create and check decoder instance
        enc = j2k_encoder_create(&params->enc->enc_params);
        if(0 == enc) {
            error = true;
        }
    }
    
    // send initialization result work item to main thread using output queue
    work_item * const item = new work_item;
    item->out_size = error ? -1 : 0;
    item->index = params->work_item_idx;
    params->enc->output.put(item);
    
    // start decoding if no error occured
    if(!error) {
        j2k_encoder_run(enc, params->enc, on_input, on_output, 0);
    }
    
    // destroy all stuff
    j2k_encoder_destroy(enc);
    delete params;
    
    // return value ignored
    return 0;
}



/// Creates and initializes new instance of JPEG 2000 encoder.
/// Output codestreams are 4K DCI compatible (24fps 2K for subsampled frames).
/// @param gpu_indices_ptr   pointer to array of indices of GPUs to be used 
///                          for encoding or null to use all available GPUs
/// @param gpu_indices_count count of GPU indices (unused if pointer is null)
/// @param size_x            image width in pixels
/// @param size_y            image height in pixels
/// @param dwt_level_count   number of DWT decomposition levels (5 is OK for 4K)
/// @param max_quality       maximal quality (0.0f to 1.2f, limits buffers size)
/// @return either pointer to new instance of encoder or null if error occured
struct demo_enc * demo_enc_create(const int * gpu_indices_ptr,
                                  int gpu_indices_count,
                                  int size_x,
                                  int size_y,
                                  int dwt_level_count,
                                  float quality_upper_bound) {
    demo_enc * enc = 0;
    try {
        // create the instance
        enc = new demo_enc;
        if(0 == enc) {
            throw "new demo_enc";
        }
        
        // initialize encoder parameters
        j2k_encoder_params_set_default(&enc->enc_params);
        enc->enc_params.compression = CM_LOSSY_FLOAT;
        enc->enc_params.size.width = size_x;
        enc->enc_params.size.height = size_y;
        enc->enc_params.bit_depth = 10;
        enc->enc_params.is_signed = 0;
        enc->enc_params.comp_count = 3;
        enc->enc_params.resolution_count = dwt_level_count + 1;
        enc->enc_params.progression_order = PO_CPRL;
        enc->enc_params.quality_limit = quality_upper_bound;
        enc->enc_params.mct = 1;
        enc->enc_params.use_sop = 0;
        enc->enc_params.use_eph = 0;
        enc->enc_params.print_info = 0;
        enc->enc_params.capabilities = J2K_CAP_DCI_4K;
        enc->enc_params.out_bit_depth = 12;
        
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
        
        // start one thread for each listed GPU
        for(int i = gpu_indices_count; i--;) {
            // parameters for the thread
            const int gpu_idx = gpu_indices_ptr[i];
            thread_params * const params = new thread_params(gpu_idx, enc, i);
            
            // create the thread
            pthread_t thread;
            if(pthread_create(&thread, 0, enc_thread_impl, (void*)params)) {
                delete params;
                throw "pthread_create";
            }
            
            // rememeber thread's ID
            enc->threads.push_back(thread);
        }
        
        // collect responses of all threads (all should be 0 if initialized OK)
        int status = 0;
        for(int remaining = enc->threads.size(); remaining--; ) {
            const work_item * const item = enc->output.get();
            status |= item->out_size;
            delete item;
        }
        if(status) {
            throw "thread initialization";
        }
        enc->output.reset();
        
        // everything OK
        return enc;
    } catch (...) {
        if(enc) {
            demo_enc_stop(enc);
            demo_enc_destroy(enc);
        }
        return 0;  // allocation error
    }
}



/// Releases all resources of the encoder instance. 
/// Effects are undefined if any thread waits for output when this is called.
/// @param enc_ptr pointer to encoder instance
void demo_enc_destroy(struct demo_enc * enc_ptr) {
    // stop threads
    demo_enc_stop(enc_ptr);
    
    // wait for all encoder threads to stop
    for(int thread_idx = enc_ptr->threads.size(); thread_idx--;) {
        pthread_join(enc_ptr->threads[thread_idx], 0);
    }
    
    // destroy the instance
    delete enc_ptr;
}



/// Submits frame for encoding.
/// @param enc_ptr          pointer to encoder instance
/// @param custom_data_ptr  custom pointer associated with frame
/// @param out_buffer_ptr   pointer to ouptut buffer
/// @param out_buffer_size  ouptut buffer capacity (in bytes)
/// @param src_ptr          pointer to source RGB data: 10 bits per sample,
///                         each pixel packed to MSBs of 32bit little endian 
///                         integer (with 2 LSBs unused), without any padding
/// @param required_size    required output size of the encoded frame (in bytes)
///                         or 0 for unlimited size (NOTE: actual output may be 
///                         smaller or even slightly bigger)
/// @param quality          encoded frame quality:
///                             0.1f = poor
///                             0.7f = good
///                             1.2f = perfect
///                         (also bound by encoder-creation-time quality limit)
/// @param subsampled       0 for full resolution frame (same as input)
///                         or nonzero to subdivide input data along both axes
void demo_enc_submit(struct demo_enc * enc_ptr,
                     void * custom_data_ptr,
                     void * out_buffer_ptr,
                     int out_buffer_size,
                     const void * src_ptr,
                     int required_size,
                     float quality,
                     int subsampled) {
    // compose new work item
    work_item * const item = new work_item;
    j2k_image_params_set_default(&item->params);
    item->index = -1;
    item->custom_data_ptr = custom_data_ptr;
    item->out_buffer_ptr = out_buffer_ptr;
    item->out_buffer_size = out_buffer_size;
    item->subsampled = subsampled;
    item->src_ptr = src_ptr;
    item->params.output_byte_count = required_size;
    item->params.quality = quality;
    item->out_size = -1;
    
    // submit the work item into the queue
    enc_ptr->input.put(item);
}



/// Unblocks all waiting threads and stops encoding.
/// (Indicated by return value of demo_enc_wait.)
/// @param enc_ptr  pointer to encoder instance
void demo_enc_stop(struct demo_enc * enc_ptr) {
    enc_ptr->input.stop();
    enc_ptr->output.stop();
}



/// Waits for next encoded image of for encoder deallocation.
/// @param enc_ptr             pointer to encoder instance
/// @param custom_data_ptr_out null or pointer to pointer, where custom data 
///                            pointer associated with the frame is written
/// @param out_buffer_ptr_out  null or pointer to pointer, where provided 
///                            output buffer pointer is written
/// @param src_ptr_out         null or pointer to pointer, where provided 
///                            input data pointer is written
/// @return positive output size (in bytes) if frame encoded correctly,
///         0 if encoder was stopped while waiting (outputs are undefined),
///         -1 if error occured when encoding the frame
int demo_enc_wait(struct demo_enc * enc_ptr,
                  void ** custom_data_ptr_out,
                  void ** out_buffer_ptr_out,
                  const void ** src_ptr_out) {
    // Wait for next frame
    work_item * const item = enc_ptr->output.get();
    
    // the decoder was stopped while waiting if item is null
    if(0 == item) {
        return 0;
    }
    
    // fill in output parameters
    if(custom_data_ptr_out) {
        *custom_data_ptr_out = item->custom_data_ptr;
    }
    if(out_buffer_ptr_out) {
        *out_buffer_ptr_out = item->out_buffer_ptr;
    }
    if(src_ptr_out) {
        *src_ptr_out = item->src_ptr;
    }
    
    // remember correct result code according to decoding result 
    // before destroying the work item structure
    const int result = item->out_size;
    delete item;
    return result;
}

