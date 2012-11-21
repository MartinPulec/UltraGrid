/* 
 * Copyright (c) 2011, Martin Jirman
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef J2K_ENCODER_EXTENDED_H
#define J2K_ENCODER_EXTENDED_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "j2k_encoder.h"




/** Types of special input formats supported by encoder. */
enum j2k_input_format
{
    /**
     * default interleaved format: RGB sample order, each sample is stored 
     * either in 1 byte (bit depth up to 8) or in 2 little endian bytes 
     * (bit depth 9 - 16) or in 4 little endian bytes (bit depth 17 - 32)
     */
    J2K_FMT_DEFAULT,
    
    /** 
     * 3 samples of each pixel stored in 32 bits 
     * (little endian, 3 x 10bit unsigned sample + 2 ignored bits): 
     *              +----------+----------+----------+--+
     *  contents:   | R (10 b) | G (10 b) | B (10 b) |??|
     *              +----------+----------+----------+--+
     *  bit #:       31      22 21      12 11       2 1 0
     */
    J2K_FMT_R10_G10_B10_X2_L,
    
    /** 
     * 3 samples of each pixel stored in 32 bits 
     * (big endian, 3 x 10bit unsigned sample + 2 ignored bits): 
     *              +----------+----------+----------+--+
     *  contents:   | R (10 b) | G (10 b) | B (10 b) |??|
     *              +----------+----------+----------+--+
     *  bit #:       31      22 21      12 11       2 1 0
     */
    J2K_FMT_R10_G10_B10_X2_B,
    
    /** unsigned little endian 16bit samples, interleaved, in RGB order */
    J2K_FMT_R16_G16_B16_L,
    
    /** unsigned big endian 16bit samples, interleaved, in RGB order */
    J2K_FMT_R16_G16_B16_B
};


/**
 * Extended encoder interface input callback type. Called by j2k_encoder_run 
 * when encoder needs another input image. Implementation of this callback 
 * should call j2k_encoder_set_input to actually set next input data and should
 * return as soon as possible. Otherwise, it would block the encoding process, 
 * because this callback is called in the thread, where corresponding 
 * j2k_encoder_run function runs.
 * @param encoder  pointer to instance which called the callback
 * @param user_callback_data  user pointer given by caller to j2k_encoder_run
 * @param should_block  nonzero if decoder has nothing else to do
 * @return nonzero if there will be more images, 0 if encoder should stop
 */
typedef int
(*j2k_encoder_in_callback)
(
    struct j2k_encoder * encoder,
    void * user_callback_data,
    int should_block
);


/**
 * Returns input buffer to caller after it is no longer needed for encoding.
 * Called from same thread, where corresponding j2k_encoder_run runs. Always 
 * called before next input callback is called (this means that only single 
 * buffer can be in use in each encoder instance at any time).
 * @param encoder  pointer to encoder instance
 * @param user_callback_data  user pointer given by caller to j2k_encoder_run
 * @param user_input_data  user poiner given to encoder together with buffer
 * @param data  pointer to returned buffer with input data
 */
typedef void
(*j2k_encoder_buffer_callback)
(
    struct j2k_encoder * encoder,
    void * user_callback_data,
    void * user_input_data,
    const void * data
);

/**
 * Extended encoder interface output callback type. Called by j2k_encoder_run
 * when encoder finished encoding of some image. Callback implementation should
 * use function j2k_encoder_get_output to get the output data. Implementation
 * of this callback should return as soon as possible, because the callback 
 * is called in the same thread, where j2k_encoder_run function runs and 
 * therefore blocks the encoding process.
 * @param encoder  encoder instance which called the callback
 * @param user_callback_data  user pointer given by caller to j2k_encoder_run
 * @param user_input_data  user pointer given to encoder together with image 
 */
typedef void
(*j2k_encoder_out_callback)
(
    struct j2k_encoder * encoder,
    void * user_callback_data,
    void * user_input_data
);


/** 
 * Encoder interface for image series encoding. Calls input and output 
 * callbacks in single run, interleaving GPU computation with data transfers 
 * for optimal GPU utilization. By calling input callbacks, it retrieves new
 * input images from caller and using output callback, it provides encoded 
 * codestreams to caller. If input callback signalizes it to stop, it encoded 
 * remaining images, calls correpsonding output callbacks and returns back 
 * to caller. Must be called in the same thread, where the encoder instance was 
 * created (because the CUDA context is bound to the thread). All callbacks are 
 * called in the same thread as this function. Caller provides pointer, whose 
 * value is later passed to all callbacks. Aditionally, caller provides
 * another pointer for each input image, which is passed to corresponding
 * output callback (and therefore can represent some image-specific data).
 *
 * Same encoder instance can be used for multiple j2k_encoder_run and/or 
 * j2k_encoder_compress function calls as long as calls do not overlap
 * (can be called again after previous call returns).
 * 
 * @param encoder  pointer to encoder instance created in same thread
 * @param user_callback_data  some pointer passed to all callbacks in this run
 * @param in_callback  pointer to implementation of callback which provides 
 *                     input images from caller
 * @param out_callback  pointer to implementation of callback which retrieves
 *                      output codestreams from encoder
 * @param buffer_callback  pointer to implementation of callback which reports
 *                         no-longer-needed input buffers or NULL if such 
 *                         callback is not needed by caller
 * @return 0 after all submitted images were encoded or nonzero error code.
 */
int
j2k_encoder_run
(
    struct j2k_encoder * encoder,
    void * user_callback_data,
    j2k_encoder_in_callback in_callback,
    j2k_encoder_out_callback out_callback,
    j2k_encoder_buffer_callback buffer_callback
);


/**
 * Can be used only in output callback to retrieve encoded image. Can be called 
 * multiple times in each callback (but always gets the same image). Encoded 
 * codestream is already written in the buffer when this function returns
 * (if the buffer is big enough.)
 * @param enc  pointer to encoder instance which has called the output callback
 * @param output_buffer_ptr  pointer to output buffer (need not be page locked)
 * @param output_buffer_size  size of output buffer in bytes
 * @param output_size_ptr  pointer to variable, where output byte count 
 *                         should be written (if successful)
 * @return 0 = OK, 1 = small buffer, negative error code = failure
 */
int
j2k_encoder_get_output
(
    struct j2k_encoder * enc,
    void * output_buffer_ptr,
    size_t output_buffer_size,
    size_t * output_size_ptr
);


/**
 * Should be used only in the input callback to provide new input image data
 * in one of supported formats to encoder. Should NOT be used in single 
 * callback again, if previous call succeded (returned 0). No encoding is done
 * if this function is not called in some callback (but callback may be 
 * called again later - depending on return value of the callback).
 * 
 * Encoder initiates memcpy into device memory directly from provided buffer 
 * and returns immediately, so the buffer should NOT be written to until 
 * j2k_encoder_buffer_callback reports the buffer as unused. Note that memcpy
 * to device memory is much faster, if page-locked memory source buffer is used 
 * (see functions j2k_encoder_buffer_pagelock, j2k_encoder_buffer_pageunlock,
 * j2k_encoder_pagelocked_alloc and j2k_encoder_pagelocked_free for working 
 * with page-locked buffers).
 * 
 * Format of input samples and size of input data is determined by parameter 
 * 'format'. Implied format parameters (sample bit depth, component count, ...)
 * must match corresponding parameters passed to encoder contructor.
 * 
 * Arbitrary data pointer can be provided by this call, which is later passed 
 * to corresponding j2k_encoder_buffer_callback and j2k_encoder_out_callback.
 * 
 * @param enc  pointer to encoded instance which called the input callback
 * @param data  pointer to buffer with data in specified format
 * @param format  format of the input data
 * @param params  parameters for encoding of this input (or NULL for defaults)
 * @param user_input_data  some pointer passed to corresponding callbacks
 * @return 0 if OK, nonzero for error (e.g. if format parameters don't match 
 *         parameters passed to encoder constructor)
 */
int
j2k_encoder_set_input
(
    struct j2k_encoder * enc,
    const void * data,
    enum j2k_input_format format,
    const struct j2k_image_params * params,
    void * user_input_data
);


/**
 * Allocates page-locked buffer (for faster encoder input loading)
 * Such buffers should be freed with j2k_encoder_pagelocked_free.
 * @param size  size of the buffer in bytes
 * @return  pointer to buffer if OK, NULL if failed
 */
void *
j2k_encoder_pagelocked_alloc(size_t size);


/**
 * Frees page locked buffer allocated with j2k_encoder_pagelocked_alloc.
 * @param buffer  pointer to buffer allocated by j2k_encoder_pagelocked_alloc
 * @return 0 if OK, nonzero for error
 */
int 
j2k_encoder_pagelocked_free(void * buffer);


/**
 * Page-locks range of memory. Memory should be later unlocked using a call 
 * to function j2k_encoder_buffer_pageunlock.
 * @param buffer  pointer to begin of the memory range to be locked|
 * @param size  size (in bytes) of the memory range to be locked
 * @return  0 if OK, nonzero for failure
 */
int
j2k_encoder_buffer_pagelock(void * buffer, size_t size);


/**
 * Unlocks page-locked memory.
 * @param buffer  pointer to begin of memory area prevously page-locked 
 *                by a call to j2k_encoder_buffer_pagelock
 * @return 0 if OK, nonzero for error
 */
int
j2k_encoder_buffer_pageunlock(void * buffer);



#ifdef __cplusplus
} /* end of extern "C" */
#endif 

#endif /* J2K_ENCODER_EXTENDED_H */
