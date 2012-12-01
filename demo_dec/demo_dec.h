/**
 * @file    demo_dec.h
 * @author  Martin Jirman (jirman@cesnet.cz)
 * @brief   Multiple GPU decoder wrapper for 2012 demo.
 */

#ifndef DEMO_DEC_H
#define DEMO_DEC_H

#ifdef __cplusplus
extern "C" {
#endif



/** Multiple-GPU JPEG 2000 decoder instance type for 2012 demo. */
struct demo_dec;



/**
 * Creates and initializes new instance of JPEG 2000 decoder.
 * @param gpu_indices_ptr   pointer to array of indices of GPUs to be used 
 *                          for decoding or null to use all available GPUs
 * @param gpu_indices_count count of GPU indices (unused if pointer is null)
 * @return either pointer to new instance of decoder or null if error occured
 */
struct demo_dec * 
demo_dec_create
(
    const int * gpu_indices_ptr,
    int gpu_indices_count
);



/**
 * Releases all resources of the instance. 
 * Effects are undefined if any thread waits for output when this is called.
 * @param dec_ptr pointer to decoder instance
 */
void
demo_dec_destroy
(
    struct demo_dec * dec_ptr
);



/**
 * Submits frame for decoding.
 * @param dec_ptr         pointer to decoder instance
 * @param custom_data_ptr custom pointer associated with frame
 * @param out_buffer_ptr  pointer to ouptut buffer with sufficient capacity
 * @param codestream_ptr  pointer to JPEG 2000 codestream
 * @param codestream_size size of given codestream (in bytes)
 * @param double_sized    nonzero for output size to be double sized
 */
void
demo_dec_submit
(
    struct demo_dec * dec_ptr,
    void * custom_data_ptr,
    void * out_buffer_ptr,
    const void * codestream_ptr,
    int codestream_size,
    int double_sized
);


/**
 * Unblocks all waiting threads and stops decoding.
 * (Indicated by return value of demo_dec_wait.)
 * @param dec_ptr  pointer to decoder instance
 */
void
demo_dec_stop
(
    struct demo_dec * dec_ptr
);


/**
 * Waits for next decoded image of for decoder deallocation.
 * @param dec_ptr             pointer to decoder instance
 * @param custom_data_ptr_out null or pointer to pointer, where custom data 
 *                            pointer associated with the frame is written
 * @param out_buffer_ptr_out  null or pointer to pointer, where provided 
 *                            output buffer pointer is written
 * @param codestream_ptr_out  null or pointer to pointer, where provided 
 *                            input codestream pointer is written
 * @return 0 if frame decoded correctly,
 *         1 if decoder was stopped while waiting (outputs are undefined),
 *         2 if error occured when decoding the frame
 */
int
demo_dec_wait
(
    struct demo_dec * dec_ptr,
    void ** custom_data_ptr_out,
    void ** out_buffer_ptr_out,
    const void ** codestream_ptr_out
);



/**
 * Gets count of GPU threads of the decoder.
 * @return GPU decoding thread count of the decoder instance
 */
int
demo_dec_gpu_count
(
    struct demo_dec * dec_ptr
);



/**
 * Gets basic info about given codestream.
 * @param codestream_ptr   pointer to codestream
 * @param codestream_size  codestream size in bytes
 * @param comp_count_out   pointer to int where color component count is 
 *                         written or null
 * @param size_x_out       null or pointer to int where image width is written
 * @param size_y_out       null or pointer to int where image height is written
 * @return 0 if input is definitely NOT valid JPEG 2000 codestream 
 *         (outputs are undefined), nonzero if it may be valid
 */
int
demo_dec_image_info
(
    const void * codestream_ptr,
    int codestream_size,
    int * comp_count_out,
    int * size_x_out,
    int * size_y_out
);



/**
 * Gets size of v210 encoded image.
 * @param size_x  image width
 * @param size_y  image height
 * @return byte size of v210 encoded image (including all sorts of padding)
 */
int
demo_dec_v210_size
(
    int size_x,
    int size_y
);


#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* DEMO_DEC_H */
