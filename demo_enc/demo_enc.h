/**
 * @file    demo_enc.h
 * @author  Martin Jirman (jirman@cesnet.cz)
 * @brief   Multiple GPU encoder wrapper for 2012 demo.
 */

#ifndef DEMO_ENC_H
#define DEMO_ENC_H

#ifdef __cplusplus
extern "C" {
#endif


/** Multiple-GPU JPEG 2000 encoder instance type for 2012 demo. */
struct demo_enc;


/**
 * Creates and initializes new instance of JPEG 2000 encoder.
 * Output codestreams are 4K DCI compatible (24fps 2K for subsampled frames).
 * @param gpu_indices_ptr   pointer to array of indices of GPUs to be used 
 *                          for encoding or null to use all available GPUs
 * @param gpu_indices_count count of GPU indices (unused if pointer is null)
 * @param size_x            image width in pixels
 * @param size_y            image height in pixels
 * @param dwt_level_count   number of DWT decomposition levels (5 is OK for 4K)
 * @param max_quality       maximal quality (0.0f to 1.2f, limits buffers size)
 * @return either pointer to new instance of encoder or null if error occured
 */
struct demo_enc * 
demo_enc_create
(
    const int * gpu_indices_ptr,
    int gpu_indices_count,
    int size_x,
    int size_y,
    int dwt_level_count,
    float quality_upper_bound
);


/**
 * Releases all resources of the encoder instance. 
 * Effects are undefined if any thread waits for output when this is called.
 * @param enc_ptr pointer to encoder instance
 */
void
demo_enc_destroy
(
    struct demo_enc * enc_ptr
);


/**
 * Submits frame for encoding.
 * @param enc_ptr          pointer to encoder instance
 * @param custom_data_ptr  custom pointer associated with frame
 * @param out_buffer_ptr   pointer to ouptut buffer
 * @param out_buffer_size  ouptut buffer capacity (in bytes)
 * @param src_ptr          pointer to source RGB data: 10 bits per sample,
 *                         each pixel packed to MSBs of 32bit little endian 
 *                         integer (with 2 LSBs unused), without any padding
 * @param required_size    required output size of the encoded frame (in bytes)
 *                         or 0 for unlimited size (NOTE: actual output may be 
 *                         smaller or even slightly bigger)
 * @param quality          encoded frame quality:
 *                             0.1f = poor
 *                             0.7f = good
 *                             1.2f = perfect
 *                         (also bound by encoder-creation-time quality limit)
 * @param subsampling      0 for full resolution frame (same as input)
 *                         1 for half width and height, 2 for quarter, ...
 *                         (up to dwt level count given to constructor)
 * @param logo_text        overlay text pointer or null not to overlay at all
 */
void
demo_enc_submit
(
    struct demo_enc * enc_ptr,
    void * custom_data_ptr,
    void * out_buffer_ptr,
    int out_buffer_size,
    void * src_ptr,
    int required_size,
    float quality,
    int subsampling,
    const char * logo_text
);


/**
 * Unblocks all waiting threads and stops encoding.
 * (Indicated by return value of demo_enc_wait.)
 * @param enc_ptr  pointer to encoder instance
 */
void
demo_enc_stop
(
    struct demo_enc * enc_ptr
);


/**
 * Waits for next encoded image of for encoder deallocation.
 * @param enc_ptr             pointer to encoder instance
 * @param custom_data_ptr_out null or pointer to pointer, where custom data 
 *                            pointer associated with the frame is written
 * @param out_buffer_ptr_out  null or pointer to pointer, where provided 
 *                            output buffer pointer is written
 * @param src_ptr_out         null or pointer to pointer, where provided 
 *                            input data pointer is written
 * @return positive output size (in bytes) if frame encoded correctly,
 *         0 if encoder was stopped while waiting (outputs are undefined),
 *         -1 if error occured when encoding the frame
 */
int
demo_enc_wait
(
    struct demo_enc * enc_ptr,
    void ** custom_data_ptr_out,
    void ** out_buffer_ptr_out,
    const void ** src_ptr_out
);



#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* DEMO_ENC_H */
