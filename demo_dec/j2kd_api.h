/**
 * @file    j2kd_api.h
 * @author  Martin Jirman (martin.jirman@cesnet.cz)
 * @brief   Interface of the CUDA JPEG 2000 decoder.
 */

#ifndef CUJ2K_DEC_H
#define CUJ2K_DEC_H

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif


/** Forward declaration of type of CUDA JPEG 2000 decoder instance. */
struct j2kd_decoder;


/** Information about the decoded image. */
struct j2kd_image_info
{
    /** position of top left pixel */ 
    int image_begin_x;
    int image_begin_y;
    
    /** position of bottom-right pixel + (1,1) */
    int image_end_x;
    int image_end_y;
    
    /** dimensions of tiles */
    int tile_size_x;
    int tile_size_y;
    
    /** top left pixel position of top-left tile */
    int tile_origin_x;
    int tile_origin_y;
    
    /** codestream type/capabilities ID (as in SIZ marker) */
    int capabilities;
    
    /** number of components */
    int comp_count;
};


/** Information about one encoded component. */
struct j2kd_component_info
{
    /** bit depth of encoded samples (possibly including the sign bit) */
    int bit_depth;
    
    /** 1 if encoded samples were signed, 0 otherwise */
    int is_signed;
    
    /** 0 based index of the component */
    int index;
};


/** Data type for standard output format. */
enum j2kd_data_type {
    J2KD_TYPE_INT8,
    J2KD_TYPE_INT16,
    J2KD_TYPE_INT32
};


/** Specification of output component. */
struct j2kd_component_format {
    /** Zero based index of formatted color component. */
    int component_idx;
    
    /** Data type for the component storage */
    enum j2kd_data_type type;
    
    /** Offset of top-left pixel in the buffer (in units of size of the type) */
    int offset;
    
    /**
     * Difference between output indices of two horizontally neighboring pixels
     * (in units of size of the selected data type - e.g. 4 for 32bit int)
     * Can be negative (e.g. for horizontally mirrored images).
     */
    int stride_x;
    
    /**
     * Difference between output indices of two horizontally neighboring pixels
     * (in units of size of the selected data type - e.g. 4 for 32bit int)
     * Can be negative (e.g. for vertically mirrored images).
     */
    int stride_y;
    
    /** Required output bit depth (independent of selected type). */
    int bit_depth;
    
    /** 
     * Nonzero if samples should be signed (0 = gray),
     * or 0 for unsigned samples (0 = black).
     */
    int is_signed;
    
    /**
     * Size of final left bit shift (0 for no left shift).
     * (withing selected type, before saving the value to buffer).
     */
    int final_shl;
    
    /**
     * Nonzero, if values should be OR-combined with previously written value 
     * at the same place in output buffer or 0 to overwrite.
     */
    int combine_or;
};


/** JPEG 2000 decoder interface status codes. */
enum j2kd_status_code
{
    J2KD_OK = 0,
    J2KD_ERROR_BAD_CODESTREAM,
    J2KD_ERROR_CPU_ALLOC,
    J2KD_ERROR_CUDA,
    J2KD_ERROR_SMALL_BUFFER,
    J2KD_ERROR_UNSUPPORTED,
    J2KD_ERROR_ARGUMENT_OUT_OF_RANGE,
    J2KD_ERROR_ARGUMENT_NULL,
    J2KD_ERROR_UNKNOWN
    
    /* TODO: add more error codes */
};


/**
 * Gets basic info about given codestream.
 * @param codestream_ptr    pointer to begin of codestream
 * @param codestream_size   size of codestream in bytes
 * @param output_info_ptr   pointer to structure for output info
 * @return J2KD_OK if succeded, or some error if failed
 */
enum j2kd_status_code 
j2kd_get_image_info
(
    const void * const codestream_ptr,
    const size_t codestream_size,
    struct j2kd_image_info * const output_info_ptr
);


/**
 * Gets info about specified component.
 * @param codestream_ptr    pointer to begin of codestream
 * @param codestream_size   size of codestream in bytes
 * @param component_idx     index of the component
 * @param output_info_ptr   pointer to structure for component info
 * @return J2KD_OK if succeded, or some error if failed
 */
enum j2kd_status_code 
j2kd_get_component_info
(
    const void * const codestream_ptr,
    const size_t codestream_size,
    const int component_idx,
    struct j2kd_component_info * const output_info_ptr
);


/**
 * Creates new instance of CUDA JPEG 2000 decoder. Because this function uses 
 * current thread's CUDA context, make sure to call cudaSetDevice prior to the 
 * creation of the decoder if you want to use other GPU than the default one.
 * @param error_out_ptr     set to status code, if not null
 * @return  either pointer to newly created instance of CUDA JPEG 2000 decoder
 *          or 0 if initialization failed
 */
struct j2kd_decoder *
j2kd_create
(
    enum j2kd_status_code * const status_out_ptr
);


/**
 * Destroys an instance of CUDA JPEG 2000 decoder and releases all resources 
 * associated with it.
 * @param dec_ptr  pointer to instance of the decoder to be destroyed
 *                 (instance must be created in current CUDA context)
 * @return  either 0 to indiceate success or nonzero for failure (error code)
 */
enum j2kd_status_code
j2kd_destroy
(
    struct j2kd_decoder * const dec_ptr
);


// /**
//  * Should be called only within j2kd_decoder_input_callback and only once 
//  * in each callback call. Both input and output buffer should be located 
//  * in page locked host memory for optimal performance.
//  * @param dec_ptr  decoder instance pointer (which called the input callback)
//  * @param custom_image_ptr  custom pointer (only passes unchanged to 
//  *                          corresponding output callback)
//  * @param input_ptr  input buffer pointer (in host memory)
//  * @param output_ptr  outpu tbuffer pointer (either in host or device memory)
//  * @param output_capacity  capacity of output buffer
//  * @param input_size  input codestream byte count
//  * @param output_in_device_mem  nonzero if output pointer points to 
//  *                              device memory, 0 for host memory
//  * @param comp_format_ptr  pointer to array of output component formatting 
//  *                         info structures in host memory
//  * @param comp_format_count  count of formatted components
//  * @return status code (NOTE: even if success is reported, some problems 
//  *         may occur later in decoding proces - those problems are reported 
//  *         in corresponding output callback)
//  */
// enum j2kd_status_code
// j2kd_decoder_set_input
// (
//     struct j2kd_decoder * dec_ptr,
//     void * custom_image_ptr,
//     const void * const input_ptr,
//     const struct j2kd_component_format * const comp_format_ptr,
//     const int comp_format_count
// );
// 
// 
// 
// enum j2kd_status_code
// j2kd_decoder_get_output
// (
//     struct j2kd_decoder * dec_ptr,
//     void * const output_ptr,
//     const int output_in_device_mem,
//     const size_t output_capacity,
//     const size_t input_size
// );
// 
// 
// /**
//  * Called when decoder wants another j2k codestream to be decoded. Callback 
//  * implementation should call j2kd_decoder_set_input to provide next codestream 
//  * to be decoded. Provided input and output buffers should NOT be touched until
//  * corresponding output callback returns the buffers to the caller.
//  * @param dec_ptr  decoder instance pointer
//  * @param custom_callback_ptr  custom pointer passed to j2kd_decoder_run
//  * @return nonzero if there will be more codestreams to be decoded 
//  *         or 0 if the callback should not be called again
//  */
// typedef int
// (*j2kd_decoder_input_callback)
// (
//     struct j2kd_decoder * dec_ptr,
//     void * custom_callback_ptr
// );
// 
// 
// typedef void
// (*j2kd_decoder_buffer_callback)
// (
//     struct j2kd_decoder * dec_ptr,
//     void * custom_callback_ptr,
//     void * custom_image_ptr,
//     const void * const input_ptr
// );
// 
// 
// /**
//  * Called when image decoding ends. Not input nor output buffer is not needed 
//  * by decoder when this callback is called. Pointers 'custom_image_ptr',
//  * 'input_ptr' and 'output_ptr' were passed together to the decoder using the 
//  * same call to j2kd_decoder_set_input. This callback is called for all 
//  * inputs 
//  * @param status  image decoding status
//  * @param dec_ptr  pointer to corresponding decoder instance
//  * @param custom_callback_ptr  custom pointer passed to j2kd_decoder_run
//  * @param custom_image_ptr  custom image-specific pointer
//  * @param input_ptr  input buffer pointer
//  * @param output_ptr  output buffer pointer
//  */
// typedef void
// (*j2kd_decoder_output_callback)
// (
//     enum j2kd_status_code status,
//     struct j2kd_decoder * dec_ptr,
//     void * custom_callback_ptr,
//     void * custom_image_ptr,
//     const void * input_ptr
// );
// 
// 
// /**
//  * Runs the encoder which then calls given callbacks to report decoded images 
//  * and get new images for decoding. 
//  * @param dec_ptr  pointer to decoder instance
//  * @param in_callback  custom input callback implementation pointer
//  * @param out_callback  custom output callback implementation pointer 
//  * @param custom_callback_ptr  custom value passed to callbacks
//  * @return  last error code or OK if all images were decoded correctly
//  */
// enum j2kd_status_code
// j2kd_decoder_run
// (
//     struct j2kd_decoder * const dec_ptr,
//     j2kd_decoder_input_callback in_callback,
//     j2kd_decoder_output_callback out_callback,
//     j2kd_decoder_buffer_callback buffer_calback,
//     void * const custom_callback_ptr
// );


/**
 * Decodes given JPEG 2000 codestream and puts decoded image into given buffer.
 * Error is reported if output buffer is too small for the image.
 * @param dec_ptr  pointer to decode instance (created in current CUDA context)
 * @param codestream_ptr  pointer to begin of codestream in main system memory
 * @param codestream_size  size of codestream in bytes
 * @param output_buffer_ptr  pointer to output buffer or null if no output 
 *                           should be saved (e.g to save the info only)
 * @param output_buffer_capacity  size of output buffer in bytes
 * @param output_buffer_on_gpu  nonzero if output_buffer_ptr points to GPU 
 *                              memory, 0 if it points to main system memory
 * @param comp_format_ptr  pointer to array with info about required output 
 *                         format of some components - each structure for one
 *                         component
 * @param comp_format_count  number of components to be saved (corresponds 
 *                           to number of items in comp_format_ptr)
 * @return status code - either 0 for success or nonzero for some error
 */
enum j2kd_status_code
j2kd_decode
(
    struct j2kd_decoder * const dec_ptr,
    const void * const codestream_ptr,
    const size_t codestream_size,
    void * output_buffer_ptr,
    const size_t output_buffer_capacity,
    const int output_buffer_on_gpu,
    const struct j2kd_component_format * const comp_format_ptr,
    const int comp_format_count
);


/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/



typedef int j2kd_input_begin_callback
(
    void * custom_callback_ptr,
    void ** custom_image_ptr_out,
    const void ** codestream_ptr_out,
    size_t * codestream_size_out,
    const struct j2kd_component_format ** comp_format_ptr_out,
    int * comp_format_count_out,
    int should_block
);


typedef void j2kd_input_end_callback
(
    void * custom_callback_ptr,
    void * custom_image_ptr,
    const void * codestream_ptr
);


typedef void j2kd_output_callback
(
    void * custom_callback_ptr,
    void * custom_image_ptr,
    void ** output_ptr_out,
    size_t * output_capacity_out,
    int * output_in_device_mem_out
);


/**
 * TODO: add description
 * 
 * 
 * @return output data size
 */
typedef size_t j2kd_postprocessing_callback
(
    void * custom_callback_ptr,
    void * custom_image_ptr,
    void * src,
    void * dest,
    const void * cuda_stream_id_ptr
);


typedef void j2kd_decoding_end_callback
(
    void * custom_callback_ptr,
    void * custom_image_ptr,
    enum j2kd_status_code status,
    const struct j2kd_component_format * comp_format_ptr,
    void * output_ptr
);


enum j2kd_status_code j2kd_run
(
    struct j2kd_decoder * const dec_ptr,
    j2kd_input_begin_callback in_begin_callback,
    j2kd_input_end_callback in_end_callback,
    j2kd_output_callback out_callback,
    j2kd_postprocessing_callback postproc_callback,
    j2kd_decoding_end_callback dec_end_callback,
    void * const custom_callback_ptr
);







/**
 * Gets corresponding English status message describing last decoder's error.
 * (Returned pointer is valid as long as correponding decoded instance exists).
 * @param decoder  pointer to some instance of JPEG 2000 decoder
 * @return textual description of last decoder's call status
 */
const char * j2kd_status(struct j2kd_decoder * const decoder);


#ifdef __cplusplus
}  /* end of extern "C" */
#endif

#endif /* CUJ2K_DEC_H */
