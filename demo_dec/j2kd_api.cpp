///
/// @file    j2kd_api.cpp
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Implementation of API functions.
///



#include "j2kd_decoder.h"


using cuj2kd::Decoder;
using cuj2kd::u8;
using cuj2kd::Tier2;
using cuj2kd::Error;



struct j2kd_decoder {
    /// maximal status message size
    enum { MAX_MSG_LEN = 1024 * 4 };
    
    /// last status message
    char statusMessage[MAX_MSG_LEN + 1];
    
    /// copies the message to the internal buffer
    void setMsg(const char * const message) {
        strncpy(statusMessage, message, MAX_MSG_LEN);
    }
    
    /// Decoder instance
    Decoder dec;
};


/**
 * Gets basic info about given codestream.
 * @param codestream_ptr    pointer to begin of codestream
 * @param codestream_size   size of codestream in bytes
 * @param output_info_ptr   pointer to structure for output info
 * @return J2KD_OK if succeded, or some error if failed
 */
enum j2kd_status_code j2kd_get_image_info
(
    const void * const codestream_ptr,
    const size_t codestream_size,
    struct j2kd_image_info * const output_info_ptr
) {
    try {
        // check arguments
        if(0 == codestream_ptr || 0 == output_info_ptr) {
            return J2KD_ERROR_ARGUMENT_NULL;
        }
        
        // get the info
        Tier2::getImageInfo((const u8*)codestream_ptr,
                                    codestream_size, output_info_ptr);
        
        return J2KD_OK;
    } catch (Error & error) {
        return error.getStatusCode();
    } catch (...) {
        return J2KD_ERROR_UNKNOWN;
    }
}



/**
 * Gets info about specified component.
 * @param codestream_ptr    pointer to begin of codestream
 * @param codestream_size   size of codestream in bytes
 * @param component_idx     index of the component
 * @param output_info_ptr   pointer to structure for component info
 * @return J2KD_OK if succeded, or some error if failed
 */
enum j2kd_status_code j2kd_get_component_info
(
    const void * const codestream_ptr,
    const size_t codestream_size,
    const int component_idx,
    struct j2kd_component_info * const output_info_ptr
) {
    try {
        // check arguments
        if(0 == codestream_ptr || 0 == output_info_ptr) {
            return J2KD_ERROR_ARGUMENT_NULL;
        }
        if(0 > component_idx) {
            return J2KD_ERROR_ARGUMENT_OUT_OF_RANGE;
        }
        
        // get the info
        Tier2::getCompInfo((const u8*)codestream_ptr,
                                   codestream_size,
                                   component_idx, output_info_ptr);
        
        return J2KD_OK;
    } catch (Error & error) {
        return error.getStatusCode();
    } catch (...) {
        return J2KD_ERROR_UNKNOWN;
    }
}



/**
 * Creates new instance of CUDA JPEG 2000 decoder. Because this function uses 
 * current thread's CUDA context, make sure to call cudaSetDevice prior to the 
 * creation of the decoder if you want to use other GPU than the default one.
 * @param error_out_ptr     set to status code, if not null
 *                          (either 0 for OK, or nonzero for error)
 * @return  either pointer to newly created instance of CUDA JPEG 2000 decoder
 *          or 0 if initialization failed
 */
struct j2kd_decoder * j2kd_create (
    enum j2kd_status_code * const status_out_ptr
) {
    // result
    j2kd_decoder * dec = 0;
    j2kd_status_code status = J2KD_ERROR_CPU_ALLOC;
    
    // try to create the decoder
    try {
        dec = new j2kd_decoder;
        status = J2KD_OK;
        dec->setMsg("Decoder not used yet.");
    } catch (cuj2kd::Error & error) {
        status = error.getStatusCode();
    } catch (...) {
        status = J2KD_ERROR_UNKNOWN;
    }
    
    // possibly clean up if unsuccessfull 
    if(J2KD_OK != status) {
        j2kd_destroy(dec);
    }
    
    // return results
    if(status_out_ptr) {
        *status_out_ptr = status;
    }
    return dec;
}



/**
 * Destroys an instance of CUDA JPEG 2000 decoder and releases all resources 
 * associated with it.
 * @param dec  pointer to instance of the decoder wrapper to be destroyed
 *             (instance must be created in current CUDA context)
 * @return  either 0 to indiceate success or nonzero for failure (error code)
 */
enum j2kd_status_code j2kd_destroy(struct j2kd_decoder * const dec) {
    try {
        if(dec) {
            delete dec;
        }
        return J2KD_OK;
    } catch (Error & error) {
        return error.getStatusCode();
    } catch (...) {
        return J2KD_ERROR_UNKNOWN;
    }
}



/**
 * Decodes given JPEG 2000 codestream and puts decoded image into given buffer.
 * Error is reported if output buffer is too small for the image.
 * @param dec  pointer to decoder wrapper (created in current CUDA context)
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
enum j2kd_status_code j2kd_decode
(
    struct j2kd_decoder * const dec,
    const void * const codestream_ptr,
    const size_t codestream_size,
    void * output_buffer_ptr,
    const size_t output_buffer_capacity,
    const int output_buffer_on_gpu,
    const struct j2kd_component_format * const comp_format_ptr,
    const int comp_format_count
) {
    // check arguments
    if(!dec || !codestream_ptr || !output_buffer_ptr || !comp_format_ptr) {
        return J2KD_ERROR_ARGUMENT_NULL;
    }
    if(comp_format_count < 0) {
        return J2KD_ERROR_ARGUMENT_OUT_OF_RANGE;
    }
    
    // try to run the decoding
    try {
        
        dec->dec.decode(
            (const u8*)codestream_ptr,
            codestream_size,
            output_buffer_ptr,
            output_buffer_capacity,
            output_buffer_on_gpu,
            comp_format_ptr,
            comp_format_count
        );
        
        // no exception => signalize success
        dec->setMsg("OK");
        return J2KD_OK;
    } catch (Error & error) {
        dec->setMsg(error.getMessage());
        return error.getStatusCode();
    } catch (...) {
        dec->setMsg("UNKNOWN ERROR");
        return J2KD_ERROR_UNKNOWN;
    }
}




enum j2kd_status_code j2kd_run
(
    struct j2kd_decoder * const dec,
    j2kd_input_begin_callback in_begin_callback,
    j2kd_input_end_callback in_end_callback,
    j2kd_output_callback out_callback,
    j2kd_postprocessing_callback postproc_callback,
    j2kd_decoding_end_callback dec_end_callback,
    void * const custom_callback_ptr
) {
    // check arguments
    if(!dec) {
        return J2KD_ERROR_ARGUMENT_NULL;
    }
    
    // try to run the decoding
    try {
        dec->dec.run(
            in_begin_callback,
            in_end_callback,
            out_callback,
            postproc_callback,
            dec_end_callback,
            custom_callback_ptr
        );
        
        // no exception => signalize success
        dec->setMsg("OK");
        return J2KD_OK;
    } catch (Error & error) {
        dec->setMsg(error.getMessage());
        return error.getStatusCode();
    } catch (...) {
        dec->setMsg("UNKNOWN ERROR");
        return J2KD_ERROR_UNKNOWN;
    }
}



/**
 * Gets corresponding "English" status message describing last decoder's error.
 * (Returned pointer is valid as long as correponding decoded instance exists).
 * @param dec  pointer to some instance of JPEG 2000 decoder
 * @return textual description of last decoder's call status
 */
const char * j2kd_status(struct j2kd_decoder * const dec) {
    // cast to right pointer type
    return dec ? dec->statusMessage
               : "NULL argument of j2kd_decoder_status.";
}



