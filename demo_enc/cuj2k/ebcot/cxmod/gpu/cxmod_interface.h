/**
 * @file    cxmod_interface.h
 * @author  Martin Jirman (martin.jirman@cesnet.cz)
 * @date    2011-06-08 14:35
 * @brief   Interface of self-contained GPU context modeller.
 * 
 * Limitations of the implementation (compared to standard):
 *     - Valid codeblock dimensions are powers of 2, greater or equal to 4. 
 *       Moreover, the product of width and height of the codeblock must be 
 *       less or equal to 4096. However, some weird (but valid) codeblock 
 *       dimensions are not supported (e.g. 1024x4).
 *     - Maximal number of bitplanes can be in range 0 to 16 (both inclusive).
 *       This does not include sign bit (includes magnitude only).
 *     - Does not support vertical causal mode.
 */


#ifndef CXMOD_INTERFACE_H
#define CXMOD_INTERFACE_H

#include "../../../j2k.h"

#ifdef __cplusplus
extern "C" {
#endif




/**
 * Creates a new instance of context modeller for given parameteers.
 * @param p initialized parameters of J2K encoder
 * @return null = error, non-null = new valid instance of context modeller
 */
void * cxmod_create(const struct j2k_encoder_params * const p);



/**
 * Releases all resources associated with given context modeller instance.
 * @param cxmod_ptr  pointer to internal stuff of context modeller in main 
 *                   system memory returned by 'cxmod_create'
 * @return zero if destroyed OK, nonzero otherwise
 */
int cxmod_destroy(void * const cxmod_ptr);



/**
 * Uses given context modeller structure to encode codeblocks in specified 
 * buffer, to save output CX,D pairs to the other specified buffer 
 * and to write info about output codeblocks and passes with specified 
 * formatting.
 * @param cxmod_ptr  main system memory pointer to context modeller 
 *        internal data (returned by 'cxmod_create')
 * @param cblk_count  number of codeblocks to be encoded
 * @param cblks_gpu_ptr  GPU memory pointer to array with info about codeblocks
 * @param bands_gpu_ptr  GPU memory pointer to array with info about bands 
 * @param in_pixels_gpu_ptr  GPU memory pointer to all input pixels
 * @param out_cxd_gpu_ptr  GPU memory pointer to buffer for output CX,D pairs
 *        NOTE: must be aligned to 16byte boundary!
 * @param cuda_stream_ptr  pointer to variable of type cudaStream_t in main 
 *        system memory, specifying CUDA stream, in which the context modeller
 *        kernel should be launched (can be NULL for default stream)
 * @return zero if successfully launched, nonzero otherwise
 */
int cxmod_encode(
    void * const cxmod_ptr,
    const int cblk_count,
    struct j2k_cblk * const cblks_gpu_ptr,
    const struct j2k_band * const bands_gpu_ptr,
    const int * const in_pixels_gpu_ptr,
    unsigned char * const out_cxd_gpu_ptr,
    const void * const cuda_stream_ptr
);


    
#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* CXMOD_INTERFACE_H */
