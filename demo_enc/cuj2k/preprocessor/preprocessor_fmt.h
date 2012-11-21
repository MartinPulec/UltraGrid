/* 
 * Copyright (c) 2009, Martin Jirman
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

#ifndef J2K_PREPROCESSOR_FMT_H
#define J2K_PREPROCESSOR_FMT_H

#include "../j2k.h"

#ifdef __cplusplus
extern "C" {
#endif



/** Forward declaration of preprocessor instance type. */
struct j2k_fmt_preprocessor;


/**
 * Initialize a new instance of fixed-format-data preprocessor.
 * @return either a new instance of fixed-format-preprocessor or null if failed
 */
struct j2k_fmt_preprocessor *
j2k_fmt_preprocessor_create();


/**
 * Releases all resources associated with some instance of preprocessor.
 * @param preprocessor  pointer to preprocessor instance
 */
void 
j2k_fmt_preprocessor_destroy(struct j2k_fmt_preprocessor * preprocessor);


/** 
 * Preprocess image data saved in GPU buffer in one of supported formats.
 * @param preprocessor  pointer to instance of the preprocessor
 * @param format  input data format
 * @param size  input data array size
 * @param comp_count  expected number of color components
 * @param bit_depth  expected pixel-component bit depth
 * @param is_signed  nonzero if pixel values should be signed
 * @param mode  output floats or ints
 * @param in_gpu_ptr  pointer to input GPU data
 * @param out_gpu_ptrs  pointer to output GPU pointers for each component
 * @param is_log  nonzero if input is in logarithmic color space (Cineon Log)
 * @param mct  nonzero for MCT to be used
 * @param stream  stream to run in
 * @return 0 if OK, nonzero otherwise
 */
int
j2k_fmt_preprocessor_preprocess(
    struct j2k_fmt_preprocessor * const instance,
    enum j2k_input_format format,
    struct j2k_size size,
    int comp_count,
    int bit_depth,
    int is_signed,
    enum j2k_compression_mode mode,
    const void * in_gpu_ptr,
    void ** out_gpu_ptrs,
    int is_log,
    int mct,
    cudaStream_t stream
);


#ifdef __cplusplus
}
#endif

#endif // J2K_PREPROCESSOR_FMT_H
