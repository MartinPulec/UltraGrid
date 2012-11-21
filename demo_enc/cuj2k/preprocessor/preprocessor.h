/* 
 * Copyright (c) 2009, Jiri Matela
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

#ifndef J2K_PREPROCESSOR_H
#define J2K_PREPROCESSOR_H

#include "../j2k.h"
#include "preprocessor_ct_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Preprocessor function */
typedef void (*preprocessor_function_t)(void*, void*, void*, const unsigned char*, int, int, cudaStream_t);

/**
 * Get preprocessor function based on parameters
 *
 * @param bit_depth  Pixel bit depth
 * @param is_signed  Flag if pixel value is with sign (1 - true | 0 - false), if is -> normalization will be done
 * @param comp_count  Color components count
 * @param compression_mode either CM_LOSSLESS or CM_LOSSY_FLOAT
 * @param transform  Color tranformation (CT_NONE | CT_REVERSIBLE | CT_IRREVERSIBLE)
 * @return preprocessor function
 */
preprocessor_function_t
preprocessor_get_function(int bit_depth, int is_signed, int comp_count, enum j2k_compression_mode compression_mode, enum j2k_component_transform transform);

/** 
 * Preprocess image data using function stored in structure
 *
 * @param preprocessor  Preprocessor structure
 * @param d_comp_data  Array of pointers to color component buffers
 * @return 0 if OK, nonzero otherwise
 */
int
preprocessor_process(preprocessor_function_t preprocessor_function, void** d_comp_data, int comp_count,
                     unsigned char* d_source, struct j2k_size size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // J2K_PREPROCESSOR_H
