/* 
 * Copyright (c) 2011, Martin Srom
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

#ifndef MQC_H
#define MQC_H

#include "../../j2k.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Configuration of MQ-Coder
 */
struct mqc_configuration
{
    // String parameter, content depends on concrete version of MQ-Coder
    const char * configuration;
};

/**
 * Initialize MQ-Coder
 *
 * @param configuration Configuration of MQ-Coder
 * @return instance of MQ-Coder or 0 if fails
 */
void*
mqc_create(const struct j2k_encoder_params * const parameters);

/**
 * Calculate byte count for allocating output buffer 
 * before calling mqc_encode
 *
 * @param cxd_count   Count of input CX,D pairs
 */
#define mqc_calculate_byte_count(cxd_count)   (16 + cxd_count / 8)

/**
 * Encode by MQ-Coder
 *
 * @param mqc  Instance of MQ-Coder
 * @param cblk_count  Count of code-blocs in device memory
 * @param d_cblk  Array of code-blocks in device memory
 * @param d_cxd  Array of input CX,D pairs in device memory
 * @param d_byte  Array of output bytes in device memory
 * @param d_trunc_sizes  Array of byte counts for truncation points
 * @param stream  CUDA stream for kernels
 * @return 0 if OK, nonzero otherwise
 */
int
mqc_encode(
    void* mqc,
    int cblk_count,
    struct j2k_cblk * d_cblk,
    unsigned char * d_cxd, 
    unsigned char * d_byte,
    unsigned int * d_trunc_sizes,
    cudaStream_t stream
);

/**
 * Deinitialize MQ-Coder
 *
 * @return 0 if OK, nonzero otherwise
 */ 
int
mqc_destroy(void* mqc);

#ifdef __cplusplus
}
#endif

#endif // MQC_H
