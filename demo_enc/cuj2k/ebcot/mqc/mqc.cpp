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

#include "mqc.h"
#include "gpu/final.h"
#include <stdio.h>

/** Documented at declaration */
void*
mqc_create(const struct j2k_encoder_params * const /*parameters*/)
{
    struct mqc_configuration configuration;
    return mqc_gpu_final_create(&configuration);
}

/** Documented at declaration */
int
mqc_encode(
    void* mqc,
    int cblk_count,
    struct j2k_cblk * d_cblk,
    unsigned char * d_cxd, 
    unsigned char * d_byte,
    unsigned int * d_trunc_sizes,
    cudaStream_t stream
)
{
    return mqc_gpu_final_encode(
        mqc,
        d_cblk,
        cblk_count,
        d_cxd,
        d_byte,
        d_trunc_sizes,
        stream
    );
}

/** Documented at declaration */
int
mqc_destroy(void* mqc)
{
    return mqc_gpu_final_destroy(mqc);
}

