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

#ifndef MQC_GPU_COMMON_H
#define MQC_GPU_COMMON_H

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
#include <iostream>
#endif
#include "../../common.h"

/**
 * CUDA constants
 */
#define CUDA_MAXIMUM_GRID_SIZE 65535

/**
 * CX,D in byte manipulation ("0ccccc0d")
 */
#define mqc_gpu_cxd_is_pass_end(cxd)            (cxd == 0xFF)
#define mqc_gpu_cxd_make(cx,d)                  ((unsigned char)((cx << 2) | 0x2 | d))
#define mqc_gpu_cxd_get_cx(cxd)                 ((unsigned char)(cxd >> 2))
#define mqc_gpu_cxd_get_d(cxd)                  ((unsigned char)(cxd & 0x1))
#define mqc_gpu_maximum_byte_count(cxd_count)   (16 + cxd_count / 8)

/**
 * Lookup table in CPU memory
 */
#include "../mqc_table.h"

/**
 * Init GPU lookup table for MQ-Coder
 * 
 * @param table_variable  MQ-Coder lookup table in device constant memory
 */
void
mqc_gpu_init_table(const char* table_variable);

/**
 * Number of contexts
 */
#define mqc_cx_count (19)

/**
 * Reset context states
 *
 * @param cxstate  Array of indexes to lookup table for each context (defines state for each context)
 */
#ifdef __CUDACC__
__device__ inline void
mqc_gpu_reset_cxstate(uint8_t* cxstate)
{
    cxstate[0]  = (4 << 1);
    cxstate[1]  = 0;
    cxstate[2]  = 0;
    cxstate[3]  = 0;
    cxstate[4]  = 0;
    cxstate[5]  = 0;
    cxstate[6]  = 0;
    cxstate[7]  = 0;
    cxstate[8]  = 0;
    cxstate[9]  = 0;
    cxstate[10] = 0;
    cxstate[11] = 0;
    cxstate[12] = 0;
    cxstate[13] = 0;
    cxstate[14] = 0;
    cxstate[15] = 0;
    cxstate[16] = 0;
    cxstate[17] = (3 << 1);
    cxstate[18] = (46 << 1);
}
#endif

#endif // MQC_GPU_COMMON_H
