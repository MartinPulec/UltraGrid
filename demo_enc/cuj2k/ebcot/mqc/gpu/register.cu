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

#include "register.h"
#include "common.h"

#define THREAD_CBLK_SIZE 64

/** MQ-Coder lookup table */
__constant__ struct mqc_cxstate
d_mqc_gpu_register_table[mqc_table_size];

/**
 * Perform byte out procedure
 * 
 * @param c  Code register
 * @param ct  Free space in code register
 * @param bp  Output byte buffer
 */
__device__ inline void
mqc_gpu_register_byte_out(uint32_t & c, uint8_t & ct, uint8_t* & bp)
{
    if ( *bp == 0xff ) {
        bp++;
        *bp = c >> 20;
        c &= 0xfffff;
        ct = 7;
    } else {
        if ( (c & 0x8000000) == 0 ) {
            bp++;
            *bp = c >> 19;
            c &= 0x7ffff;
            ct = 8;
        } else {
            (*bp)++;
            if ( *bp == 0xff ) {
                c &= 0x7ffffff;
                bp++;
                *bp = c >> 20;
                c &= 0xfffff;
                ct = 7;
            } else {
                bp++;
                *bp = c >> 19;
                c &= 0x7ffff;
                ct = 8;
            }
        }
    }
}

/**
 * Perform code MPS procedure
 * 
 * @param a  Interval register
 * @param c  Code register
 * @param ct  Free space in code register
 * @param bp  Output byte buffer
 * @param ctx  Reference to current context
 * @param state  Current context state
 */
__device__ inline void
mqc_gpu_register_code_mps(uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & bp, uint8_t* & ctx, struct mqc_cxstate* state)
{
    int qeval = state->qeval;
    a -= qeval;
    if ( (a & 0x8000) == 0 ) {
        if (a < qeval) {
            a = qeval;
        } else {
            c += qeval;
        }
        *ctx = state->nmps;

        while ( (a & 0x8000) == 0 ) {
            a <<= 1;
            c <<= 1;
            ct--;
            if (ct == 0) {
                mqc_gpu_register_byte_out(c,ct,bp);
            }
        }
    } else {
        c += qeval;
    }
}

/**
 * Perform code LPS procedure
 * 
 * @param a  Interval register
 * @param c  Code register
 * @param ct  Free space in code register
 * @param bp  Output byte buffer
 * @param ctx  Reference to current context
 * @param state  Current context state
 */
__device__ inline void
mqc_gpu_register_code_lps(uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & bp, uint8_t* & ctx, struct mqc_cxstate* state)
{
    uint32_t qeval =  state->qeval;
    a -= qeval;
    if ( a < qeval ) {
        c += qeval;
    } else {
        a = qeval;
    }

    *ctx = state->nlps;

    while ( (a & 0x8000) == 0 ) {
        a <<= 1;
        c <<= 1;
        ct--;
        if (ct == 0) {
            mqc_gpu_register_byte_out(c,ct,bp);
        }
    }
}

/**
 * Perform flush last bytes procedure
 * 
 * @param a  Interval register
 * @param c  Code register
 * @param ct  Free space in code register
 * @param bp  Output byte buffer
 */
__device__ inline void
mqc_gpu_register_flush(uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & bp)
{
    uint64_t tempc = c + a;
    c |= 0xffff;
    if ( c >= tempc ) {
        c -= 0x8000;
    }
    c <<= ct;
    mqc_gpu_register_byte_out(c,ct,bp);
    c <<= ct;
    mqc_gpu_register_byte_out(c,ct,bp);
    if ( *bp != 0xff ) {
        bp++;
    }
    c = c;
    ct = ct;
}

/**
 * Kernel that performs MQ-Encoding for one block
 * 
 * @param d_cblk  Array of code-blocks in device memory
 * @param cblk_count  Count of code-blocks
 * @param d_cxd  Array of input CX,D pairs in device memory
 * @param d_byte  Array of output bytes in device memory
 */
__global__ void
mqc_gpu_register_encode_kernel(struct j2k_cblk* d_cblk, int cblk_count, uint8_t* d_cxd, uint8_t* d_byte)
{
    // Get and check block index
    int block_index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if ( block_index >= cblk_count )
        return;

    // Get block of CX,D pairs
    struct j2k_cblk* block = &d_cblk[block_index];
    
    // Output byte stream
    uint8_t* start = &d_byte[block->byte_index];

    // Init variables
    uint32_t a = 0x8000;
    uint32_t c = 0;
    uint8_t ct = 12;
    uint8_t* bp = start - 1;
    if ( *bp == 0xff ) {
        ct = 13;
    }

    // Init contexts
    uint8_t ctxs[19];
    mqc_gpu_reset_cxstate(ctxs);
   
    // Code CX,D pairs
    int cxd_begin = block->cxd_index;
    int cxd_count = cxd_begin + block->cxd_count;
    int cxd_index = cxd_begin;
    while ( cxd_index < cxd_count ) {
        uint8_t cxd = d_cxd[cxd_index]; 
        uint8_t* ctx = &ctxs[mqc_gpu_cxd_get_cx(cxd)];
        struct mqc_cxstate* state = &d_mqc_gpu_register_table[*ctx];
        if ( state->mps == mqc_gpu_cxd_get_d(cxd) ) {
            mqc_gpu_register_code_mps(a,c,ct,bp,ctx,state);
        } else {
            mqc_gpu_register_code_lps(a,c,ct,bp,ctx,state);
        }
        cxd_index++;
    }

    // Flush last bytes
    mqc_gpu_register_flush(a,c,ct,bp); 

    // Set output byte count
    block->byte_count = bp - start;
}

/** Documented at declaration */
void*
mqc_gpu_register_create(struct mqc_configuration * configuration)
{
    // Init lookup table
    mqc_gpu_init_table("d_mqc_gpu_register_table");

    // Configure L1
    cudaFuncSetCacheConfig(mqc_gpu_register_encode_kernel, cudaFuncCachePreferL1);
    
    return 0;
}

/** Documented at declaration */
int
mqc_gpu_register_encode(void* mqc, struct j2k_cblk* d_cblk, int cblk_count, unsigned char * d_cxd, unsigned char * d_byte)
{
    dim3 dim_grid;
    dim_grid.x = cblk_count / THREAD_CBLK_SIZE + 1;
    if ( dim_grid.x > CUDA_MAXIMUM_GRID_SIZE ) {
        dim_grid.y = dim_grid.x / CUDA_MAXIMUM_GRID_SIZE + 1;
        dim_grid.x = CUDA_MAXIMUM_GRID_SIZE;
    }

    // Run kernel encode
    mqc_gpu_register_encode_kernel<<<dim_grid,THREAD_CBLK_SIZE>>>(d_cblk, cblk_count, d_cxd, d_byte);
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        std::cerr << "Kernel encode failed: " << cudaGetErrorString(cuerr) << std::endl;
        return -1;
    }
    return 0;
}

/** Documented at declaration */
int
mqc_gpu_register_destroy(void* mqc)
{
    return 0;
}

