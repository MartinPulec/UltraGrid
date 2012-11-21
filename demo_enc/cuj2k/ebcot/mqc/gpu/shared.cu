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

#include "shared.h"
#include "common.h"

#define DEFAULT_TWC 64
#define DEFAULT_TPC 2
#include "configuration.h"

/** MQ-Coder lookup table */
__constant__ struct mqc_cxstate
d_mqc_gpu_shared_table[mqc_table_size];

/**
 * Perform byte out procedure
 * 
 * @param c  Code register
 * @param ct  Free space in code register
 * @param bp  Output byte buffer
 */
__device__ inline void
mqc_gpu_shared_byte_out(uint32_t & c, uint8_t & ct, uint8_t* & bp)
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
mqc_gpu_shared_code_mps(uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & bp, uint8_t * & ctx, struct mqc_cxstate* state)
{
    int qeval =  state->qeval;
    a -= qeval;
    if ( (a & 0x8000) == 0 ) {
        if (a < qeval) {
            a = qeval;
        } else {
            c += qeval;
        }
        *ctx = state->nmps;

        a <<= 1;
        c <<= 1;
        ct--;
        if (ct == 0) {
            mqc_gpu_shared_byte_out(c,ct,bp);
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
mqc_gpu_shared_code_lps(uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & bp, uint8_t * & ctx, struct mqc_cxstate* state)
{
    int qeval =  state->qeval;
    a -= qeval;
    if ( a < qeval ) {
        c += qeval;

        *ctx = state->nlps;

        int ns = __clz(a) - (sizeof(uint32_t) * 8 - 16);

        a <<= ns;
        while ( ct <= ns ) {
            ns -= ct;
            c <<= ct;
            mqc_gpu_shared_byte_out(c,ct,bp);
        }
        c <<= ns;
        ct -= ns;
    } else {
        a = qeval;

        int ns = state->ns;

        *ctx = state->nlps;

        a <<= ns;
        while ( ct <= ns ) {
            ns -= ct;
            c <<= ct;
            mqc_gpu_shared_byte_out(c,ct,bp);
        }
        c <<= ns;
        ct -= ns;
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
mqc_gpu_shared_flush(uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & bp)
{
    unsigned int tempc = c + a;
    c |= 0xffff;
    if ( c >= tempc ) {
        c -= 0x8000;
    }
    c <<= ct;
    mqc_gpu_shared_byte_out(c,ct,bp);
    c <<= ct;
    mqc_gpu_shared_byte_out(c,ct,bp);
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
template <
    unsigned int threadWorkCount, 
    unsigned int threadPerCount, 
    class cxdLoadType, 
    unsigned int cxdLoadCount, 
    calculate_t calculate
>
__global__ void
kernel_mqc_gpu_shared_encode(struct j2k_cblk* d_cblk, int cblk_count, unsigned char* d_cxd, unsigned char* d_byte)
{
    // Get and check block index
    int block_index = (blockIdx.y * gridDim.x + blockIdx.x) * threadWorkCount + threadIdx.x / threadPerCount;
    if ( block_index >= cblk_count )
        return;

    // Thread index in count
    int thread_index = threadIdx.x % threadPerCount;
    // Nearest working tread index (skip not working indexes)
    int thread_work_index = (threadIdx.x / threadPerCount);

    // Is this thread working
    bool work_thread = (thread_index) == 0;

    // Get block of CX,D pairs
    struct j2k_cblk* block = &d_cblk[block_index];

    // CX,D info
    int cxd_begin = block->cxd_index;
    int cxd_count = cxd_begin + block->cxd_count;
    int cxd_index = cxd_begin;

    // Shared Memory - Temporary CX,D
    __shared__ uint8_t s_cxds[threadPerCount * threadWorkCount + 1];
    // Shared Memory - Context states
    __shared__ uint8_t ctxs_data[threadWorkCount][19];

    // Output byte stream
    unsigned char* start = &d_byte[block->byte_index];

    // Init variables
    uint32_t a = 0x8000;
    uint32_t c = 0;
    uint8_t ct = 12;
    uint8_t* bp = start - 1;
    uint8_t* ctxs = ctxs_data[thread_work_index];
    if ( work_thread ) {
        if ( *bp == 0xff ) {
            ct = 13;
        }
        if ( calculate >= calculate_once ) {
            mqc_gpu_reset_cxstate(ctxs);
        }
    }

    while ( cxd_index < cxd_count ) {
        // Load CX,D by this thread
        int cxd_index_thread = cxd_index + thread_index;
        uint8_t cxd_thread = 0;
        if ( cxd_index_thread < cxd_count )
            cxd_thread = d_cxd[cxd_index_thread];
        s_cxds[threadIdx.x] = cxd_thread;

        // Only working threads will work (first thread of each group)
        if ( work_thread ) {
            // Init count
            int count = threadPerCount;
            if ( (cxd_index + count) >= cxd_count )
                count = cxd_count - cxd_index;

            for ( int index = 0; index < count; index++ ) {
                uint8_t cxd = s_cxds[threadIdx.x + index];
                if ( calculate >= calculate_once ) {
                    uint8_t* ctx = &ctxs[mqc_gpu_cxd_get_cx(cxd)];
                    struct mqc_cxstate* state = &d_mqc_gpu_shared_table[*ctx];
                    if ( state->mps == mqc_gpu_cxd_get_d(cxd) ) {
                        mqc_gpu_shared_code_mps(a,c,ct,bp,ctx,state);
                    } else {
                        mqc_gpu_shared_code_lps(a,c,ct,bp,ctx,state);
                    }
                } else {
                    a += cxd;
                }
            }
        }
        cxd_index += threadPerCount;
    }

    if ( work_thread ) {
        if ( calculate >= calculate_once ) {
            // Flush last bytes
            mqc_gpu_shared_flush(a,c,ct,bp); 

            // Set output byte count
            block->byte_count = bp - start;
        } else {
            *bp = a;
        }
    }
}

/** MQ-Coder runtime configuration */
mqc_gpu_configuration config_shared;

/** Documented at declaration */
void*
mqc_gpu_shared_create(struct mqc_configuration * configuration)
{
    // Init lookup table
    mqc_gpu_init_table("d_mqc_gpu_shared_table");

    // Reset configuration
    config_shared.reset();

    // Load configuration
    mqc_gpu_configuration_load(config_shared, configuration);

    // Select kernel
    mqc_gpu_configuration_select_CT(config_shared, kernel_mqc_gpu_shared_encode);

    // Configure L1
    if ( config_shared.kernel != 0 )
        cudaFuncSetCacheConfig(config_shared.kernel,cudaFuncCachePreferL1);
        
    return 0;
}

/** Documented at declaration */
int
mqc_gpu_shared_encode(void* mqc, struct j2k_cblk* d_cblk, int cblk_count, unsigned char * d_cxd, unsigned char * d_byte)
{
    return mqc_gpu_configuration_run(config_shared, d_cblk, cblk_count, d_cxd, d_byte);
}

/** Documented at declaration */
int
mqc_gpu_shared_destroy(void* mqc)
{
    return 0;
}

