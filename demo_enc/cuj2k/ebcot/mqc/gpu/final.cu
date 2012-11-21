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

#include "final.h"
#include "common.h"

/** MQ-Coder lookup table */
__constant__ struct mqc_cxstate
d_mqc_gpu_final_table[mqc_table_size];

/**
 * Perform byte out procedure
 * 
 * @param c  Code register
 * @param ct  Free space in code register
 * @param bp  Output byte buffer
 */
__device__ inline void
mqc_gpu_final_byte_out(uint32_t & c, uint8_t & ct, uint8_t* & bp)
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
mqc_gpu_final_code_mps(uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & bp, uint8_t & ctx, struct mqc_cxstate* state)
{
    int qeval =  state->qeval;
    a -= qeval;
    if ( (a & 0x8000) == 0 ) {
        if (a < qeval) {
            a = qeval;
        } else {
            c += qeval;
        }
        ctx = state->nmps;

        a <<= 1;
        c <<= 1;
        ct--;
        if (ct == 0) {
            mqc_gpu_final_byte_out(c, ct, bp);
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
mqc_gpu_final_code_lps(uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & bp, uint8_t & ctx, struct mqc_cxstate* state)
{
    int qeval = state->qeval;
    a -= qeval;
    if ( a < qeval ) {
        c += qeval;
    } else {
        a = qeval;
    }

    ctx = state->nlps;

    int ns = __clz(a) - (sizeof(uint32_t) * 8 - 16);
    a = a << ns;
    if ( ct > ns ) {
        c = c << ns;
        ct = ct - ns;
    } else {
        c = c << ct;
        ns = ns - ct;
        mqc_gpu_final_byte_out(c, ct, bp);
        if ( ct > ns ) {
            c = c << ns;
            ct = ct - ns;
        } else {
            c = c << ct;
            ns = ns - ct;
            mqc_gpu_final_byte_out(c, ct, bp);
            ct = ct - ns;
            c = c << ns;
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
mqc_gpu_final_flush(uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & bp)
{
    unsigned int tempc = c + a;
    c |= 0xffff;
    if ( c >= tempc ) {
        c -= 0x8000;
    }
    c <<= ct;
    mqc_gpu_final_byte_out(c,ct,bp);
    c <<= ct;
    mqc_gpu_final_byte_out(c,ct,bp);
    if ( *bp != 0xff ) {
        bp++;
    }
    c = c;
    ct = ct;
}

/**
 * Encode one CX,D pair
 * 
 * @param cxd  CX,D pair
 * @param a  Interval register
 * @param c  Code register
 * @param ct  Free space in code register
 * @param bp  Output byte buffer
 * @param cxstate  Context states
 * @param d_trunc_size  Byte sizes for truncation points
 * @return true if continue in coding, otherwise false
 */
__device__ inline void
mqc_gpu_final_encode_symbol(uint8_t cxd, uint32_t & a, uint32_t & c, uint8_t & ct, uint8_t* & start, uint8_t* & bp, uint8_t (& cxstate)[19],
                            int & cxd_index, int & cxd_count, int & pass_count, unsigned int* d_trunc_size)
{
    // Skip coding of end of pass CX,D
    if ( mqc_gpu_cxd_is_pass_end(cxd) ) {
        // Increment processed pass
        pass_count++;
        d_trunc_size[pass_count] = (bp - start) + 3;
    }
    // Perform CX,D coding of MPS or LPS
    else {
        uint8_t & ctx = cxstate[mqc_gpu_cxd_get_cx(cxd)];
        struct mqc_cxstate* state = &d_mqc_gpu_final_table[ctx];
        if ( state->mps == mqc_gpu_cxd_get_d(cxd) ) {
            mqc_gpu_final_code_mps(a, c, ct, bp, ctx, state);
        } else {
            mqc_gpu_final_code_lps(a, c, ct, bp, ctx, state);
        }
    }
}

/**
 * Kernel that performs MQ-Encoding for one block
 * 
 * @param d_cblk  Array of code-blocks in device memory
 * @param cblk_count  Count of code-blocks
 * @param d_cxd  Array of input CX,D pairs in device memory
 * @param d_byte  Array of output bytes in device memory
 * @param d_trunc_size  Array of sizes for truncation points (after each pass)
 */
template<
    // Thread Work Count (how many threads in thread block will process code-blocks)
    unsigned int threadWorkCount,
    // Thread Per Count (how many threads will be one group of one working thread, 
    // 1 means only working threads, 2 means separation by 1 thread, etc.)
    unsigned int threadPerCount,
    // Data type used for batch loading (ie. unsigned char, int, double, etc.)
    class cxdLoadType,
    // How many values of specified data type will be used for one batch load
    unsigned int cxdLoadCount
>
__global__ void
mqc_gpu_final_encode_kernel(struct j2k_cblk* d_cblk, int cblk_count, unsigned char* d_cxd, unsigned char * d_byte, unsigned int* d_trunc_sizes)
{
    // Get and check block index
    int cblk_index = (blockIdx.y * gridDim.x + blockIdx.x) * threadWorkCount + threadIdx.x / threadPerCount;
    if ( cblk_index >= cblk_count )
        return;

    // Thread index in count
    int thread_index = threadIdx.x % threadPerCount;

    // Is this thread working (not working threads do nothing)
    bool work_thread = (thread_index) == 0;
    if ( work_thread == false )
        return;

    // Get code-block for working thread to process
    struct j2k_cblk* cblk = &d_cblk[cblk_index];

    // CX,D info
    int cxd_begin = cblk->cxd_index;
    int cxd_count = cxd_begin + cblk->cxd_count;
    int cxd_index = cxd_begin;

    // Output byte stream
    uint8_t* start = &d_byte[cblk->byte_index];

    // Init variables
    uint32_t a = 0x8000;
    uint32_t c = 0;
    uint8_t ct = 12;
    uint8_t* bp = start - 1;
    uint8_t cxstate[19];
    mqc_gpu_reset_cxstate(cxstate);
    // Reset first byte (will be checked for 0xFF in byte out procedure)
    bp[0] = 0;
    int pass_count = 0;
    
    // Output truncation point size pointer (first size is 0 == no bytes at all)
    unsigned int* d_trunc_size = &d_trunc_sizes[cblk->trunc_index];
    d_trunc_size[0] = 0;
    
    if ( sizeof(cxdLoadType) == 1 && cxdLoadCount == 1 ) {
        // Encode CX,D
        for ( cxd_index = cxd_index; cxd_index < cxd_count; cxd_index++ ) {
            uint8_t cxd = d_cxd[cxd_index];
            mqc_gpu_final_encode_symbol(cxd, a, c, ct, start, bp, cxstate, cxd_index, cxd_count, pass_count, d_trunc_size);
         }
    } else {
        // Get count of CX,D for align
        int align_count = cxd_index % sizeof(cxdLoadType);
        if ( align_count > 0 ) {
            // Make differ
            align_count = cxd_index + sizeof(cxdLoadType) - align_count;
            // Check count
            if ( align_count > cxd_count )
                align_count = cxd_count;
            // Encode align symbols
            for ( cxd_index = cxd_index; cxd_index < align_count; cxd_index++ ) {
                uint8_t cxd = d_cxd[cxd_index];
                mqc_gpu_final_encode_symbol(cxd, a, c, ct, start, bp, cxstate, cxd_index, cxd_count, pass_count, d_trunc_size);
            }
        }

        // Encode
        while ( cxd_index < cxd_count ) {
            // Init count
            int count = sizeof(cxdLoadType) * cxdLoadCount;
            if ( (cxd_index + count) >= cxd_count ) {
                count = cxd_count - cxd_index;
            }

            // Load CX,D by load type
            cxdLoadType cxd_data[cxdLoadCount];
            for ( int index = 0; index < cxdLoadCount; index++ )
                cxd_data[index] = reinterpret_cast<cxdLoadType*>(&d_cxd[cxd_index])[index];

            // Encode CX,D
            for ( int index = 0; index < count; index++ ) {
                uint8_t cxd = reinterpret_cast<uint8_t*>(&cxd_data)[index];
                mqc_gpu_final_encode_symbol(cxd, a, c, ct, start, bp, cxstate, cxd_index, cxd_count, pass_count, d_trunc_size);
            }
        
            cxd_index += count;
        }
    }
    
    // Flush last bytes
    mqc_gpu_final_flush(a, c, ct, bp);

    // Set output byte count and correct last truncation point
    cblk->byte_count = bp - start;
    d_trunc_size[pass_count] = bp - start;
    // Set processed pass count
    cblk->pass_count = pass_count;
    cblk->trunc_count = pass_count + 1; // one truncation is always at the begin (meaning that the codeblock is not coded at all)
}

/** MQ-Coder kernel type */
typedef void (*mqc_kernel)(struct j2k_cblk*, int, unsigned char*, unsigned char*, unsigned int*);

/** Thread Work Count (how many threads in thread block will process code-blocks) */
const int twc = 64;
/** Thread Per Count (how many threads will be one group of one working thread, 1 means only working threads, 2 means separation by 1 thread, etc.) */
const int tpc = 1;

/** Documented at declaration */
void*
mqc_gpu_final_create(struct mqc_configuration * configuration)
{
    // Init lookup table
    mqc_gpu_init_table("d_mqc_gpu_final_table");
    
    // Select kernel   TODO: version with/without truncation point tracking
    mqc_kernel mqc = NULL;
    mqc = &mqc_gpu_final_encode_kernel<twc, tpc, uint64_t, 16>;
    assert(mqc != NULL);

    // Configure L1
    cudaFuncSetCacheConfig(mqc, cudaFuncCachePreferL1);
    
    return (void*)mqc;
}

/** Documented at declaration */
int
mqc_gpu_final_encode(void* mqc, struct j2k_cblk* d_cblk, int cblk_count, unsigned char * d_cxd, unsigned char * d_byte, unsigned int * d_trunc_sizes, cudaStream_t stream)
{
    // Calculate grid and block sizes
    int count = cblk_count / twc + 1;
    dim3 dim_grid;
    dim_grid.x = count;
    if ( dim_grid.x > CUDA_MAXIMUM_GRID_SIZE ) {
        dim_grid.x = CUDA_MAXIMUM_GRID_SIZE;
        dim_grid.y = count / CUDA_MAXIMUM_GRID_SIZE + 1;
    }
    dim3 dim_block(twc * tpc, 1);
    
    // Perform encoding
    ((mqc_kernel)mqc)<<<dim_grid, dim_block, 0, stream>>>(
        d_cblk,
        cblk_count,
        d_cxd,
        d_byte,
        d_trunc_sizes
    );
//     cudaError cuerr = cudaThreadSynchronize();
//     if ( cuerr != cudaSuccess ) {
//         std::cerr << "MQ-Coder Kernel encoding failed: " << cudaGetErrorString(cuerr) << std::endl;
//         return -1;
//     }
    return 0;
}

/** Documented at declaration */
int
mqc_gpu_final_destroy(void* mqc)
{
    // De-configure L1
    cudaFuncSetCacheConfig(((mqc_kernel)mqc), cudaFuncCachePreferNone);
    
    return 0;
}

