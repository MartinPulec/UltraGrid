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

#include "original.h"
#include "common.h"

#define THREAD_BLOCK_SIZE 64

/** MQ-Coder lookup table */
__constant__ struct mqc_cxstate
d_mqc_gpu_original_table[mqc_table_size];

/** MQ-Coder state */
struct mqc_gpu {
    // Code register
    unsigned int c;
    // Current interval value
    unsigned int a;
    // Shift counter for code register
    unsigned int ct;
    // Pointer to current byte
    unsigned char *bp;
    // Start of byte buffer
    unsigned char *start;
    // States of each context
    uint8_t ctxs[19];
    // Pointer to current context
    uint8_t* curctx;
};

/**
 * Perform byte out procedure
 * 
 * @param mqc  MQ-Coder state
 */
__device__ void
mqc_gpu_original_byteout(struct mqc_gpu *mqc)
{
    if ( *mqc->bp == 0xff ) {
        mqc->bp++;
        *mqc->bp = mqc->c >> 20;
        mqc->c &= 0xfffff;
        mqc->ct = 7;
    } else {
        if ( (mqc->c & 0x8000000) == 0 ) {
            mqc->bp++;
            *mqc->bp = mqc->c >> 19;
            mqc->c &= 0x7ffff;
            mqc->ct = 8;
        } else {
            (*mqc->bp)++;
            if ( *mqc->bp == 0xff ) {
                mqc->c &= 0x7ffffff;
                mqc->bp++;
                *mqc->bp = mqc->c >> 20;
                mqc->c &= 0xfffff;
                mqc->ct = 7;
            } else {
                mqc->bp++;
                *mqc->bp = mqc->c >> 19;
                mqc->c &= 0x7ffff;
                mqc->ct = 8;
            }
        }
    }
}

/**
 * Perform renormalize procedure
 * 
 * @param mqc  MQ-Coder state
 */
__device__ void
mqc_gpu_original_renormalize(struct mqc_gpu *mqc)
{
    do {
        mqc->a <<= 1;
        mqc->c <<= 1;
        mqc->ct--;
        if (mqc->ct == 0) {
            mqc_gpu_original_byteout(mqc);
        }
    } while ( (mqc->a & 0x8000) == 0 );
}

/**
 * Perform code MPS procedure
 * 
 * @param mqc  MQ-Coder state
 */
__device__ void
mqc_gpu_original_codemps(struct mqc_gpu *mqc)
{
    int qeval = d_mqc_gpu_original_table[*mqc->curctx].qeval;
    mqc->a -= qeval;
    if ( (mqc->a & 0x8000) == 0 ) {
        if (mqc->a < qeval) {
            mqc->a = qeval;
        } else {
            mqc->c += qeval;
        }
        *mqc->curctx = d_mqc_gpu_original_table[*mqc->curctx].nmps;
        mqc_gpu_original_renormalize(mqc);
    } else {
        mqc->c += qeval;
    }
}

/**
 * Perform code LPS procedure
 * 
 * @param mqc  MQ-Coder state
 */
__device__ void
mqc_gpu_original_codelps(struct mqc_gpu *mqc)
{
    int qeval = d_mqc_gpu_original_table[*mqc->curctx].qeval;
    mqc->a -= qeval;
    if ( mqc->a < qeval ) {
        mqc->c += qeval;
    } else {
        mqc->a = qeval;
    }
    *mqc->curctx = d_mqc_gpu_original_table[*mqc->curctx].nlps;
    mqc_gpu_original_renormalize(mqc);
}

/**
 * Perform flush last bytes procedure
 * 
 * @param mqc  MQ-Coder state
 */
__device__ void
mqc_gpu_original_flush(struct mqc_gpu *mqc)
{
    unsigned int tempc = mqc->c + mqc->a;
    mqc->c |= 0xffff;
    if ( mqc->c >= tempc ) {
        mqc->c -= 0x8000;
    }
    mqc->c <<= mqc->ct;
    mqc_gpu_original_byteout(mqc);
    mqc->c <<= mqc->ct;
    mqc_gpu_original_byteout(mqc);

    if ( *mqc->bp != 0xff ) {
        mqc->bp++;
    }
}

/**
 * Retrieve number of output bytes
 * 
 * @param mqc  MQ-Coder state
 */
__device__ int
mqc_gpu_original_numbytes(struct mqc_gpu *mqc)
{
    return mqc->bp - mqc->start;
}

/**
 * Set current context
 * 
 * @param mqc  MQ-Coder state
 * @param ctxno  Context number
 */
#define mqc_gpu_original_setcurctx(mqc, ctxno) (mqc)->curctx = &(mqc)->ctxs[(int)(ctxno)]

/**
 * Initialize encoder
 * 
 * @param mqc  MQ-Coder state
 * @param bp  Output byte buffer
 */
__device__ void
mqc_gpu_original_init(struct mqc_gpu *mqc, unsigned char *bp)
{
    mqc_gpu_original_setcurctx(mqc, 0);
    mqc->a = 0x8000;
    mqc->c = 0;
    mqc->bp = bp - 1;
    mqc->ct = 12;
    if ( *mqc->bp == 0xff ) {
        mqc->ct = 13;
    }
    mqc->start = bp;
}

/**
 * Set encoder state for context
 * 
 * @param mqc  MQ-Coder state
 * @param ctxno  Context number
 * @param msb
 * @param prob
 */
__device__ void
mqc_gpu_original_setstate(struct mqc_gpu *mqc, int ctxno, int msb, int prob)
{
    mqc->ctxs[ctxno] = msb + (prob << 1);
}

/**
 * Encode decision value by current context
 * 
 * @param mqc  MQ-Coder state
 * @param d  Decision value
 */
__device__ void
mqc_gpu_original_encode(struct mqc_gpu *mqc, int d)
{
    if ( d_mqc_gpu_original_table[*mqc->curctx].mps == d ) {
        mqc_gpu_original_codemps(mqc);
    } else {
        mqc_gpu_original_codelps(mqc);
    }
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
mqc_gpu_original_encode_kernel(struct j2k_cblk* d_cxd_blocks, int cxd_block_count, unsigned char* d_cxds, unsigned char* d_bytes)
{
    int block_index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if ( block_index >= cxd_block_count )
        return;    

    // Get CX,D block and buffer pointer
    struct j2k_cblk* cxd_block = &d_cxd_blocks[block_index];
    unsigned char* bytes = &d_bytes[cxd_block->byte_index];

    // Create encoder handle
    __shared__ struct mqc_gpu mqc_buffer[THREAD_BLOCK_SIZE];
    struct mqc_gpu* mqc = &mqc_buffer[threadIdx.x];

    // Initialize encoder
    mqc_gpu_original_init(mqc,bytes);
    mqc_gpu_reset_cxstate(mqc->ctxs);

    // Process all CX,D pairs
    int cxd_count = cxd_block->cxd_index + cxd_block->cxd_count;
    for ( int cxd_index = cxd_block->cxd_index; cxd_index < cxd_count; cxd_index++ ) {
        unsigned char cxd = d_cxds[cxd_index];
        mqc_gpu_original_setcurctx(mqc,mqc_gpu_cxd_get_cx(cxd));
        mqc_gpu_original_encode(mqc,mqc_gpu_cxd_get_d(cxd));
    }
    mqc_gpu_original_flush(mqc);

    // Set output byte count
    cxd_block->byte_count = mqc_gpu_original_numbytes(mqc);
}

/** Documented at declaration */
void*
mqc_gpu_original_create(struct mqc_configuration * configuration)
{
    // Init lookup table
    mqc_gpu_init_table("d_mqc_gpu_original_table");

    // Configure L1
    cudaFuncSetCacheConfig(mqc_gpu_original_encode_kernel, cudaFuncCachePreferShared);
    
    return 0;
}

/** Documented at declaration */
int
mqc_gpu_original_encode(void* mqc, struct j2k_cblk* d_block, int block_count, unsigned char * d_cxd, unsigned char * d_byte)
{
    dim3 dim_grid;
    dim_grid.x = block_count / THREAD_BLOCK_SIZE + 1;
    if ( dim_grid.x > CUDA_MAXIMUM_GRID_SIZE ) {
        dim_grid.y = dim_grid.x / CUDA_MAXIMUM_GRID_SIZE + 1;
        dim_grid.x = CUDA_MAXIMUM_GRID_SIZE;
    }

    // Run kernel encode
    mqc_gpu_original_encode_kernel<<<dim_grid,THREAD_BLOCK_SIZE>>>(d_block, block_count, d_cxd, d_byte);
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        std::cerr << "Kernel encode failed: " << cudaGetErrorString(cuerr) << std::endl;
        return -1;
    }
    return 0;
}

/** Documented at declaration */
int
mqc_gpu_original_destroy(void* mqc)
{
    return 0;
}

