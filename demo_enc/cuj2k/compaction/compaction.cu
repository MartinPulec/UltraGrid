/* 
 * Copyright (c) 2012, Martin Jirman (martin.jirman@cesnet.cz)
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

#include "compaction.h"




/// Only initializes total size to 0 before actual compaction.
/// @param size  pointer to size variable, which should be cleared
__global__ static void compaction_init_kernel(unsigned int * const size) {
    *size = 0;
}



/// Compacts streams of all codeblocks - they are scattered in big buffer. 
/// This places them into small region of the buffer to speed up the 
/// memcpy to host.
/// @param cblk_count  total count of codeblocks
/// @param cblks  pointer to array of codeblock info structures
/// @param total_size  pointer to place, where total byte siz eis accumulated
/// @param src_buffer  pointer to source byte buffer
/// @param dest_buffer  pointer to destination byte buffer
__global__ static void compaction_copy_kernel(const int cblk_count,
                                              j2k_cblk * const cblks,
                                              unsigned int * const total_size,
                                              const void * const src_buffer,
                                              void * const dest_buffer) {
    // get global index of this thread's warp (each warp processes 1 codeblock)
    const int threadblock_idx = blockIdx.x + gridDim.x * blockIdx.y;
    const int global_warp_idx = threadblock_idx * blockDim.y + threadIdx.y;
    
    // make sure that this warp's codeblock index is in bounds
    if(global_warp_idx >= cblk_count) {
        return;
    }
    
    // pointer to shared memory for sharing data offsets
    extern __shared__ unsigned int s_offsets_all[];
    
    // pointer to warp's codeblock
    j2k_cblk * const cblk = cblks + global_warp_idx;
    
    // size of warp's codeblock (rounded up to 16 bytes)
    const unsigned int cblk_size = (cblk->byte_count + 15) & ~15;
    
    // first thread of each warp allocates space for codeblock's bytes 
    // in compact buffer
    if(0 == threadIdx.x) {
        const unsigned int offset = atomicAdd(total_size, cblk_size);
        s_offsets_all[threadIdx.y] = offset;
        cblk->byte_index_compact = offset;
    }
    
    // All threads of each warp read offset (written by thread #0) of compacted
    // bytes in output buffer and prepare input and output pointers with 
    // pre-added thread index. Each thread copies 16 bytes in each iteration.
    const unsigned int dest_offset = s_offsets_all[threadIdx.y];
    const unsigned int src_offset = cblk->byte_index;
    const int4 * src = (const int4*)src_buffer + src_offset / 16 + threadIdx.x;
    int4 * dest = (int4*)dest_buffer + dest_offset / 16 + threadIdx.x;
    
    // number of 16byte chunks to be copied 
    const unsigned int copy_count = cblk_size / 16;

    // copy all codeblock's bytes in parallel
    for(int copy_idx = threadIdx.x; copy_idx < copy_count; copy_idx += 32) {
        // copy 16 bytes
        *dest = *src;
        
        // advance both pointers
        dest += 32;
        src += 32;
    }
}



/// Initializes compaction stuff.
/// @return 0 for success, nonzero for failure
int j2k_compaction_init() {
    // configure more shared memory for both kernels
    const cudaFuncCache p = cudaFuncCachePreferShared;
    if(cudaSuccess != cudaFuncSetCacheConfig(compaction_init_kernel, p)) {
        return -1;
    }
    if(cudaSuccess != cudaFuncSetCacheConfig(compaction_copy_kernel, p)) {
        return -2;
    }
    
    // indicate success
    return 0;
}



/// Compacts streams of codeblocks into small memory region for faster memcpy.
/// @param enc  pointer to encoder instance
/// @param stream  stream to run in
/// @return 0 for success, nonzero for failure
int j2k_compaction_run(struct j2k_encoder* enc, cudaStream_t stream) {
    // clear the compact size
    compaction_init_kernel<<<1, 1, 0, stream>>>(enc->d_compact_size);
    
    // compaction kernel launch configuration
    const int warps_per_block = 8;
    const dim3 block_size(32, warps_per_block);
    dim3 grid_size((enc->cblk_count + warps_per_block - 1) / warps_per_block);
    
    // adjust grid size not to exceed limits
    while(grid_size.x >= 65536) {
        grid_size.x = (grid_size.x + 1) / 2;
        grid_size.y *= 2;
    }
    
    // dynamic shared memory size of the compaction kernel (integer per warp)
    const int shmem_size = sizeof(unsigned int) * warps_per_block;
    
    // launch the compaction kernel in the right stream
    compaction_copy_kernel<<<grid_size, block_size, shmem_size, stream>>>(
        enc->cblk_count,
        enc->d_cblk,
        enc->d_compact_size,
        enc->d_byte,
        enc->d_byte_compact
    );
    
    // indicate success
    return 0;
}

