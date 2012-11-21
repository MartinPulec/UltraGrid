/// 
/// @file    copyback.cu
/// @brief   Replacement of memcpy for CUDA DWT.
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @date    2012-09-10 11:23
///
///
/// Copyright (c) 2011 Martin Jirman
/// All rights reserved.
/// 
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
/// 
///     * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
///     * Redistributions in binary form must reproduce the above copyright
///       notice, this list of conditions and the following disclaimer in the
///       documentation and/or other materials provided with the distribution.
/// 
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
/// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
/// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
/// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
/// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
/// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
/// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.
///


#include "dwt.h"
#include "common.h"



/// Copies 4byte aligned memory from one buffer to another (non overlaping).
/// @param dest  destination for the data (aligned to 4 bytes)
/// @param src  source pointer  (aligned to 4 bytes)
/// @param byteCount  number of bytes to be copied
__global__ static void copyKernel(void * dest, const void * src, int size) {
    // get global thread ID
    const int blockIndex = blockIdx.x + gridDim.x * blockIdx.y;
    const int globalThreadIndex = threadIdx.x + blockDim.x * blockIndex;
    
    // compute and check index of copied bytes
    if(globalThreadIndex * 4 < size) {
        // copy 4 bytes
        ((int*)dest)[globalThreadIndex] = ((const int*)src)[globalThreadIndex];
    }
}


/// Copies partially transformed data from buffer back into input buffer.
/// @param dest  destination for the data (aligned to 4 bytes)
/// @param src  source pointer  (aligned to 4 bytes)
/// @param byteCount  number of bytes to be copied
/// @param stream  pointer to CUDA stream to run in or NULL for default stream
void dwt_cuda_copy(
    void * dest,
    const void * src,
    int byteCount,
    const void * stream
) {
    // select stream
    const cudaStream_t str = stream ? *(const cudaStream_t*)stream : 0;
    
    // grid dimensions
    const int blockSize = 256;
    const int bytesPerBlock = blockSize * sizeof(int);
    dim3 gridSize((byteCount + bytesPerBlock - 1) / bytesPerBlock);
    
    // adjust the grid dimensions to match limits
    while(gridSize.x >= 65536) {
        gridSize.x = (gridSize.x + 1) / 2;
        gridSize.y *= 2;
    }
    
    // launch the copy kernel
    copyKernel<<<gridSize, blockSize, 0, str>>>(dest, src, byteCount);
}



