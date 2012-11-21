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
 
#include <unistd.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>

#include "../common.h"
#include "preprocessor.h"
#include "preprocessor_ct.h"
#include "preprocessor_store.h"

#define RGB_THREADS(bits) 256

// #define RGB_8BIT_THREADS 256
// #define RGB_10BIT_THREADS 256
// #define RGB_12BIT_THREADS 256
// #define RGB_14BIT_THREADS 256
// #define RGB_16BIT_THREADS 256
// #define RGB_THREADS(bits) \
//     (bits == 8 ? RGB_8BIT_THREADS : \
//     (bits == 10 ? RGB_10BIT_THREADS : \
//     (bits == 12 ? RGB_12BIT_THREADS : \
//     (bits == 14 ? RGB_14BIT_THREADS : \
//     (bits == 16 ? RGB_16BIT_THREADS : 0))))) \
// 
// #define BW_8BIT_THREADS RGB_8BIT_THREADS
// #define BW_10BIT_THREADS RGB_10BIT_THREADS
// #define BW_12BIT_THREADS RGB_12BIT_THREADS
// #define BW_14BIT_THREADS RGB_14BIT_THREADS
// #define BW_16BIT_THREADS RGB_16BIT_THREADS
#define BW_THREADS(bits) RGB_THREADS(bits)



// reads consecutive BYTE_COUNT bytes as little endian number 
// starting at 'offset' in 'data' buffer
template <int BYTE_COUNT>
__device__ inline int read_bytes(const unsigned char * data, int offset);

template <>
__device__ inline int read_bytes<0>(const unsigned char * data, int offset) {
    return 0;
}

template <int BYTE_COUNT>
__device__ inline int read_bytes(const unsigned char * data, int offset) {
    enum { B = BYTE_COUNT - 1 };
    return data[offset + B] * (1 << (8 * B)) + read_bytes<B>(data, offset);
}



/**
 * Kernel - Copy image source data into three separated component buffers
 * Each thread loads one pixel (all component samples of the pixel).
 *
 * @param d_c1  First component buffer
 * @param d_c2  Second component buffer
 * @param d_c3  Third component buffer
 * @param d_source  Image source data
 * @param pixel_count  Number of pixels to copy
 * @return void
 */
template<int bit_depth, bool is_signed, enum j2k_component_transform transform, class data_type>
__global__ void d_rgb_to_comp(data_type* d_c1, data_type* d_c2, data_type* d_c3, const unsigned char *d_source, int pixel_count)
{
    enum { COMP_BYTES = bit_depth > 8 ? (bit_depth > 16 ? 4 : 2) : 1 };    // number of bytes per pixel's component
    enum { PIXEL_BYTES = 3 * COMP_BYTES };               // number of bytes per pixel
    enum { SHIFT = 8 * COMP_BYTES - bit_depth };  // shift to discard unused lsbs
    
    // offset of the loaded pixel in image (in raster order)
    const int pix_idx = threadIdx.x + blockIdx.x * RGB_THREADS(bit_depth);
    if(pix_idx < pixel_count) {
        const int offset = pix_idx * PIXEL_BYTES;
    
        // TODO: implement signed loading
        const int r = read_bytes<COMP_BYTES>(d_source, offset + 0 * COMP_BYTES) >> SHIFT;
        const int g = read_bytes<COMP_BYTES>(d_source, offset + 1 * COMP_BYTES) >> SHIFT;
        const int b = read_bytes<COMP_BYTES>(d_source, offset + 2 * COMP_BYTES) >> SHIFT;
        
        j2k_store_component<bit_depth, is_signed, transform>::perform(d_c1, d_c2, d_c3, r, g, b, pix_idx);
    }
}

/**
 * Copy image source data into three separated component buffers
 *
 * @param d_c1  First component buffer
 * @param d_c2  Second component buffer
 * @param d_c3  Third component buffer
 * @param d_source  Image source data
 * @return void
 */
template<int bit_depth,  bool is_signed, enum j2k_component_transform transform, class data_type>
inline void rgb_to_comp(data_type* d_c1, data_type* d_c2, data_type* d_c3, const unsigned char * d_source, int width, int height, cudaStream_t stream) {
    const int pixel_count = width * height;
    const int thread_count = RGB_THREADS(bit_depth);
    const int tblock_count = (pixel_count + thread_count - 1) / thread_count;
    
    // Kernel
//     assert(alignedSize % (RGB_THREADS(bit_depth) * 3) == 0);

    //CTIMERSTART (cstart);
    d_rgb_to_comp<bit_depth, is_signed, transform>
                 <<<tblock_count, thread_count, 0, stream>>>
                 (d_c1, d_c2, d_c3, d_source, pixel_count);
    //CTIMERSTOP (cstop);

    //printf ("   - RGB -> %dbit Components: ", bit_depth);
    //PRINTMS;
//     cudaCheckAsyncError ("rgb_to_comp kernel");
}

/**
 * Kernel - Copy Black-White image source data into one component buffer
 *
 * @param d_c  Component buffer
 * @param d_source  Image source data
 * @param pixel_count  Number of pixels to copy
 * @return void
 */
template<int bit_depth, bool is_signed, class data_type>
__global__ void d_bw_to_comp(data_type* d_c, const unsigned char * d_source, int pixel_count)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;

    __shared__ unsigned char s_data[BW_THREADS(bit_depth)];

    // Copy data to shared mem by 4bytes other checks are not necessary, 
    // since d_source buffer is aligned to sharedDataSize
    if ((x * 4) < BW_THREADS(bit_depth)) {
        int *s = (int *) d_source;
        int *d = (int *) s_data;
        d[x] = s[(gX >> 2) + x];
    }
    __syncthreads();

    int c;
    
    switch (bit_depth) {
        case 8:
            c = (int)(s_data[x]);
            break;
        // TODO: implement correct reading for non 8-bit values
    }

    int globalOutputPosition = gX + x;
    if ( globalOutputPosition < pixel_count ) {
        j2k_store_component<bit_depth, is_signed>::perform(d_c, c, globalOutputPosition);
    }
}

/**
 * Copy Black-White image source data into one component buffer
 *
 * @param d_c  Component buffer
 * @param d_source  Image source data
 * @return void
 */
template<int bit_depth, bool is_signed, class data_type>
inline void bw_to_comp(data_type* d_c, const unsigned char * d_source, int width, int height, cudaStream_t stream) {
    int pixel_count = width * height;
    // Aligned to thread block size
    int alignedSize = DIVANDRND (width * height, BW_THREADS(bit_depth)) * BW_THREADS(bit_depth);
    // Timing
    //CTIMERINIT;
    
    // Kernel
    dim3 threads (BW_THREADS(bit_depth));
    dim3 grid (alignedSize / BW_THREADS(bit_depth));
    assert (alignedSize % BW_THREADS(bit_depth) == 0);
    
    //CTIMERSTART (cstart);
    d_bw_to_comp<bit_depth, is_signed><<<grid, threads, 0, stream>>> ((data_type*)d_c, d_source, pixel_count);
    //CTIMERSTOP (cstop);

    //printf ("   - BW -> %dbit Components: ", bits);
    //PRINTMS;
//     cudaCheckAsyncError ("bw_to_comp kernel");
}

/**
 * Preprocessor functions
 *
 * @template bit_depth
 * @template is_signed
 * @template data_type
 * @template tranform
 */
template<int bit_depth, bool is_signed, typename data_type, enum j2k_component_transform transform>
struct preprocessor_function;

/** Specialization [data_type = int] */
template<int bit_depth,  bool is_signed, enum j2k_component_transform transform>
struct preprocessor_function<bit_depth, is_signed, int, transform> {
    /** 1 color component */
    static void comp1(void* d_c, void* dummy1, void* dummy2, const unsigned char* d_source, int width, int height, cudaStream_t stream)
    {
        bw_to_comp<bit_depth, is_signed>((int*)d_c, d_source, width, height, stream);
    }
    /** 3 color component */
    static void comp3(void* d_c1, void* d_c2, void* d_c3, const unsigned char* d_source, int width, int height, cudaStream_t stream)
    {
        rgb_to_comp<bit_depth, is_signed, transform>((int*)d_c1, (int*)d_c2, (int*)d_c3, d_source, width, height, stream);
    }
};

/** Specialization [data_type = float] */
template<int bit_depth, bool is_signed, enum j2k_component_transform transform>
struct preprocessor_function<bit_depth, is_signed, float, transform> {
    /** 1 color component */
    static void comp1(void* d_c, void* dummy1, void* dummy2, const unsigned char* d_source, int width, int height, cudaStream_t stream)
    {
        bw_to_comp<bit_depth, is_signed>((float*)d_c, d_source, width, height, stream);
    }
    /** 3 color component */
    static void comp3(void* d_c1, void* d_c2, void* d_c3, const unsigned char* d_source, int width, int height, cudaStream_t stream)
    {
        rgb_to_comp<bit_depth, is_signed, transform>((float*)d_c1, (float*)d_c2, (float*)d_c3, d_source, width, height, stream);
    }
};

/** Select [comp_count] */
template<int bit_depth, bool is_signed, typename data_type, enum j2k_component_transform transform>
preprocessor_function_t
preprocessor_get_function(int comp_count)
{
    assert((comp_count == 1 && transform == CT_NONE) || comp_count == 3);

    switch ( comp_count ) {
        case 1:
            return &preprocessor_function<bit_depth, is_signed, data_type, transform>::comp1;
        case 3:
            return &preprocessor_function<bit_depth, is_signed, data_type, transform>::comp3;
    }
    return 0;
}

/** Select [bit_depth] */
template<bool is_signed, typename data_type, enum j2k_component_transform transform>
preprocessor_function_t
preprocessor_get_function(int bit_depth, int comp_count)
{
    assert(bit_depth == 8 || bit_depth == 10 || bit_depth == 12 || bit_depth == 14 || bit_depth == 16);

    switch ( bit_depth ) {
        case 8:
            return preprocessor_get_function<8, is_signed, data_type, transform>(comp_count);
        case 10:
            return preprocessor_get_function<10, is_signed, data_type, transform>(comp_count);
        case 12:
            return preprocessor_get_function<12, is_signed, data_type, transform>(comp_count);
        case 14:
            return preprocessor_get_function<14, is_signed, data_type, transform>(comp_count);
        case 16:
            return preprocessor_get_function<16, is_signed, data_type, transform>(comp_count);
    }
    return 0;
}

/** Select [is_signed] */
template<typename data_type, enum j2k_component_transform transform>
preprocessor_function_t
preprocessor_get_function(int bit_depth, int is_signed, int comp_count)
{
    switch ( is_signed ) {
        case 0:
            return preprocessor_get_function<false, data_type, transform>(bit_depth, comp_count);
        default:
            return preprocessor_get_function<true, data_type, transform>(bit_depth, comp_count);
    }
}

/** Select [data_type] and [transform] */
preprocessor_function_t
preprocessor_get_function(int bit_depth, int is_signed, int comp_count, enum j2k_compression_mode compression_mode, enum j2k_component_transform transform)
{
    switch ( compression_mode ) {
        case CM_LOSSLESS:
            switch ( transform ) {
                case CT_NONE:
                    return preprocessor_get_function<int, CT_NONE>(bit_depth, is_signed, comp_count);
                case CT_REVERSIBLE:
                    return preprocessor_get_function<int, CT_REVERSIBLE>(bit_depth, is_signed, comp_count);
            }
            break;
        case CM_LOSSY_FLOAT:
            switch ( transform ) {
                case CT_NONE:
                    return preprocessor_get_function<float, CT_NONE>(bit_depth, is_signed, comp_count);
                case CT_IRREVERSIBLE:
                    return preprocessor_get_function<float, CT_IRREVERSIBLE>(bit_depth, is_signed, comp_count);
            }
            break;
    }
    return 0;
}

