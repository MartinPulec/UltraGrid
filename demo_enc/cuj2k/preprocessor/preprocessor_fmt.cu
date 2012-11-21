/* 
 * Copyright (c) 2009, Martin Jirman
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

#include <stdlib.h>
#include <string.h>
#include "preprocessor_fmt.h"



/// Irreversible (float) MCT.
__device__ inline void mct(float & r, float & g, float & b) {
    const float new_r = r * 0.29900f + g * 0.58700f + b * 0.11400f;
    const float new_g = -r * 0.16875f - g * 0.33126f + b * 0.50000f;
    const float new_b = r * 0.50000f - g * 0.41869f - b * 0.08131f;
    r = new_r;
    g = new_g;
    b = new_b;
}


/// Reversible (integer) MCT.
__device__ inline void mct(int & r, int & g, int & b) {
    const int new_r = (r + 2 * g + b) >> 2;
    const int new_g = b - g;
    const int new_b = r - g;
    r = new_r;
    g = new_g;
    b = new_b;
}


/// Common loading function template.
template <j2k_input_format FMT> 
__device__ inline void load(const void * data, const int index,
                            int & r, int & b, int & g);


/// Decomposes 3 unsigned 10bit values from 32bit DPX-style pixel packing.
__device__ inline void decompose_10b_pixel(const uint in,
                                           int & r, int & g, int & b) {
    r = 0x3FF & (in >> 22);
    g = 0x3FF & (in >> 12);
    b = 0x3FF & (in >> 2);
}


/// Loads 3 10bit samples of pixel with index 'index', 
/// packed into 32bit little endian value.
template <> 
__device__ inline void load<J2K_FMT_R10_G10_B10_X2_L>(const void * data,
                                                     const int index,
                                                     int & r, int & g, int & b) {
    decompose_10b_pixel(((uint*)data)[index], r, g, b);
}


/// Loads 3 10bit samples of pixel with index 'index', 
/// packed into 32bit big endian value.
template <> 
__device__ inline void load<J2K_FMT_R10_G10_B10_X2_B>(const void * data,
                                                      const int index,
                                                      int & r, int & g, int & b) {
    decompose_10b_pixel(__byte_perm(((uint*)data)[index], 0, 0x0123), r, g, b);
}


/// Loads 3 little endian samples of pixel with index 'index'.
template <> 
__device__ inline void load<J2K_FMT_R16_G16_B16_L>(const void * data,
                                                   const int index,
                                                   int & r, int & g, int & b) {
    const ushort3 val = ((ushort3*)data)[index];
    r = val.x;
    g = val.y;
    b = val.z;
}


/// Swaps bytes in unsigned 16bit value.
__device__ inline int bswap_u16(const int in) {
    return (in >> 8) + (in & 0xFF) * 256;
}


/// Loads 3 big endian samples of pixel with index 'index'.
template <>
__device__ inline void load<J2K_FMT_R16_G16_B16_B>(const void * data,
                                                   const int index,
                                                   int & r, int & g, int & b) {
    // load as little endian and swap bytes
    load<J2K_FMT_R16_G16_B16_L>(data, index, r, g, b);
    r = bswap_u16(r);
    g = bswap_u16(g);
    b = bswap_u16(b);
}


/// Parameters of the preprocessor kernel.
struct preprocessor_params {
    const void * data;   ///< pointer to packed data
    void * out_0;        ///< pointer to output buffer for first component
    void * out_1;        ///< pointer to output buffer for second component
    void * out_2;        ///< pointer to output buffer for third component
    int pix_count;       ///< total count of pixels
};


/// General implementation of formatted preprocessor
/// @tparam FMT  one of supported input data formats
/// @tparam MCT  true if MCT transform should be done
/// @tparam LOG  true if input values are in Cineon logarithmic scale
/// @tparam DEPTH  bit depth of input data
/// @tparam OUT_T  type of output values (float of int)
/// @param params  common parameters for preprocessor
/// @param lut  pointer to lookup table for Cineon LOG to linear conversion
///             (the table has 16bit indices and values have correct type)
///             or NULL if input is linear (and LOG is thus false)
template <j2k_input_format FMT, bool MCT, bool LOG, int DEPTH, typename OUT_T>
__global__ static void fmt_kernel(const preprocessor_params params,
                                  const OUT_T * const lut) {
    // index of the pixel
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int pix_idx = threadIdx.x + block_idx * blockDim.x;
    
    // load 3 pixel coefficients if pixel not out of range
    if(pix_idx < params.pix_count) {
        int x_in, y_in, z_in;
        load<FMT>(params.data, pix_idx, x_in, y_in, z_in);
        
        // apply the LUT or only alter the range of values to be symetric
        // (LUT's range of values is already symmetric.)
        const int MID = 1 << (DEPTH - 1);
        OUT_T x = LOG ? lut[x_in] : (OUT_T)(x_in - MID);
        OUT_T y = LOG ? lut[y_in] : (OUT_T)(y_in - MID);
        OUT_T z = LOG ? lut[z_in] : (OUT_T)(z_in - MID);
        
        // possibly apply MCT
        if(MCT) {
            mct(x, y, z);
        }
        
        // finally save 3 values
        ((OUT_T*)params.out_0)[pix_idx] = x;
        ((OUT_T*)params.out_1)[pix_idx] = y;
        ((OUT_T*)params.out_2)[pix_idx] = z;
    }
}


/// A pair of simmilar luts for float and for int data types.
struct lut_pair {
    int * ints;
    float * floats;
};


/// Declaration of preprocessor instance type.
struct j2k_fmt_preprocessor {
    lut_pair log2lin_16b;
    lut_pair log2lin_10b;
};


template <j2k_input_format FMT, bool MCT, bool LOG, int DEPTH, typename OUT_T>
int run(const preprocessor_params & p, const OUT_T * lut, cudaStream_t str) {
    // prepare launch configuration
    const int bSize = 256;
    dim3 gSize((p.pix_count + bSize - 1) / bSize);
    
    // adjust grid size
    while(gSize.x >= 65536) {
        gSize.x = (gSize.x + 1) / 2;
        gSize.y *= 2;
    }
    
    // run the kernel
    fmt_kernel<FMT, MCT, LOG, DEPTH, OUT_T><<<gSize, bSize, 0, str>>>(p, lut);
    
    // nothing checked => nothing fails
    return 0;
}


/// Selects preprocessor kernel according to MCT, lossy/lossles and log/lin.
template <j2k_input_format FMT, int BPP>
int select_kernel(const preprocessor_params & p,
                  bool mct,
                  bool log,
                  enum j2k_compression_mode compression_mode,
                  const lut_pair & luts,
                  cudaStream_t str) {
    // select the right template version of the kernel
    if(compression_mode == CM_LOSSLESS) {
        if(mct) {
            if(log) {
                return run<FMT, true, true, BPP, int>(p, luts.ints, str);
            } else {
                return run<FMT, true, false, BPP, int>(p, luts.ints, str);
            }
        } else {
            if(log) {
                return run<FMT, false, true, BPP, int>(p, luts.ints, str);
            } else {
                return run<FMT, false, false, BPP, int>(p, luts.ints, str);
            }
        }
    } else if(compression_mode == CM_LOSSY_FLOAT) {
        if(mct) {
            if(log) {
                return run<FMT, true, true, BPP, float>(p, luts.floats, str);
            } else {
                return run<FMT, true, false, BPP, float>(p, luts.floats, str);
            }
        } else {
            if(log) {
                return run<FMT, false, true, BPP, float>(p, luts.floats, str);
            } else {
                return run<FMT, false, false, BPP, float>(p, luts.floats, str);
            }
        }
    } else {
        return -1;
    }
}



/// Computes one value of the LOG->LIN LUT with given size.
/// Output values' range is balanced (centered at zero).
/// @tparam IN_RANGE   number of input vaues (depends on input bit depth)
/// @tparam OUT_RANGE  number of output values (depends on output bit depth)
/// @param idx     index of the value to be computed
/// @param outBpp  bits per pixel of output values (determines their count)
template <int IN_RANGE, int OUT_RANGE>
__device__ static float lut_prepare_idx(const int idx) {
    // numbers in comments refer to document:
    //   Conversion of 10-bit Log Film Data To 8-bit Linear or Video Data
    //   for The Cineon Digital Film System
    //   Version 2.1
    //   July 26, 1995
    
    // Constants:
    const int refWhite = (685 * IN_RANGE) / 1024;  // input white point (685 for Cineon Log)
    const int refBlack = (95 * IN_RANGE) / 1024;   // input black point (95 for Cineon Log)
    const float outGamma = 1.7f;                   // output gamma (e.g. 1.7 for display, 1.0 for film, ...)
    const float density = 1.7f;                    // output gamma density (1.7 for Cineon Log)
    const float filmGamma = 0.6f;                  // input film gamma (0.6 for Cineon Log)
    const int softClip = (0 * IN_RANGE) / 1024;    // white soft clipping, (0 = no soft clipping (default), 20% of range = very noticeable clipping)
    const float balance = OUT_RANGE * 0.5f;
    
    // 5.1 Determine the breakpoint for softclip
    const int breakpoint = refWhite - softClip;
    
    // 5.2 Clamp black to 0
    if(idx < refBlack) {
        return 0.0f - balance;
    }
    
    // 5.3 Compute LUT values between Refblack and breakpoint
    const float outWhite = OUT_RANGE - 1.0f;
    const float scale = (0.002f/filmGamma) * (outGamma/density);
    const float gain = outWhite / (1 - pow(10.0f, (refBlack - refWhite) * scale));
    const float offset = gain - outWhite;
    float result;
    if(idx < breakpoint) {
        result = pow(10.0f, (idx - refWhite) * scale) * gain - offset;
    } else {
        // 5.4 Compute softclip above breakpoint
        const float kneeOffset = pow(10.0f, (breakpoint - refWhite) * scale) * gain - offset;
        const float kneeGain = (outWhite - kneeOffset) / pow(5.0f * softClip, softClip / 100.0f);
        result = pow((float)(idx - breakpoint), softClip / 100.0f) * kneeGain + kneeOffset;
    }
    
    // 5.5 Clip white to max
    return min(outWhite, result) - balance;
}



/// Prepares GPU lookup tables.
__global__ static void lut_prepare_kernel(const j2k_fmt_preprocessor p) {
    // index of this thread among all threads of all blocks
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < 65536) {
        const float val16 = lut_prepare_idx<65536, 65536>(idx);
        p.log2lin_16b.floats[idx] = val16;
        p.log2lin_16b.ints[idx] = (int)round(val16);
        
        if(idx < 1024) {
            const float val10 = lut_prepare_idx<1024, 1024>(idx);
            p.log2lin_10b.floats[idx] = val10;
            p.log2lin_10b.ints[idx] = (int)round(val10);
        }
    }
}



/**
 * Initialize a new instance of fixed-format-data preprocessor.
 * @return either a new instance of fixed-format-preprocessor or null if failed
 */
struct j2k_fmt_preprocessor * j2k_fmt_preprocessor_create() {
    const size_t size = sizeof(j2k_fmt_preprocessor);
    j2k_fmt_preprocessor * const p = (j2k_fmt_preprocessor*)malloc(size);
    
    if(p) {
        memset(p, 0, size);
        cudaMalloc(&p->log2lin_16b.ints, sizeof(int) * 65536);
        cudaMalloc(&p->log2lin_16b.floats, sizeof(float) * 65536);
        cudaMalloc(&p->log2lin_10b.ints, sizeof(int) * 1024);
        cudaMalloc(&p->log2lin_10b.floats, sizeof(float) * 1024);
        if(p->log2lin_16b.floats && p->log2lin_16b.ints
                && p->log2lin_10b.floats && p->log2lin_10b.ints) {
            lut_prepare_kernel<<<256, 256>>>(*p);
            if(cudaSuccess == cudaThreadSynchronize()) {
                return p;
            }
        }
        j2k_fmt_preprocessor_destroy(p);
    }
    return 0;
}


/**
 * Releases all resources associated with some instance of preprocessor.
 * @param preprocessor  pointer to preprocessor instance
 */
void j2k_fmt_preprocessor_destroy(struct j2k_fmt_preprocessor * p) {
    if(p) {
        if(p->log2lin_16b.floats) {
            cudaFree(p->log2lin_16b.floats);
        }
        if(p->log2lin_16b.ints) {
            cudaFree(p->log2lin_16b.ints);
        }
        if(p->log2lin_10b.floats) {
            cudaFree(p->log2lin_10b.floats);
        }
        if(p->log2lin_10b.ints) {
            cudaFree(p->log2lin_10b.ints);
        }
        free(p);
    }
}


/** 
 * Preprocess image data saved in GPU buffer in one of supported formats.
 * @param preprocessor  pointer to instance of the preprocessor
 * @param format  input data format
 * @param size  input data array size
 * @param comp_count  expected number of color components
 * @param bit_depth  expected pixel-component bit depth
 * @param is_signed  nonzero if pixel values should be signed
 * @param mode  output floats or ints
 * @param in_gpu_ptr  pointer to input GPU data
 * @param out_gpu_ptrs  pointer to output GPU pointers for each component
 * @param is_log  nonzero if input is in logarithmic color space (Cineon Log)
 * @param mct  nonzero for MCT to be used
 * @param stream  stream to run in
 * @return 0 if OK, nonzero otherwise
 */
int
j2k_fmt_preprocessor_preprocess(
    struct j2k_fmt_preprocessor * const instance,
    enum j2k_input_format format,
    struct j2k_size size,
    int comp_count,
    int bit_depth,
    int is_signed,
    enum j2k_compression_mode mode,
    const void * in_gpu_ptr,
    void ** out_gpu_ptrs,
    int is_log,
    int mct,
    cudaStream_t stream
) {
    // get expected parameter values according to format
    int expect_bit_depth = 8;
    int expect_comp_count = 3;
    bool expect_signed = false;
    switch(format) {
        case J2K_FMT_R10_G10_B10_X2_L:
        case J2K_FMT_R10_G10_B10_X2_B:
            expect_bit_depth = 10;
            break;
        case J2K_FMT_R16_G16_B16_L:
        case J2K_FMT_R16_G16_B16_B:
            expect_bit_depth = 16;
            break;
        default:
            return -1; // unknown output format
    }
    
    // check parameter values
    if(expect_signed != (bool)is_signed) return -2;
    if(expect_bit_depth != bit_depth) return -3;
    if(expect_comp_count != comp_count) return -4;
    if(0 == in_gpu_ptr) return -5;
    
    // pack parameters for the kernel
    preprocessor_params params;
    params.data = in_gpu_ptr;
    params.out_0 = out_gpu_ptrs[0];
    params.out_1 = out_gpu_ptrs[1];
    params.out_2 = out_gpu_ptrs[2];
    params.pix_count = size.height * size.width;
    
    // preprocess it!
    switch(format) {
        case J2K_FMT_R10_G10_B10_X2_L:
            return select_kernel<J2K_FMT_R10_G10_B10_X2_L, 10>
                    (params, mct, is_log, mode, instance->log2lin_10b, stream);
        case J2K_FMT_R10_G10_B10_X2_B:
            return select_kernel<J2K_FMT_R10_G10_B10_X2_B, 10>
                    (params, mct, is_log, mode, instance->log2lin_10b, stream);
        case J2K_FMT_R16_G16_B16_L:
            return select_kernel<J2K_FMT_R16_G16_B16_L, 16>
                    (params, mct, is_log, mode, instance->log2lin_16b, stream);
        case J2K_FMT_R16_G16_B16_B:
            return select_kernel<J2K_FMT_R16_G16_B16_B, 16>
                    (params, mct, is_log, mode, instance->log2lin_16b, stream);
        default:
            return -1; // unknown input format
    }
}

