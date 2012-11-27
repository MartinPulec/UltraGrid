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

#ifndef J2K_QUANTIZER_H
#define J2K_QUANTIZER_H

#include "../j2k.h"

#ifdef __cplusplus
extern "C" {
#endif

// /**
//  * Calculate step size from parameters
//  * 
//  * @param bit_depth
//  * @param exponent
//  * @param mantisa
//  * @return float step size
//  */
// float
// quantizer_calculate_stepsize(int bit_depth, int exponent, int mantisa);

/**
 * Gets band gain according to its type.
 */
int
quantizer_get_band_log2_gain(enum j2k_band_type type);

// /**
//  * Setup implicit quantization
//  * 
//  * @param parameters  J2K encoder parameters
//  */
// void
// quantizer_setup_implicit(struct j2k_encoder_parameters* parameters);
// 
// /**
//  * Setup implicit quantization
//  * 
//  * @param parameters  J2K encoder parameters
//  */
// void
// quantizer_setup_explicit(struct j2k_encoder_parameters* parameters);

/** 
 * Setup bit depths and bitplane limits for lossless compression. 
 * Stepsizes are initialized too, not to have unitialized data.
 * 
 * @param encoder  instance of encoder with parameters set and structure ready
 */
void
quantizer_setup_lossless(struct j2k_encoder* encoder);

/** 
 * Setup bit depths, bitplane limits and quantization stepsizes for lossy 
 * compression with explicit quantization. Possibly also in GPU buffers.
 * 
 * @param encoder  instance of encoder with parameters set and structure ready
 * @param quality  quality parameter for stepsizes computation
 * @param subsampled  nonzero to set maximal coefficients for highest resolution
 */
void
quantizer_setup_lossy(struct j2k_encoder* encoder, float quality, int subsampled);

/** 
 * Perform quantization
 *
 * @param encoder  J2K encoder structure
 * @param stream  CUDA stream to run in
 * @return 0 if OK, nonzero otherwise
 */
int
quantizer_process(struct j2k_encoder* encoder, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // J2K_QUANTIZER_H
