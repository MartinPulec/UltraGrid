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

#include <stdlib.h>
#include <assert.h>

#include "../common.h"
#include "quantizer.h"
#include <cmath>

#define THREAD_BLOCK_SIZE 256

/**
 * Quantization
 *
 * References:
 * -----------
 * [1] http://www.jpeg.org/public/fcd15444-1.pdf, str. 105
 * [2] http://www.whydomath.org/node/wavlets/jpeg2000quantization.html
 * [3] JPEG2000 Image Compression Fundamentals, Standards, Practise (od Taubmana), str. 436
 * [4] http://encyclopedia.jrank.org/articles/pages/6781/JPEG-2000-Image-Coding-Standard.html
 */

/** Documented at declaration */
static float
compose_stepsize(int bit_depth, int exponent, int mantisa)
{
    assert(exponent >= 0x0000 && exponent <= 0x001F); // 5 bits
    assert(mantisa >= 0x0000 && mantisa <= 0x07FF); // 11 bits
    return (float)((1.0 + mantisa / 2048.0) * pow(2.0, bit_depth - exponent));
}


/**
 * Gets band gain according to its type.
 */
int
quantizer_get_band_log2_gain(enum j2k_band_type type)
{
    // (see figure E-2 in http://www.jpeg.org/public/fcd15444-1.pdf on page 106)
    int gain = 0;
    switch ( type ) {
        case LL: gain = 0; break;
        case HL:
        case LH: gain = 1; break;
        case HH: gain = 2; break;
    }
    return gain;
}


/** Decomposes given stepsize into mantisa and exponent */
static void
decompose_stepsize(const double stepsize,
                   int * exponent_out,
                   int * mantisa_out,
                   const int bit_depth)
{
    // decompose and encode mantisa
    const int exponent = (int)floor(log2(stepsize));
    *mantisa_out = (int)(2048.0 * (stepsize * pow(0.5, exponent) - 1.0));
    
    // encode exponent
    *exponent_out = bit_depth - exponent;
    
    // check bounds: max 5 bits for exponent
    const int MAX_EXP = 31;
    const int MAX_MAN = 2047;
    // TODO: check, whether to check overflow/underflow
    if(*exponent_out > MAX_EXP) {
        *exponent_out = MAX_EXP;
        *mantisa_out = MAX_MAN;
    } else if (*exponent_out < 0) {
        *exponent_out = 0;
        *mantisa_out = 0;
    } else if (*mantisa_out > MAX_MAN) { // exp. OK => max 11 bits for mantisa
        *mantisa_out = MAX_MAN;
    } else if (*mantisa_out < 0) {
        *mantisa_out = 0;
    }
}



// /** Documented at declaration */
// void
// quantizer_setup_implicit(struct j2k_encoder* encoder, float quality)
// {
//     // pointer to parametrers structure 
//     const struct j2k_encoder_parameters * const params = encoder->parameters;
//     
//     // set quantization type
//     encoder->quantization_mode = QM_IMPLICIT;
//     
//     // base quality (derived from number of bitplanes and user defined quality)
//     const double base_ssize = pow(0.5, quality * params->bit_depth);
//     
//     // base stepsize (en exponent and mantisa representation)
//     // set exponent and mantisa for LL band
//     decompose_stepsize(base_ssize, params->band_stepsize, params->bit_depth);
//     
//     // Set exponent and mantisa for other bands by formula
//     //      (e_b, u_b) = (e_o + nsd_b - nsd_o, u_o)
//     // see formula E.2 in http://www.jpeg.org/public/fcd15444-1.pdf on page 105
//     // set stepsizes for all other resolutions
//     const int band_count = params->resolution_count * 3 - 2;
//     for ( int band = 1; band < band_count; band++ ) {
//         // use base stepsize with adjusted exponent
//         params->band_stepsize[band] = params->band_stepsize[0];
//         params->band_stepsize[band].exponent -= ((band - 1) / 3);
// 
//         // possibly clamp to 0 if exponent below 0
//         if ( params->band_stepsize[band].exponent < 0 ) {
//             params->band_stepsize[band].exponent = 0;
//             params->band_stepsize[band].mantisa = 0;
//         }
//     }
// }


static double 
get_lowpass_weight(const int level) {
    // Weights of 1D lowpass transform outputs at different levels.
    // Copied from JasPer library.
    const int MAX_LEVELS = 21;
    static const double lowpass_weights[MAX_LEVELS] = 
    {
        1.0,
        1.4021081679297411,
        2.0303718560817923,
        2.9011625562785555,
        4.1152851751758002,
        5.8245108637728071,
        8.2387599345725171,
        11.6519546479210838,
        16.4785606470644375,
        23.3042776444606794,
        32.9572515613740435,
        46.6086013487782793,
        65.9145194076860861,
        93.2172084551803977,
        131.8290408510004283,
        186.4344176300625691,
        263.6580819564562148,
        372.8688353500955373,
        527.3161639447193920,
        745.7376707114038936,
        1054.6323278917823245
    };
    
    // clamp to max level count (approximations follow after that level)
    return lowpass_weights[(level < MAX_LEVELS) ? level : (MAX_LEVELS - 1)];
}


static double 
get_highpass_weight(const int level)
{
    // Weights of 1D highpass transform outputs at different levels.
    // Copied from JasPer library.
    const int MAX_LEVELS = 21;
    static const double highpass_weights[MAX_LEVELS] = 
    {
        1.0,
        1.4425227650161456,
        1.9669426082455688,
        2.8839248082788891,
        4.1475208393432981,
        5.8946497530677817,
        8.3471789178590949,
        11.8086046551047463,
        16.7012780415647804,
        23.6196657032246620,
        33.4034255108592362,
        47.2396388881632632,
        66.8069597416714061,
        94.4793162154500692,
        133.6139330736999113,
        188.9586372358249378,
        267.2278678461869390,
        377.9172750722391356,
        534.4557359047058753,
        755.8345502191498326,
        1068.9114718353569060
    };
    
    // clamp to max level count (approximations follow after that level)
    return highpass_weights[(level < MAX_LEVELS) ? level : (MAX_LEVELS - 1)];
}

/** Documented at declaration */
void
quantizer_setup_lossless(struct j2k_encoder* encoder)
{
    // frequently used values
    const int input_bit_depth = encoder->params.bit_depth;
    const int guard_bits = encoder->params.guard_bits;
    
    // Prepare bit_depth and clear stepsize for all bands in all components
    for ( int comp_idx = encoder->params.comp_count; comp_idx--; ) {
        // pointer to current component
        const struct j2k_component* const comp_ptr = encoder->component + comp_idx;
        
        // for all component's resolutions:
        for ( int res_idx = encoder->params.resolution_count; res_idx--; ) {
            // pointer to current  resolution
            const struct j2k_resolution* const res_ptr = encoder->resolution + comp_ptr->resolution_index + res_idx;
            
            // for all reslution's bands:
            for ( int bnd = 0; bnd < res_ptr->band_count; bnd++ ) {
                // pointer to the band
                struct j2k_band* const band_ptr = encoder->band + res_ptr->band_index + bnd;
                
                // Prepare bit depth gain 
                const int gain = quantizer_get_band_log2_gain(band_ptr->type);

                // Set bit depth according to formula [M_b = G + e_b - 1] 
                // (see formula E.3 in http://www.jpeg.org/public/fcd15444-1.pdf)
                band_ptr->bit_depth = input_bit_depth + gain + guard_bits - 1;

                // For completeness, set stepsize to 1.0 (but it isn't used anywhere)
                band_ptr->stepsize = 1.0f;
                band_ptr->stepsize_exponent = 0;
                band_ptr->stepsize_mantisa = 0;
            }
        }
    }
}


/** Documented at declaration */
void
quantizer_setup_lossy(struct j2k_encoder* encoder, float quality)
{
    // set quantization type
    encoder->quantization_mode = QM_EXPLICIT;
    
    // pointer to parametrers structure 
    const struct j2k_encoder_params * const params = &encoder->params;
    
    // Prepare bit_depth and clear stepsize for all bands in all components
    for ( int comp_idx = params->comp_count; comp_idx--; ) {
        // pointer to current component
        const struct j2k_component* const comp_ptr = encoder->component + comp_idx;
    
        // TODO: try to increase base stepsize for chroma components when using MCT!
        // base quality for the component (derived from number of bitplanes 
        // and user defined quality)
        const double base_stepsize = pow(2.0, (1.0 - quality) * params->bit_depth);
        
        // for all component's resolutions:
        for ( int res_idx = params->resolution_count; res_idx--; ) {
            // pointer to current  resolution
            const struct j2k_resolution* const res_ptr = encoder->resolution + comp_ptr->resolution_index + res_idx;

            // number of DWT levels needed to get this resolution
            const int dwt_level_count = params->resolution_count - max(1, res_idx);
            
            // weights of filter outputs at this resolution
            const double l_weight = get_lowpass_weight(dwt_level_count);
            const double h_weight = get_highpass_weight(dwt_level_count);
            
            // for all resolution's bands:
            for ( int bnd = 0; bnd < res_ptr->band_count; bnd++ ) {
                // pointer to the band
                struct j2k_band* const band_ptr = encoder->band + res_ptr->band_index + bnd;
                
                // combined weight of filter outputs
                double weight = 0.0;
                float gain = 0.0f;
                switch(band_ptr->type) {
                    case LL: weight = l_weight * l_weight; gain = 1.0f; break;
                    case HL: // same as LH
                    case LH: weight = h_weight * l_weight; gain = 2.0f; break;
                    case HH: weight = h_weight * h_weight; gain = 4.0f; break;
                }
                
                // decompose the stepsize into mantisa and exponent
                decompose_stepsize(
                    base_stepsize / weight,
                    &band_ptr->stepsize_exponent,
                    &band_ptr->stepsize_mantisa,
                    params->bit_depth
                );
                
                // reproduce precision loss by recomposing the stepsize
                // and add gain
                band_ptr->stepsize = gain * compose_stepsize(
                    params->bit_depth,
                    band_ptr->stepsize_exponent,
                    band_ptr->stepsize_mantisa
                );
                
                // Set bit depth according to formula [M_b = G + e_b - 1] 
                // (formula E.3 in http://www.jpeg.org/public/fcd15444-1.pdf)
                band_ptr->bit_depth = band_ptr->stepsize_exponent
                                    + params->guard_bits - 1;
            }
        }
    }
}

/**
 * Kernel - Quantization
 *
 * @param d_data_input  Input buffer
 * @param d_data_output  Output buffer
 * @param data_size  Data buffer size
 * @return void
 */
template<class data_type>
__global__ void
quantizer_kernel(data_type* d_data_input, int* d_data_output, int data_size, float stepsize)
{
    int index  = blockDim.x * blockIdx.x + threadIdx.x;
    if ( index < data_size ) {
        // Perform quantization
        data_type value = d_data_input[index] / stepsize;
        
        // Round value
        d_data_output[index] = static_cast<int>(value);
    }
}

/**
 * Quantization
 *
 * @param d_data_input  Input buffer
 * @param d_data_output  Output buffer
 * @param data_size  Data buffer size
 * @param stepsize  Stepsize for the data
 * @param stream  CUDA stream to run in
 */
template<class data_type>
inline void
quantizer_perform(data_type* d_data_input, int* d_data_output, int data_size, float stepsize, cudaStream_t stream)
{
    dim3 block(THREAD_BLOCK_SIZE);
    dim3 grid(DIVANDRND(data_size, THREAD_BLOCK_SIZE));

    quantizer_kernel<<<grid, block, 0, stream>>>(d_data_input, d_data_output, data_size, stepsize);
//     cudaCheckAsyncError("Quantization kernel");
}

/** Documented at declaration */
int
quantizer_process(struct j2k_encoder* encoder, cudaStream_t stream)
{
    // Perform quantization for each band in each resolution in each component
    for ( int comp = 0; comp < encoder->params.comp_count; comp++ ) {
        struct j2k_component* component = &encoder->component[comp];
        for ( int res = 0; res < encoder->params.resolution_count; res++ ) {
            struct j2k_resolution* resolution = &encoder->resolution[component->resolution_index + res];
            for ( int bnd = 0; bnd < resolution->band_count; bnd++ ) {
                // pointer to band structure
                struct j2k_band* band = &encoder->band[resolution->band_index + bnd];
                
                // pixel count
                const int pixel_count = band->size.width * band->size.height;
                
                // Perform proper quantization
                if ( encoder->params.compression == CM_LOSSLESS )
                    quantizer_perform(&((int*)encoder->d_data)[band->data_index], &encoder->d_data_quantizer[band->data_index], pixel_count, band->stepsize, stream);
                else if ( encoder->params.compression == CM_LOSSY_FLOAT )
                    quantizer_perform(&((float*)encoder->d_data)[band->data_index], &encoder->d_data_quantizer[band->data_index], pixel_count, band->stepsize, stream);
                else
                    assert(0);
            }
        }
    }
    return 0;
}



