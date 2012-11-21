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
 
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "j2k.h"


#include "compaction/compaction.h"
#include "common.h"
#include "j2k_encoder_init.h"
#include "preprocessor/preprocessor.h"
#include "preprocessor/preprocessor_fmt.h"
#include "dwt/gpu/dwt.h"
#include "quantizer/quantizer.h"
#include "ebcot/mqc/mqc.h"
#include "ebcot/cxmod/gpu/cxmod_interface.h"
#include "t2/t2.h"
#include "rate_control/rate_control.h"
#include "j2k_timer.h"


/** Documented at declaration */
void
j2k_encoder_params_set_default(struct j2k_encoder_params* parameters)
{
    parameters->compression = CM_LOSSLESS;
    parameters->size.width = 0;
    parameters->size.height = 0;
    parameters->bit_depth = 8;
    parameters->is_signed = 0;
    parameters->comp_count = 3;
    parameters->resolution_count = 3;
    parameters->mct = 0;
    
    // Code-block defaults
    parameters->cblk_size.width = 32;
    parameters->cblk_size.height = 32;
//     parameters->cblk_style = 0;
    
    // Precincts sizes and ordering
    parameters->progression_order = PO_LRCP;
    parameters->precinct_size[0].width = 128;
    parameters->precinct_size[0].height = 128;
    for ( int resolution = 1; resolution < J2K_MAX_RESOLUTION_COUNT; resolution++ ) {
        parameters->precinct_size[resolution].width = 256;
        parameters->precinct_size[resolution].height = 256;
    }
    
    // Quantization
    parameters->guard_bits = 2;
    parameters->quality_limit = 1.0f;
//     parameters->quantization = QM_EXPLICIT;
//     parameters->guard_bits = 2;
//     parameters->quantizer_precision = 1.0f;
//     for ( int band = 0; band < J2K_MAX_BAND_COUNT; band++ ) {
//         parameters->band_stepsize[band].exponent = 0;
//         parameters->band_stepsize[band].mantisa = 0;
//     }
    
    // error resilience: default is to use both start and end of packet markers
    parameters->use_sop = 1;
    parameters->use_eph = 1;
    
    // quiet operation by default
    parameters->print_info = 0;
    
    // no codestream restrictions by default (capabilities = 0)
    parameters->capabilities = J2K_CAP_DEFAULT;
    parameters->out_bit_depth = -1;
}


/** Documented at declaration */
void
j2k_image_params_set_default(struct j2k_image_params * params)
{
    params->output_byte_count = 0;
    params->quality = 1.0f;
}



/**
 * Sets maixmal output codestream size in bytes for all following encoded frames.
 * @param encoder  pointer to initialized instance of the encoder
 * @param max_byte_count  maximal number of final codestream bytes (0 == unlimited)
 */
void
j2k_encoder_set_byte_count_limit(struct j2k_encoder* encoder, size_t max_byte_count) {
    if(max_byte_count) {
        // subtract estimated headers size if not 0 
        const size_t t2_overhead = j2k_t2_get_overhead(encoder, max_byte_count);
        
        // make sure that the result is still greater than 0
        if(t2_overhead < max_byte_count) {
            max_byte_count -= t2_overhead;
        } else {
            max_byte_count = 1;
        }
    }
    
    encoder->max_byte_count = max_byte_count;
    
    // TODO: set limits for individual components if DCI profile is used
}


/**
 * Gets nonzero if given number is NOT an integral power of two
 */
static int
not_a_pow_of_2(const int n) {
    return n & (n - 1);
}


/**
 * Checks given parameters, possibly correcting them.
 * Returns nonzero if there were any unrecoverable errors among parameters.
 */
static int
j2k_encoder_check_parameters(struct j2k_encoder_params * const params)
{
    // nonzero if error occured
    int error = 0;
    
    // general checks
    if(params->compression != CM_LOSSLESS && params->compression != CM_LOSSY_FLOAT) {
        printf("J2K ERROR: compression type must be either CM_LOSSLESS or CM_LOSSY_FLOAT.\n");
        error = 1;
    }
    if(params->is_signed & ~1) {  // params->is_signed is not 0 nor 1
        printf("J2K ERROR: is_signed must be either 0 or 1 - not %d.\n", params->is_signed);
        error = 1;
    }
    if(params->comp_count > J2K_MAX_COMP_COUNT) {
        printf("J2K ERROR: component count limit (%d) exceeded: %d.\n", J2K_MAX_COMP_COUNT, params->comp_count);
        error = 1;
    }
    if(params->resolution_count > J2K_MAX_RESOLUTION_COUNT) {
        printf("J2K ERROR: resolution count limit (%d) exceeded: %d.\n", J2K_MAX_RESOLUTION_COUNT, params->resolution_count);
        error = 1;
    }
    if(params->size.width <= 0) {
        printf("J2K ERROR: invalid image width: %d.\n", params->size.width);
        error = 1;
    }
    if(params->size.height <= 0) {
        printf("J2K ERROR: invalid image height: %d.\n", params->size.height);
        error = 1;
    }
    if(params->bit_depth <= 0) {
        printf("J2K ERROR: invalid bit depth: %d.\n", params->bit_depth);
        error = 1;
    }
    if(params->out_bit_depth == 0 || params->out_bit_depth < -1) {
        printf("J2K ERROR: invalid output bit depth: %d.\n", params->out_bit_depth);
        error = 1;
    }
    if(params->comp_count < 1) {
        printf("J2K ERROR: invalid component count: %d.\n", params->comp_count);
        error = 1;
    }
    if(params->resolution_count < 1) {
        printf("J2K ERROR: invalid resolution count: %d.\n", params->resolution_count);
        error = 1;
    }
    if(params->comp_count < 3 && params->mct) {
        printf("J2K ERROR: too few components (%d) for MCT.\n", params->comp_count);
        error = 1;
    }
    if(not_a_pow_of_2(params->cblk_size.width)) {
        printf("J2K ERROR: cblk width %d is not a power of 2.\n", params->cblk_size.width);
        error = 1;
    }
    if(not_a_pow_of_2(params->cblk_size.height)) {
        printf("J2K ERROR: cblk height %d is not a power of 2.\n", params->cblk_size.height);
        error = 1;
    }
    if(params->cblk_size.height * params->cblk_size.width > 4096) {
        printf("J2K ERROR: cblk size %dx%d too big.\n", params->cblk_size.width, params->cblk_size.height);
        error = 1;
    }
    for(int res_idx = params->resolution_count; res_idx--; ) {
        const struct j2k_size * const prec_size = params->precinct_size + res_idx;
        if(not_a_pow_of_2(prec_size->width)) {
            printf("J2K ERROR: prec width %d (res #%d) is not a power of 2.\n",
                   prec_size->width, res_idx);
            error = 1;
        }
        if(not_a_pow_of_2(prec_size->height)) {
            printf("J2K ERROR: prec height %d (res #%d) is not a power of 2.\n",
                   prec_size->height, res_idx);
            error = 1;
        }
    }
    if(params->guard_bits < 0 || params->guard_bits > 7) {
        printf("J2K ERROR: invalid guard bit count: %d.\n", params->guard_bits);
        error = 1;
    }
    if(params->compression == CM_LOSSY_FLOAT && params->quality_limit <= 0.0) {
        printf("J2K ERROR: invalid quality limit: %f.\n", params->quality_limit);
        error = 1;
    }
    
    // DCI specific checks
    if(params->capabilities == J2K_CAP_DCI_4K
            || params->capabilities == J2K_CAP_DCI_2K_24
            || params->capabilities == J2K_CAP_DCI_2K_48) {
        if(params->cblk_size.width != 32) {
            printf("J2K WARNING: cblk width set to 32 from %d for DCI.\n", params->cblk_size.width);
            params->cblk_size.width = 32;
        }
        if(params->cblk_size.height != 32) {
            printf("J2K WARNING: cblk height set to 32 from %d for DCI.\n", params->cblk_size.height);
            params->cblk_size.height = 32;
        }
        if(128 != params->precinct_size->width) {
            printf("J2K WARNING: prec width %d (res #0) set to 128 for DCI.\n",
                   params->precinct_size->width);
            params->precinct_size->width = 128;
        }
        if(128 != params->precinct_size->height) {
            printf("J2K WARNING: prec height %d (res #0) set to 128 for DCI.\n",
                   params->precinct_size->height);
            params->precinct_size->height = 128;
        }
        for(int res_idx = 1; res_idx < params->resolution_count; res_idx++) {
            struct j2k_size * const prec_size = params->precinct_size + res_idx;
            if(256 != prec_size->width) {
                printf("J2K WARNING: prec width %d (res #%d) set to 256 for DCI.\n",
                       prec_size->width, res_idx);
                prec_size->width = 256;
            }
            if(256 != prec_size->height) {
                printf("J2K WARNING: prec height %d (res #%d) set to 256 for DCI.\n",
                       prec_size->height, res_idx);
                prec_size->height = 256;
            }
        }
//         if(0 != params->cblk_style) {
//             printf("J2K WARNING: cblk coding style set to 0 from %d for DCI.\n", params->cblk_style);
//             params->cblk_style = 0;
//         }
        if(params->use_eph) {
            printf("J2K WARNING: EPH turned off for DCI.\n");
            params->use_eph = 0;
        }
        if(params->use_sop) {
            printf("J2K WARNING: SOP turned off for DCI.\n");
            params->use_sop = 0;
        }
        
        // 2K DCI specific
        if(params->capabilities == J2K_CAP_DCI_2K_24
                || params->capabilities == J2K_CAP_DCI_2K_48) {
            if(params->resolution_count > 6) {
                printf("J2K WARNING: res count %d set to 6 for 2K DCI.\n", params->resolution_count);
                params->resolution_count = 6;
            }
            if(params->size.width > 2048) {
                printf("J2K ERROR: width %d too big for DCI 2K.\n", params->size.width);
                error = 1;
            }
            if(params->size.height > 1080) {
                printf("J2K ERROR: height %d too big for DCI 2K.\n", params->size.height);
                error = 1;
            }
        }
        
        // 4K DCI specific
        if(params->capabilities == J2K_CAP_DCI_4K) {
            if(params->resolution_count > 7) {
                printf("J2K WARNING: res count %d set to 7 for 4K DCI.\n", params->resolution_count);
                params->resolution_count = 7;
            }
            if(params->resolution_count < 2) {
                printf("J2K WARNING: res count %d set to 2 for 4K DCI.\n", params->resolution_count);
                params->resolution_count = 2;
            }
            if(params->size.width > 4096) {
                printf("J2K ERROR: width %d too big for DCI 4K.\n", params->size.width);
                error = 1;
            }
            if(params->size.height > 2160) {
                printf("J2K ERROR: height %d too big for DCI 4K.\n", params->size.height);
                error = 1;
            }
        }
    }
    
    // possibly indicate error
    return error;
}



/**
 * Initializes empty encoder structure. The structure is assumed to be 
 * filled with zeros except of parameters (which are assumed to be valid).
 * @param enc  zeroed encoder structure pointer with valid parameters set
 * @return 0 for success, nonzero for error
 */
static int 
j2k_encoder_init(struct j2k_encoder * encoder)
{
    // Direct pointer to parameters structure
    const struct j2k_encoder_params * const params = &encoder->params;
    
    // Get device into
    int device_id;
    if(cudaSuccess != cudaGetDevice(&device_id)) {
        printf("Cannot get current device info.\n");
        return -1;
    }
    if(cudaSuccess != cudaGetDeviceProperties(&encoder->gpu_info, device_id)) {
        printf("Error getting info about device #%d.\n", device_id);
        return -2;
    }
    
    // Setup explicit quantization for lossy compression
    encoder->quantization_mode = params->compression == CM_LOSSY_FLOAT
            ? QM_EXPLICIT : QM_NONE;
        
    // Initialize structure, allocated data buffers, etc.
    if(j2k_encoder_init_buffer(encoder))
        return -3;
    
    // Select preprocessor function
    const enum j2k_component_transform transform = params->mct
            ? params->compression == CM_LOSSLESS ? CT_REVERSIBLE : CT_IRREVERSIBLE
            : CT_NONE;
    encoder->preprocessor = preprocessor_get_function(
        params->bit_depth,
        params->is_signed,
        params->comp_count,
        params->compression,
        transform
    );
    if(0 == encoder->preprocessor) {
        return -4;
    }
    
    // Create context-modeller
    if(0 == (encoder->cxmod = cxmod_create(params))) {
        return -5;
    }
    
    // Create MQ-Coder
    if(0 == (encoder->mqc = mqc_create(params))) {
        return -6;
    }
    
    // Create T2 encoder
    if(0 == (encoder->t2 = j2k_t2_create(encoder))) {
        return -7;
    }
    
    // try to initialize preprocessor for fixed formats
    if(0 == (encoder->fmt_preprocessor = j2k_fmt_preprocessor_create())) {
        return -8;
    }
    
    // initialize maximal byte count for rate control (to value "NO LIMIT")
    encoder->max_byte_count = 0;
    
    // initialize rate control
    if(j2k_rate_control_init())
        return -9;
    
    // initialize callback state machine variables
    encoder->need_input = 0;
    encoder->have_output = 0;
    
    // initialize events and streams for encoding pipelining
    for(int i = 3; i--; ) {
        encoder->pipeline[i].active = 0;
        if(cudaSuccess != cudaEventCreate(&encoder->pipeline[i].event)) {
            return -10;
        }
        if(cudaSuccess != cudaStreamCreate(&encoder->pipeline[i].stream)) {
            return -11;
        }
    }
    
    // initialize output compaction
    if(j2k_compaction_init()) {
        return -12;
    }
    
    // allocate timers if required
    if(params->print_info) {
        encoder->timer_h_to_d = j2k_gpu_timer_create();
        encoder->timer_preproc = j2k_gpu_timer_create();
        encoder->timer_dwt = j2k_gpu_timer_create();
        encoder->timer_quant = j2k_gpu_timer_create();
        encoder->timer_cxmod = j2k_gpu_timer_create();
        encoder->timer_mqc = j2k_gpu_timer_create();
        encoder->timer_rate = j2k_gpu_timer_create();
        encoder->timer_compact = j2k_gpu_timer_create();
        encoder->timer_d_to_h = j2k_gpu_timer_create();
        encoder->timer_t2 = j2k_cpu_timer_create();
        encoder->timer_run = j2k_cpu_timer_create();
    }
    
    // No errors => incidcate success.
    return 0;
}


/** Documented at declaration */
struct j2k_encoder*
j2k_encoder_create(const struct j2k_encoder_params* params)
{
    // make sure that parameters pointer is not null
    if(0 == params) {
        return 0;
    }
    
    // Create J2K encoder structure and clear it
    struct j2k_encoder* enc = (struct j2k_encoder*)malloc(sizeof(struct j2k_encoder));
    if(0 == enc) {
        return 0;
    }
    memset(enc, 0, sizeof(struct j2k_encoder));
    
    // Copy original parameters into the structure
    enc->params = *params;
    
    // Check parameters and try to initialize the instance
    if(j2k_encoder_check_parameters(&enc->params) || j2k_encoder_init(enc)) {
        // Error occured => destroy partially initialized instance.
        j2k_encoder_destroy(enc);
        return 0;
    }
    
    // success => return the instance
    return enc;
}


/** Documented at declaration */
void
j2k_encoder_destroy(struct j2k_encoder* encoder)
{    
    if(encoder) {
        j2k_encoder_free_buffer(encoder);
        if(encoder->mqc)
            mqc_destroy(encoder->mqc);
        if(encoder->cxmod)
            cxmod_destroy(encoder->cxmod);
        if(encoder->t2)
            j2k_t2_destroy(encoder->t2);
        if(encoder->fmt_preprocessor)
            j2k_fmt_preprocessor_destroy(encoder->fmt_preprocessor);
        for(int i = 3; i--; ) {
            cudaEventDestroy(encoder->pipeline[i].event);
            cudaStreamDestroy(encoder->pipeline[i].stream);
        }
        /* timer destructors handle NULLs safely */
        j2k_gpu_timer_destroy(encoder->timer_h_to_d);
        j2k_gpu_timer_destroy(encoder->timer_preproc);
        j2k_gpu_timer_destroy(encoder->timer_dwt);
        j2k_gpu_timer_destroy(encoder->timer_quant);
        j2k_gpu_timer_destroy(encoder->timer_cxmod);
        j2k_gpu_timer_destroy(encoder->timer_mqc);
        j2k_gpu_timer_destroy(encoder->timer_rate);
        j2k_gpu_timer_destroy(encoder->timer_compact);
        j2k_gpu_timer_destroy(encoder->timer_d_to_h);
        j2k_cpu_timer_destroy(encoder->timer_t2);
        j2k_cpu_timer_destroy(encoder->timer_run);
        free(encoder);
    }
}

/**
 * Perform preprocessing
 * 
 * @param encoder  J2K encoder structure
 * @param image  J2K image structure
 */
static int
j2k_preprocess(struct j2k_encoder* encoder, cudaStream_t stream, enum j2k_input_format fmt)
{    
    assert(encoder->d_source != NULL);
    assert(encoder->default_source_size > 0);
    assert(encoder->preprocessor != NULL);
    assert(encoder->d_data_preprocessor != NULL);
    
    // Fill array of color component device data buffers
    void* d_comp_data[J2K_MAX_COMP_COUNT];
    for ( int comp = 0; comp < encoder->params.comp_count; comp++ ) {
        // Assign d_data_preprocessor[data_index] to d_comp_data[comp]
        d_comp_data[comp] = (char*)encoder->d_data_preprocessor 
                          + encoder->component[comp].data_index * 4; // 4 = size of int or float
    }
    
    // Use either default preprocessor or the one for special formats.
    if(fmt == J2K_FMT_DEFAULT) {
        if(preprocessor_process(
            encoder->preprocessor,
            d_comp_data, 
            encoder->params.comp_count,
            encoder->d_source, 
            encoder->params.size,
            stream
        )) return -1;
    } else {
        if(j2k_fmt_preprocessor_preprocess(
            encoder->fmt_preprocessor,
            fmt,
            encoder->params.size,
            encoder->params.comp_count,
            encoder->params.bit_depth,
            encoder->params.is_signed,
            encoder->params.compression,
            encoder->d_source,
            d_comp_data,
            0,
            encoder->params.mct,
            stream
        )) return -2;
    }
    
    // Set current data buffer
    encoder->d_data = encoder->d_data_preprocessor;
    encoder->data_size = encoder->data_preprocessor_size;
    
    return 0;
}

/**
 * Perform discrete wavelet tranform
 * 
 * @param encoder  J2K encoder structure
 * @param stream  CUDA stream to run in
 */
int
j2k_encoder_dwt(struct j2k_encoder* encoder, cudaStream_t stream)
{    
    // If DWT level is greater than zero, perform proper DWT
    if ( encoder->params.resolution_count > 1 ) {
        for ( int comp = 0; comp < encoder->params.comp_count; comp++ ) {
            struct j2k_component* component = &encoder->component[comp];
        
            // For lossless compression use 5/3 DWT
            if ( encoder->params.compression == CM_LOSSLESS ) {
                dwt_forward_53(
                    &((int*)encoder->d_data)[component->data_index],
                    &((int*)encoder->d_data_dwt)[component->data_index],
                    encoder->params.size.width,
                    encoder->params.size.height,
                    encoder->params.resolution_count - 1,
                    (void*)&stream
                );
            }
            // For floating point lossy compression use 9/7 DWT
            else if ( encoder->params.compression == CM_LOSSY_FLOAT ) {
                dwt_forward_97(
                    &((float*)encoder->d_data)[component->data_index],
                    &((float*)encoder->d_data_dwt)[component->data_index],
                    encoder->params.size.width,
                    encoder->params.size.height,
                    encoder->params.resolution_count - 1,
                    (void*)&stream
                );
            } else
                assert(0);
        }
    }
    // If DWT level is zero only copy data (it will be one LL band)
    else {
        // Copy data as one LL band (at once for all components)
        assert(encoder->data_dwt_size == encoder->data_preprocessor_size);
        assert(encoder->data_preprocessor_size > 0);
        dwt_cuda_copy((void*)encoder->d_data_dwt, (void*)encoder->d_data_preprocessor, encoder->data_preprocessor_size, (void*)&stream);
    }
    
    // Set current data buffer
    encoder->d_data = encoder->d_data_dwt;
    encoder->data_size = encoder->data_dwt_size;
    
    return 0;
}

/**
 * Perform quantization or ranging
 * 
 * @param encoder  J2K encoder structure
 * @param stream  CUDA stream to run in
 */
int
j2k_encoder_quantize(struct j2k_encoder* encoder, cudaStream_t stream)
{
    // Perform quantization
    if ( encoder->quantization_mode == QM_IMPLICIT || encoder->quantization_mode == QM_EXPLICIT )
        quantizer_process(encoder, stream);

    // Set current data buffer
    encoder->d_data = encoder->d_data_quantizer;
    encoder->data_size = encoder->data_quantizer_size;
    
    return 0;
}

// /**
//  * Perform embedded block coding
//  * 
//  * @param encoder  J2K encoder structure
//  * @param image  J2K image structure
//  */
// int
// j2k_encoder_ebcot(struct j2k_encoder* encoder)
// {
//     // Process Tier-1
//     if ( j2k_t1_encode(encoder) != 0 )
//         return -1;
//         
//     return 0;
// }


/**
 * Prints time.
 * @param what  time description
 * @param time  time in milliseconds
 */
static void
print_time(const char * const what, const double time) {
    printf("%20s time: %.4f ms\n", what, time);
}



/** 
 * Prints timer of non-NULL GPU timer together with some message.
 * NULL timers are safely ignored.
 * @param what  timer description
 * @param timer  pointer to GPU timer or NULL
 */
static void
print_timer(const char * const what, struct j2k_gpu_timer * const timer) {
    if(timer) {
        print_time(what, j2k_gpu_timer_time_ms(timer));
    }
}



/**
 * Sets encoding quality. (Up to limit specified at creation time.)
 * Ignored for lossless encoding. Lower quality means smaller output PSNR,
 * smaller output size and smaller compression time.
 * 
 * @param encoder  encoder instance
 * @param quality  0.2 = poor quality, 1.0 = full quality, 1.2 = extra quality
 */
static void
j2k_encoder_set_quality(struct j2k_encoder* encoder, float quality)
{
    if(encoder->params.compression == CM_LOSSY_FLOAT) {
        const float limited_quality = quality < encoder->params.quality_limit
                ? quality : encoder->params.quality_limit;
        quantizer_setup_lossy(encoder, limited_quality);
    }
}



/**
 * Asynchronous host to device memcpy wrapper with error checking.
 * @param dest    destination pointer
 * @param src     source pointer
 * @param size    byte count
 * @param stream  stream to run in
 * @return 0 if OK, nonzero for error
 */
static int
host_to_dev(void * dest, const void * src, size_t size, cudaStream_t str) {
    return cudaSuccess == cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, str) ? 0 : 1;
}


/**
 * Asynchronous device to host memcpy wrapper with error checking.
 * @param dest    destination pointer
 * @param src     source pointer
 * @param size    byte count
 * @param stream  stream to run in
 * @return 0 if OK, nonzero for error
 */
static int
dev_to_host(void * dest, const void * src, size_t size, cudaStream_t str) {
    return cudaSuccess == cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, str) ? 0 : 1;
}





/** 
 * Encoder interface for image series encoding. Calls input and output 
 * callbacks in single run, interleaving GPU computation with data transfers 
 * for optimal GPU utilization. By calling input callbacks, it retrieves new
 * input images from caller and using output callback, it provides encoded 
 * codestreams to caller. If input callback signalizes it to stop, it encoded 
 * remaining images, calls correpsonding output callbacks and returns back 
 * to caller. Must be called in the same thread, where the encoder instance was 
 * created (because the CUDA context is boud to the thread). All callbacks are 
 * called in the same thread as this function. Caller provides pointer, whose 
 * value is later passed to all callbacks. Aditionally, caller provides
 * another pointer for each input image, which is passed to corresponding
 * output callback (and therefore can represent some image-specific data).
 *
 * Same encoder instance can be used for multiple j2k_encoder_run and/or 
 * j2k_encoder_compress function calls as long as calls do not overlap
 * (can be called again after previous call returns).
 * 
 * @param encoder  pointer to encoder instance created in same thread
 * @param user_callback_data  some pointer passed to all callbacks in this run
 * @param in_callback  pointer to implementation of callback which provides 
 *                     input images from caller
 * @param out_callback  pointer to implementation of callback which retrieves
 *                      output codestreams from encoder
 * @param buffer_callback  pointer to implementation of callback which reports
 *                         no-longer-needed input buffers or NULL if such 
 *                         callback is not needed by caller
 * @return 0 after all submitted images were encoded or nonzero error code.
 */
int
j2k_encoder_run(struct j2k_encoder * encoder,
                void * user_callback_data,
                j2k_encoder_in_callback in_callback,
                j2k_encoder_out_callback out_callback,
                j2k_encoder_buffer_callback buffer_callback
) {
    // check arguments
    if(0 == encoder || 0 == in_callback || 0 == out_callback) {
        return 0;
    }
    
    // start 'run' timer
    j2k_cpu_timer_start(encoder->timer_run);
    
    // contains nonzero as long as input callback returns nenozero (= continue)
    int should_load = 1;
    
    // three stages of encoding pipeline (rotated at the end of each iteration)
    struct j2k_pipeline_stream * in_str = encoder->pipeline + 0;
    struct j2k_pipeline_stream * enc_str = encoder->pipeline + 1;
    struct j2k_pipeline_stream * out_str = encoder->pipeline + 2;
    
    // set all streams as empty
    in_str->active = 0;
    enc_str->active = 0;
    out_str->active = 0;
    
    // encode all input images
    do {
        // set buffer pointers for current encoding stage
        encoder->d_band = enc_str->d_band;
        encoder->d_cblk = enc_str->d_cblk;
        
        // run preprocessor on encoding stage (if it has image loaded)
        if(enc_str->active) {
            j2k_gpu_timer_start(encoder->timer_preproc, enc_str->stream);
            encoder->error |= j2k_preprocess(encoder, enc_str->stream, enc_str->format);
            j2k_gpu_timer_stop(encoder->timer_preproc, enc_str->stream);
        }
        
        // record event in encoding stage to sync with preprocessor end
        if(cudaSuccess != cudaEventRecord(enc_str->event, enc_str->stream)) {
            encoder->error = -123;
        }
        
        // issue more kernels for encoding stage (if image is loaded)
        if(enc_str->active) {
            // DWT
            j2k_gpu_timer_start(encoder->timer_dwt, enc_str->stream);
            encoder->error |= j2k_encoder_dwt(encoder, enc_str->stream);
            j2k_gpu_timer_stop(encoder->timer_dwt, enc_str->stream);
            
            // Quantization
            if(encoder->quantization_mode != QM_NONE) {
                j2k_gpu_timer_start(encoder->timer_quant, enc_str->stream);
                encoder->error |= j2k_encoder_quantize(encoder, enc_str->stream);
                j2k_gpu_timer_stop(encoder->timer_quant, enc_str->stream);
            }
            
            // Context modeller
            j2k_gpu_timer_start(encoder->timer_cxmod, enc_str->stream);
            encoder->error |= cxmod_encode(
                encoder->cxmod,
                encoder->cblk_count,
                encoder->d_cblk,
                encoder->band,
                (int*)encoder->d_data,
                encoder->d_cxd,
                (void*)&enc_str->stream
            );
            j2k_gpu_timer_stop(encoder->timer_cxmod, enc_str->stream);
            
            // MQ Coder
            j2k_gpu_timer_start(encoder->timer_mqc, enc_str->stream);
            encoder->error |= mqc_encode(
                encoder->mqc, 
                encoder->cblk_count, 
                encoder->d_cblk, 
                encoder->d_cxd, 
                encoder->d_byte,
                encoder->d_trunc_sizes,
                enc_str->stream
            );
            j2k_gpu_timer_stop(encoder->timer_mqc, enc_str->stream);
            
            // Setup rate control
            j2k_encoder_set_byte_count_limit(encoder, enc_str->image_params.output_byte_count);
            
            // Run rate control
            j2k_gpu_timer_start(encoder->timer_rate, enc_str->stream);
            encoder->error |= j2k_rate_control_reduce(encoder, enc_str->stream);
            j2k_gpu_timer_stop(encoder->timer_rate, enc_str->stream);
        }
        
        // possibly copy result from previous iteration into host buffer
        if(out_str->active) {
            j2k_gpu_timer_start(encoder->timer_d_to_h, out_str->stream);
            
            // copy size of compact output
            unsigned int out_size;
            encoder->error |= dev_to_host(
                &out_size,
                encoder->d_compact_size,
                sizeof(out_size),
                out_str->stream
            );
            
            // wait for the size to be copied
            if(cudaSuccess != cudaStreamSynchronize(out_str->stream)) {
                encoder->error = -124;
            }
            
            // copy fixed-size codeblock info structures into host buffer
            encoder->error |= dev_to_host(
                encoder->cblk,
                out_str->d_cblk,
                encoder->cblk_size,
                out_str->stream
            );
            
            // if no error occured, copy output back
            if(!encoder->error) {
                encoder->error |= dev_to_host(
                    encoder->c_byte_compact,
                    encoder->d_byte_compact,
                    out_size,
                    out_str->stream
                );
            }
            j2k_gpu_timer_stop(encoder->timer_d_to_h, out_str->stream);
        }
        
        // always record event at the end of output memcpy
        if(cudaSuccess != cudaEventRecord(out_str->event, out_str->stream)) {
            encoder->error = -12345;
        }
        
        // continue with encoding after the end of memcpy
        if(enc_str->active) {
            // let the encoding stream wait for output memcpy 
            // (not to overwrite output form previous iteration)
            if(cudaSuccess != cudaStreamWaitEvent(enc_str->stream, out_str->event, 0)) {
                encoder->error = -7;
            }
            
            // run output compaction
            j2k_gpu_timer_start(encoder->timer_compact, enc_str->stream);
            encoder->error |= j2k_compaction_run(encoder, enc_str->stream);
            j2k_gpu_timer_stop(encoder->timer_compact, enc_str->stream);
        }
        
        // possibly run input callback to get next input image, while GPU is coding
        if(should_load) {
            // wait for preprocessor to terminate before replacing 
            // input buffer contents with newly submitted input image
            if(cudaSuccess != cudaStreamWaitEvent(in_str->stream, enc_str->event, 0)) {
                encoder->error = -761;
            }
            
            // 1 if encoder has nothing else to do but waiting for new image
            const int should_block = !enc_str->active && !out_str->active;
            
            // reset loading state variables and load new input image
            encoder->need_input = 1;
            encoder->in_stream = in_str;
            should_load = in_callback(encoder, user_callback_data, should_block);
            encoder->need_input = 0;
        }
        
        // sync host with saving stream (waits for codestream memcpy)
        if(cudaSuccess != cudaStreamSynchronize(out_str->stream)) {
            encoder->error = -430;
        }

        // call output callback if no error occured (and if output stage is active)
        if(0 == encoder->error && out_str->active) {
            // print output memcpy time        
            if(out_str->active) {
                print_timer("Output memcpy", encoder->timer_d_to_h);
            }
            
            // Call T2 from output callback
            encoder->have_output = 1;
            out_callback(encoder, user_callback_data, out_str->user_input_data);
            encoder->have_output = 0;
        }
        
        
        // sync with all streams, printing timers
        cudaStreamSynchronize(in_str->stream);
        if(in_str->active) {
            // return the input buffer
            if(buffer_callback) {
                buffer_callback(encoder, user_callback_data, in_str->user_input_data, in_str->source_ptr);
            }
            
            // print input memcpy timer
            print_timer("Input memcpy", encoder->timer_h_to_d);
        }
        cudaStreamSynchronize(enc_str->stream);
        if(enc_str->active) {
            print_timer("Prepocessor", encoder->timer_preproc);
            print_timer("DWT", encoder->timer_dwt);
            if(encoder->quantization_mode != QM_NONE) {
                print_timer("Quantization", encoder->timer_quant);
            }
            print_timer("Context modeller", encoder->timer_cxmod);
            print_timer("MQ coder", encoder->timer_mqc);
            if(enc_str->image_params.output_byte_count) {
                print_timer("Rate allocation", encoder->timer_rate);
            }
            print_timer("Output compaction", encoder->timer_compact);
        }
            
        // saving stage is not active after providing encoded codestream
        out_str->active = 0;
        
        // rotate stages
        struct j2k_pipeline_stream * unused_stage = out_str;
        out_str = enc_str;
        enc_str = in_str;
        in_str = unused_stage;
    } while((should_load || enc_str->active || out_str->active)
            && 0 == encoder->error);
    
    // possibly call output callback for all stages, that are still active (in case of error)
    for(int i = 3; i--; ) {
        if(encoder->pipeline[i].active) {
            encoder->have_output = 1;
            out_callback(encoder, user_callback_data, encoder->pipeline[i].user_input_data);
            encoder->have_output = 0;
        }
    }
    
    // print run timer
    j2k_cpu_timer_stop(encoder->timer_run);
    if(encoder->timer_run) {
        print_time("Encoder run", j2k_cpu_timer_time_ms(encoder->timer_run));
    }
    
    return encoder->error;
}



/**
 * Can be used only in output callback to retrieve encoded image. Can be called 
 * multiple times in each callback (but always gets the same image). Encoded 
 * codestream is already written in the buffer when this function returns
 * (if the buffer is big enough.)
 * @param enc  pointer to encoder instance which has called the output callback
 * @param output_buffer_ptr  pointer to output buffer (need not be page locked)
 * @param output_buffer_size  size of output buffer in bytes
 * @param output_size_ptr  pointer to variable, where output byte count 
 *                         should be written (if successful)
 * @return 0 = OK, 1 = small buffer, negative error code = failure
 */
int
j2k_encoder_get_output(struct j2k_encoder * enc,
                       void * output_buffer_ptr,
                       size_t output_buffer_size,
                       size_t * output_size_ptr) {
    // check parameters
    if(0 == enc || 0 == output_buffer_ptr || 0 == output_size_ptr) {
        return -1;
    }
    
    // make sure that this was called within output callback
    if(0 == enc->have_output) {
        printf("J2K encoder API error: j2k_encoder_get_output not called "
               "within output callback of right encoder instance.\n");
        return -2;
    }
    
    // should not be called again in this callback
    enc->have_output = 0;
    
    // report error if encoder encountered error
    if(enc->error) {
        return -1;
    }
    
    // run T2 (surrounded by time measurement)
    j2k_cpu_timer_start(enc->timer_t2);
    const int t2_result = j2k_t2_encode(enc, enc->t2, (unsigned char *)output_buffer_ptr, output_buffer_size);
    j2k_cpu_timer_stop(enc->timer_t2);
    if(enc->timer_t2) {
        print_time("T2", j2k_cpu_timer_time_ms(enc->timer_t2));
    }
    
    // return correct error code
    if(t2_result > 0) {
        *output_size_ptr = t2_result;
        return 0;
    }
    return 1; // indicate failure
}





/**
 * Should be used only in the input callback to provide new input image data
 * in one of supported formats to encoder. Should NOT be used in single 
 * callback again, if previous call succeded (returned 0). No encoding is done
 * if this function is not called in some callback (but callback may be 
 * called again later - depending on return value of the callback).
 * 
 * Encoder initiates memcpy into device memory directly from provided buffer 
 * and returns immediately, so the buffer should NOT be written to until 
 * j2k_encoder_buffer_callback reports the buffer as unused. Note that memcpy
 * to device memory is much faster, if page-locked memory source buffer is used 
 * (see functions j2k_encoder_buffer_pagelock, j2k_encoder_buffer_pageunlock,
 * j2k_encoder_pagelocked_alloc and j2k_encoder_pagelocked_free for working 
 * with page-locked buffers).
 * 
 * Format of input samples and size of input data is determined by parameter 
 * 'format'. Implied format parameters (sample bit depth, component count, ...)
 * must match corresponding parameters passed to encoder contructor.
 * 
 * Arbitrary data pointer can be provided by this call, which is later passed 
 * to corresponding j2k_encoder_buffer_callback and j2k_encoder_out_callback.
 * 
 * @param enc  pointer to encoded instance which called the input callback
 * @param data  pointer to buffer with data in specified format
 * @param format  format of the input data
 * @param params  parameters for encoding of this input (or NULL for defaults)
 * @param user_input_data  some pointer passed to corresponding callbacks
 * @return 0 if OK, nonzero for error (e.g. if format parameters don't match 
 *         parameters passed to encoder constructor)
 */
int
j2k_encoder_set_input
(
    struct j2k_encoder * enc,
    const void * data,
    enum j2k_input_format format,
    const struct j2k_image_params * params,
    void * user_input_data
) {
    // check arguments
    if(0 == enc) {
        return -1;
    }
    if(0 == enc->need_input) {
        printf("J2K encoder API error: j2k_encoder_set_input not called "
               "within input callback of right encoder instance.\n");
        return -2;
    }
    if(0 == data) {
        return -3;
    }
    
    // get size of the input and check parameters (according to format)
    size_t source_size = enc->default_source_size;
    switch(format) {
        // default format => default parameters and default size
        case J2K_FMT_DEFAULT:
            break;
        
        // packed 10bit RGB (DPX) => check params and adjust size
        case J2K_FMT_R10_G10_B10_X2_L:
        case J2K_FMT_R10_G10_B10_X2_B:
            if(enc->params.bit_depth != 10) return -4;
            if(enc->params.comp_count != 3) return -5;
            if(enc->params.is_signed) return -6;
            source_size = (source_size / 3) * 2;
            break;
            
        // 16bit RGB (default size, but parameters must be checked)
        case J2K_FMT_R16_G16_B16_L:
        case J2K_FMT_R16_G16_B16_B:
            if(enc->params.bit_depth != 16) return -7;
            if(enc->params.comp_count != 3) return -8;
            if(enc->params.is_signed) return -9;
            break;
        
        // unknown format => indicate failure
        default:
            return -10;
    }
    
    // pointer to currently loading stage
    struct j2k_pipeline_stream * const loading_stage = enc->in_stream;
    
    // use either default parameters or explicit ones (if provided)
    if(params) {
        loading_stage->image_params = *params;
    } else {
        j2k_image_params_set_default(&loading_stage->image_params);
    }
    
    // remember the buffer pointer and format
    loading_stage->source_ptr = data;
    loading_stage->format = format;
    
    // replace input buffer contents with newly submitted image after the end 
    // of preprocessor (stream is already synced with the end of preprocessor)
    j2k_gpu_timer_start(enc->timer_h_to_d, loading_stage->stream);
    enc->error |= host_to_dev(
        enc->d_source,
        data, source_size,
        loading_stage->stream
    );
    
    // quantizer setup (for lossy encoding)
    if(enc->params.compression == CM_LOSSY_FLOAT) {
        j2k_encoder_set_quality(enc, loading_stage->image_params.quality);
        
        // copy stepsizes into GPU buffer
        enc->error |= host_to_dev(
            loading_stage->d_band, 
            enc->band, 
            enc->band_size, 
            loading_stage->stream
        );
    }
    j2k_gpu_timer_stop(enc->timer_h_to_d, loading_stage->stream);
    
    // save user image-specific data pointer
    loading_stage->user_input_data = user_input_data;
    
    // indicate that the input image is ready
    enc->need_input = 0;
    loading_stage->active = 1;
    return 0;
}






/**
 * ****************************************************************************
 * ****************************************************************************
 * ***                                                                      ***
 * ***  IMPLEMENTATION OF BASIC ENCODER INTERFACE USING EXTENDED INTERFACE  ***
 * ***                                                                      ***
 * ****************************************************************************
 * ****************************************************************************
 */




struct single_image_params {
    const void * input;
    void * output_buffer;
    size_t output_buffer_capacity;
    const struct j2k_image_params * params;
    long int result;
};



/**
 * Input callback for encoding single image using extended encoder interface.
 * @param encoder  pointer to instance which called the callback
 * @param user_callback_data  pointer to encoding parameters
 * @param should_block  nonzero if this threead should be blocked while waiting
 * @return always 0 (there will be no more images)
 */
static int
single_image_input_callback(struct j2k_encoder * enc,
                            void * user_callback_data,
                            int should_block) {
    // cast the parameter to encoding info
    struct single_image_params * const p
            = (struct single_image_params*) user_callback_data;
    
    // submit the image
    if(j2k_encoder_set_input(enc, p->input, J2K_FMT_DEFAULT, p->params, 0)) {
        p->result = -1;
    }
    
    // indicate that there will be no more images
    return 0;
}



/**
 * Output callback for encoding single image using extended encoder interface.
 * @param encoder  encoder instance which called the callback
 * @param user_callback_data  pointer to encoding parameters
 * @param user_input_data  not used
 */
static void
single_image_output_callback(struct j2k_encoder * encoder,
                             void * user_callback_data,
                             void * user_input_data) {
    // cast the parameter to encoding info
    struct single_image_params * const params
            = (struct single_image_params*) user_callback_data;
    
    // retrieve the output codestream
    size_t out_size;
    const int result = j2k_encoder_get_output(
            encoder,
            params->output_buffer,
            params->output_buffer_capacity,
            &out_size
    );
    
    // prepare return code for image encoding
    switch(result) {
        case 0:
            params->result = out_size;
            break;
        case 1:
            params->result = 0;
            break;
        default:
            params->result = result;
            break;
    };
}



long int /* negative = error, 0 = small buffer, positive = output byte count */ 
j2k_encoder_compress(struct j2k_encoder * encoder,
                     const void * input,
                     void * output_buffer,
                     size_t output_buffer_capacity,
                     const struct j2k_image_params * params) {
    // check arguments
    if(0 == encoder || 0 == input || 0 == output_buffer) {
        return 0;
    }
    
    // prepare parameters
    struct single_image_params image_encoding_params;
    image_encoding_params.input = input;
    image_encoding_params.output_buffer = output_buffer;
    image_encoding_params.output_buffer_capacity = output_buffer_capacity;
    image_encoding_params.params = params;
    image_encoding_params.result = -1; // default result is error
    
    // run the extended interface with special single-image callbacks
    const int run_result = j2k_encoder_run(
            encoder,
            &image_encoding_params,
            single_image_input_callback,
            single_image_output_callback,
            0
    );
    
    // result is either error of the encoder run or result of image encoding
    return run_result ? -1 : image_encoding_params.result;
}




/**
 * ****************************************************************************
 * ****************************************************************************
 * ***                                                                      ***
 * ***          IMPLEMENTATION OF PAGE-LOCKED MEMORY API FUNCTIONS          ***
 * ***                                                                      ***
 * ****************************************************************************
 * ****************************************************************************
 */



/**
 * Allocates page-locked buffer (for faster encoder input loading)
 * Such buffers should be freed with j2k_encoder_pagelocked_free.
 * @param size  size of the buffer in bytes
 * @return  pointer to buffer if OK, NULL if failed
 */
void *
j2k_encoder_pagelocked_alloc(size_t size) {
    void * buffer;
    
    return cudaSuccess == cudaHostAlloc(&buffer, size, cudaHostAllocPortable)
            ? buffer : 0;
}


/**
 * Frees page locked buffer allocated with j2k_encoder_pagelocked_alloc.
 * @param buffer  pointer to buffer allocated by j2k_encoder_pagelocked_alloc
 * @return 0 if OK, nonzero for error
 */
int 
j2k_encoder_pagelocked_free(void * buffer) {
    return cudaSuccess == cudaFreeHost(buffer) ? 0 : 1;
}


/**
 * Page-locks range of memory. Memory should be later unlocked using a call 
 * to function j2k_encoder_buffer_pageunlock.
 * @param buffer  pointer to begin of the memory range to be locked|
 * @param size  size (in bytes) of the memory range to be locked
 * @return  0 if OK, nonzero for failure
 */
int
j2k_encoder_buffer_pagelock(void * buffer, size_t size) {
    if(cudaSuccess == cudaHostRegister(buffer, size, cudaHostRegisterPortable))
        return 0;
    return 1;
}


/**
 * Unlocks page-locked memory.
 * @param buffer  pointer to begin of memory area prevously page-locked 
 *                by a call to j2k_encoder_buffer_pagelock
 * @return 0 if OK, nonzero for error
 */
int
j2k_encoder_buffer_pageunlock(void * buffer) {
    return cudaSuccess == cudaHostUnregister(buffer) ? 0 : 1;
    
}

