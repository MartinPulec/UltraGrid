/* 
 * Copyright (c) 2011, Martin Jirman
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

#ifndef J2K_ENCODER_H
#define J2K_ENCODER_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stddef.h>


/** Maximal count of resolutions. */
#define J2K_MAX_RESOLUTION_COUNT 30


/**
 * Compression mode enumeration
 *
 * CM_LOSSLESS - Lossless compression
 * CM_LOSSY_FLOAT - Lossy compression
 */
enum j2k_compression_mode { CM_LOSSLESS, CM_LOSSY_FLOAT }; 


/**
 * Progression order:
 * LRCP = Layer-Resolution-Component-Position (default)
 * RLCP = Resolution-Layer-Component-Position
 * RPCL = Resolution-Position-Component-Layer
 * PCRL = Position-Component-Resolution-Layer
 * CPRL = Component-Position-Resolution-Layer
 */
enum j2k_progression_order {
    PO_LRCP = 0,
    PO_RLCP = 1,
    PO_RPCL = 2,
    PO_PCRL = 3,
    PO_CPRL = 4
}; 


/**
 * Quantization mode enumeration
 *
 * QM_NONE - No quantization
 * QM_IMPLICIT - Scalar implicit (values signalled for LL subband only)
 * QM_EXPLICIT - Scalar explicit (values signalled for each subband)
 */
enum j2k_quantization_mode {
    QM_NONE = 0,
    QM_IMPLICIT = 1,
    QM_EXPLICIT = 2
}; 


/**
 * Codestream capabilities type.
 */
enum j2k_capabilities {
    J2K_CAP_DEFAULT = 0,   // default codestream (not restricted)
    J2K_CAP_DCI_2K_24 = 3, // codestream restricted to DCI 2K with 24 FPS
    J2K_CAP_DCI_2K_48 = 4, // codestream restricted to DCI 2K with 48 FPS
    J2K_CAP_DCI_4K = 5     // codestream restricted to DCI 4K
};


/** Type of encoder internal stuff. */
struct j2k_encoder;


/** Represents size (width and height). */
struct j2k_size
{
    // Size width
    int width;

    // Size height
    int height;
};


/** Parameters of encoder instance (common for all images encoded using it). */
struct j2k_encoder_params
{
    /** Compression mode (CM_LOSSLESS or CM_LOSSY_FLOAT) */
    enum j2k_compression_mode compression;
    
    /** Input image size (width and height in pixels). */
    struct j2k_size size;

    /** Input image samples bit depth. */
    int bit_depth;
    
    /** Set to 1 if input samples are signed, 0 for unsigned samples. */
    int is_signed;

    /** Color component count (e.g. 3 for RGB, 1 for grayscale). */
    int comp_count;

    /** 
     * Resolution count (first resolution level is single LL band, 
     * other resolution levels consist of HL, LH and HH bands each).
     */
    int resolution_count;
    
    /** Packet ordering type in final stream. */
    enum j2k_progression_order progression_order;

    /** 
     * Code-block size (width and height in pixels). Standard permits only 
     * power-of-2 sizes, with maximal sample count limited to 4096.
     */
    struct j2k_size cblk_size;

    /** Precinct size for each resolution level (dimensions in pixels). */
    struct j2k_size precinct_size[J2K_MAX_RESOLUTION_COUNT];
    
    /**
     * Quantization quality limit (can be decreased extra for each image):
     * 1.0 = up to normal quality
     * 0.1 = limited to poor quality
     * 1.2 = extra quality possible
     * (This affects total required buffers size.)
     */
    float quality_limit;
    
    /** Number of guard bits for quantization */
    int guard_bits;
    
    /** Set to 1 to perform multi component transform, 0 otherwise. */
    int mct;
    
    /** Nonzero if SOP markers (start of packet) should be used. */
    int use_sop;
    
    /** Nonzero if EPH markers (end of packet header) should be used. */
    int use_eph;
    
    /** Nonzero = print timing and other info to standard output, 0 = quiet. */
    int print_info;
    
    /** capabilities of the codestream */
    enum j2k_capabilities capabilities;
    
    /** 
     * Bit depth to be signalized (regardless of the count of actually encoded 
     * bitplanes) or -1 to signalize input bit depth
     */
    int out_bit_depth;
    
#if 0
    /** Code-block style */
    int cblk_style;
    
    /** Stepsize for each band (exponent and mantisa) */
    struct j2k_stepsize band_stepsize[J2K_MAX_BAND_COUNT];
#endif
}; /* end of j2k_encoder_params */


/** Parameters of encoding of single image. */
struct j2k_image_params
{
    /** 
     * Quantization quality for the image 
     * (limited by encoder-specific quantization quality).
     * 0.1 = poor quality, 1.0 = full quality, 1.2 = extra quality
     */
    float quality;
    
    /** 
     * Required maximal output byte count or 0 for "no limit". 
     * Used in post-compression-rate-distortion based rate control.
     * WARNING: this limit may be exceeded a bit (if headers size 
     * is not estimated correctly) as well as not fully utilized 
     * (e.g. if there is not enough information to be encoded).
     */
    size_t output_byte_count;
};


/**
 * Initializes all encoder parameters to their default values:
 *   - lossless compression
 *   - input bit depth: 8
 *   - image size 0x0 (must be set)
 *   - unsigned samples
 *   - 3 color components
 *   - 3 resolutions (2 DWT levels)
 *   - MCT off
 *   - codeblock size 32x32
 *   - layer-resolution-component-position progression
 *   - all precinct sizes 256x256 except first resolution (128x128)
 *   - 2 guard bits
 *   - quality limit 1.0
 *   - using both EPH and SOP
 *   - quiet (no info on stdout)
 *   - default codestream capabilities
 *   - output bit depth override: no override
 * @param params  structure for parameters to be initialized
 */
void
j2k_encoder_params_set_default(struct j2k_encoder_params * params);


/**
 * Sets all image parameters to default values (quality 100 - additionaly 
 * limited by encoder quality limit and no byte count limit).
 * @param params  pointer to structure with image parameters.
 */
void
j2k_image_params_set_default(struct j2k_image_params * params);


/** 
 * Allocates and initializes encoder instance. Instance is bound to the 
 * CPU thread, where it was created. If other than default GPU is required,
 * GPU selection must be done prior to calling this fucntion in the thread.
 * @param params  encoder parameters (should be always initialized with
 *                j2k_encoder_params_set_default before setting)
 * @return either pointer to newly initialized instance of jpeg 2000 encoder
 *         or NULL if error occured during initialization
 */
struct j2k_encoder *
j2k_encoder_create(const struct j2k_encoder_params * params);


/**
 * Releases all resources associated with the encoder instance.
 * @param encoder  pointer to encoder instance
 */
void
j2k_encoder_destroy(struct j2k_encoder * encoder);



/**
 * TODO: add documentation
 * 
 * Input buffer samples are interleaved (e.g. R0 G0 B0 R1 G1 B1 R2 ...) with 
 * no padding (no pixel padding, nor line padding). All components share 
 * same bit depth (specified in encoder parameters in encoder contruction time,
 * together with image size and component count). Bit depth is aligned up 
 * to either 1, 2 or 4 bytes. Sample values are in most significant bits.
 * Least significant bits are eventualy unused. For example: if sample depth
 * is 11, each sample uses 2 bytes, packed in its 11 most significant bits.
 * Values of 5 least significant bits does not matter in such case.
 * 
 * 
 */
long int /* negative = error, 0 = small buffer, positive = output byte count */ 
j2k_encoder_compress(
    struct j2k_encoder * encoder,
    const void * input,
    void * output_buffer,
    size_t output_buffer_capacity,
    const struct j2k_image_params * params
);



// // Returns output size if OK or 0 if the host output buffer is not big enough
// // Should be called only from output callback, repeated until result fits into the buffer.
// // (The pointer need not point to page locked memory.)
// size_t j2k_encoder_get_output(j2k_encoder * enc, void * output_buffer_ptr, size_t output_buffer_size);
// 
// void j2k_encoder_set_input(j2k_encoder * enc, const void * data, const j2k_image_params * params, void * input_specific_data);
// 
// 
// 
// // Calls input callbacks for getting input images 
// // and output callbacks to report encoded images.
// // All callbacks are called within the same host thread 
// // (the one, in which the j2k_enc_run was called).
// // Returns 0 after all submitted images were encoded or nonzero error code.
// int j2k_enc_run(
//     j2k_enc * enc,
//     void * callback_param,  // value will be passed to all callbacks called from this run
//     // If there are no more input images, j2k_input_none should be returned - 
//     // the input callback will not be called again in this case. If there is no prepared input 
//     // yet (but will be later), j2k_enc_input_not_ready_yet should be returned - callback will 
//     // be called later again. If input buffer was set, j2k_enc_input_ready should be returned.
//     // Memory copy to device is more efficient if input pointer points to page locked host memory. 
//     // Value of last parameter will only be passed to corresponding output callback 
//     // (good for some image specific metadata such as frame ID, filename, etc.).
//     // TODO: add frame-specific parameters (quality, rate)
//     int (*input_callback)(void * callback_param),
//     void (*output_callback)(void * callback_param, void * input_specific_data)
// )
// 
// 
// 
// 
// 
// 
// 
// 
// {
//     
//     
//     cudaStream_t host_to_device_stream, kernel_stream, device_to_host_stream;
//     
//     bool should_load = true;
//     bool should_encode = false;
//     bool should_format = false;
//     
//     
//     
//     while (should_load || should_encode || should_format) {
//         bool is_encoding = false;
// 
//         if (should_encode) {
//             run_kernels(kernel_stream)
//             is_encoding = true;
//             should_encode = false;
//         }
// 
//         bool run_t2 = false;
//         event = null
//         if(should_format) {
//             mempcy D -> H (in device_to_host_stream)
//             event = create_and_push_event();
//             run_t2 = true;
//         }
//         
//         if(should_load) {
//             input_callback()
//             if(frame == null) {
//                 should_load = false;
//             } else {
//                 memcpy H -> D (in host_to_device_stream)
//                 should_encode = true;
//             }
//         }
//         
//         if(run_t2) {
//             wait_event_and_destroy_it(event);
//             t2();
//             output_callback();
//         }
//         
//         should_format = is_encoding;
//         
//         // cycle streams
//         const cudaStream_t unused_stream = device_to_host_stream;
//         device_to_host_stream = kernel_stream;
//         kernel_stream = host_to_device_stream;
//         host_to_device_stream = unused_stream;
//     }
//     
//     
//     
//     
// }
// // 

#ifdef __cplusplus
} /* end of extern "C" */
#endif 

#endif /* J2K_ENCODER_H */
