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

#ifndef J2K_H
#define J2K_H

#include <cuda_runtime_api.h>
#include "j2k_encoder_extended.h"

#ifdef __cplusplus
extern "C" {
#endif



/** Constants */
#define J2K_MAX_COMP_COUNT          3

#define J2K_MAX_BAND_COUNT          (J2K_MAX_RESOLUTION_COUNT * 3 + 1)




/** Band orientation type */
enum j2k_band_type { LL = 0, HL = 1, LH = 2, HH = 3 };


/** Represents 2D position. */
struct j2k_position
{
    // x-axis position
    int x; 
    
    // y-axis position
    int y; 
};


/**
 * Component structure
 *
 * Each component is by dwt divided into resolutions.
 */
struct j2k_component
{
    // Component data => Index into current DATA buffer [d_data]
    int data_index;

    // Component resolutions => Index into STRUCTURE buffer [resolution]
    int resolution_index;
};


/**
 * Resolution structure
 *
 * Each resolution is by dwt divided into bands. Resolution 0 containes one LL band and
 * all other resolutions container HL, LH and HH bands.
 * Each resolution is divided into precincts too. One precinct contains few code-blocks
 * from all bands in current resolution.
 */
struct j2k_resolution
{
    // Parent component
    int component_index;
    
    // level of resolution (0 => contains LL only, higher => 3 bands)
    int level;
    
    // Precincts size (width and height as exponent power of two)
    struct j2k_size precinct_size;

    // Resolution bands => Index into STRUCTURE buffer [band]
    int band_index;

    // Bands count in resolution (1 or 3)
    int band_count;

    // Resolution precincts => Index into STRUCTURE buffer [precinct]
    int precinct_index;

    // Precincts count in resolution
    int precinct_count;
};


/** Band info structure */
struct j2k_band
{
    // Parent resolution.
    int resolution_index;
    
    // Band type
    enum j2k_band_type type;

    // Band size (width and height in pixels)
    struct j2k_size size;

    // Band data bit depth
    int bit_depth;

    // Band data => Index into current DATA buffer [d_data]
    int data_index;

    // LOSSY compression extra parameters:
    
    // Band stepsize for quantization
    float stepsize;
    
    // Band stepsize for quantization (mantisa)
    int stepsize_mantisa;
    
    // Band stepsize for quantization (exponent)
    int stepsize_exponent;
    
    // Maximal number of bitplanes encoded in this band.
    int bitplane_limit;
    
    // Visual weight of band's coefficients (higher = more important)
    float visual_weight;
};

/** Precinct info structure */
struct j2k_precinct
{
    // Precinct code-blocks => Index into STRUCTURE buffer [cblk]
    int cblk_index;
    
    // index of resolution structure to which the precinct belongs
    int resolution_idx;

    // Code-block count in precinct (for all 3 bands)
    struct j2k_size cblk_counts[3];

    // Precinct header bytes => Index into BYTE buffer [d_byte_header]
    int byte_header_index;

    // Byte count of precinct header
    int byte_header_count;
    
    // Position of the precinct (in pixels) relative to band origin
    struct j2k_position position;
    
    // Absolute position of the precinct before mapping to its resolution
    // (used for progression ordering)
    struct j2k_position abs_position;
};

/** Code-block info structure */
struct j2k_cblk
{
    // Code-block size (width and height in pixels)
    struct j2k_size size;

    // Code-block data => Index into current DATA buffer [d_data]
    int data_index;

    // Band containing this code-block through precinct => Index into STRUCTURE buffer [band]
    int band_index;

    // Code-block CX,D pairs => Index into CX,D buffer [d_cxd]
    int cxd_index;
    
    // CX,D pairs count in code-block (including "pass end" special pairs;
    // set to maximal CX,D count during initialization, later overwritten 
    // with final CX,D count by cxmod)
    int cxd_count;

    // Code-block bytes => Index into byte buffer [d_byte]
    int byte_index;

    // Bytes count in code-block (filled by mqc, possibly updated by rate control)
    int byte_count;
    
    // Code-block bytes after compaction (Index into compact buffer 
    // [d_byte_compact], set by output compaction).
    int byte_index_compact;

    // Coded passes count (filled by mqc, possibly updated by rate control)
    int pass_count;

    // Nonzero bitplanes count (significant bitplanes; filled by cxmod)
    int bitplane_count;
    
    // Index of distortion and byte size for first truncation point 
    // (multiple of 16, invariant, filled by initialization) 
    // index to both [d_trunc_distortions] and [d_trunc_sizes]
    int trunc_index;
    
    // Count of truncation points for the codeblock. (filled by cxmod)
    int trunc_count;
};


/** Tile part info structure */
struct j2k_tilepart
{
    // Tile part precincts => Index into STRUCTURE buffer [precinct]
    int precinct_index;
    
    // Precincts count in tile part
    int precinct_count;
};



/** One stream for memcpy-kernel-memcpy pipelining. */
struct j2k_pipeline_stream {
    /* CUDA stream for this pipeline stream */
    cudaStream_t stream;
    
    /* nonzero if image is being processed in this stream, 0 if unused */
    int active;
    
    /* stream's event for synchronization with this stream's actions */
    cudaEvent_t event;
    
    // All bands buffer (for all components of the stream's image)
    struct j2k_band * d_band;
    
    // All code-blocks buffer (for all components of the stream's image)
    struct j2k_cblk * d_cblk;
    
    // All bands buffer (for all components of currently encoded image)
    struct j2k_band * band;
    
    // All code-blocks buffer (for all components of currently encoded image)
    struct j2k_cblk * cblk;
    
    // Image specific parameters
    struct j2k_image_params image_params;
    
    // user data pointer associated with last input
    void * user_input_data;
    
    // pointer to last source data buffer
    const void * source_ptr;
    
    // format of last input data
    enum j2k_input_format format;
};


/** Encoder internal stuff type */
struct j2k_encoder {
    // Private copy of encoder parameters given to encoder constructor.
    struct j2k_encoder_params params;
    
    // Quantization mode
    enum j2k_quantization_mode quantization_mode;
    
    // Preprocessor
    void* preprocessor;
    
    // Context-modeller
    void* cxmod;
    
    // MQ-Coder
    void* mqc;
    
    // T2
    struct j2k_t2_encoder* t2;
    
    // STRUCTURE buffers:
    // ------------------
    // All components buffer
    struct j2k_component* component;
    // All resolutions buffer (for all components)
    struct j2k_resolution* resolution;
    // All bands buffer (for all components of currently encoded image)
    struct j2k_band* band;
    // All precincts buffer (for all components)
    struct j2k_precinct* precinct;
    // All code-blocks buffer (for all components of currently encoded image)
    struct j2k_cblk* cblk;
    // All tile parts buffer
    struct j2k_tilepart* tilepart;
    
    // GPU copies of STRUCTURE buffers:
    // ------------------
    // All components buffer
    struct j2k_component* d_component;
    // All resolutions buffer (for all components)
    struct j2k_resolution* d_resolution;
    // All bands buffer (for all components of currently encoded image)
    struct j2k_band* d_band;
    // All precincts buffer (for all components)
    struct j2k_precinct* d_precinct;
    // All code-blocks buffer (for all components of currently encoded image)
    struct j2k_cblk* d_cblk;
    // All tile parts buffer
    struct j2k_tilepart* d_tilepart;
    
    // Band total count in one component
    int comp_band_count;
    
    // Code-block total count (for all components, all bands, etc.)
    int cblk_count;
    
    // Precinct count (in all components, bands, etc...)
    int precinct_count;
    
    // Tilepart total count (for all components, all bands, etc.)
    int tilepart_count;
    
    // count of all bands in all components
    int band_count;
    
    // Source samples buffers
    void* d_source;
    
    // Default source size (if no special source format is used)
    size_t default_source_size;
    
    // DATA, CXD and BYTE buffers:
    // ---------------------------
    // Data buffer that points to current data (integer or real)
    // It points to d_datapreprocessor or d_data_dwt or d_data_quantizer)
    void* d_data;
    // Data buffer for preprocessor output data (integer or real)
    void* d_data_preprocessor;
    // Data buffer for dwt output data (integer or real)
    void* d_data_dwt;
    // Data buffer for quantizer output data (always integer)
    int* d_data_quantizer;
    // CX,D buffer for cxmod output CX,D pairs
    unsigned char* d_cxd;
    // Byte buffer for mqc output bytes (GPU)
    unsigned char* d_byte;
    // Byte buffer for compacted mqc output bytes (GPU)
    unsigned char* d_byte_compact;
    // Device memory integer for compact output size.
    unsigned int* d_compact_size;
    // Byte buffer for t2 precinct headers (GPU)
    unsigned char* d_byte_header;
    // Byte buffer for compact mqc output bytes (CPU)
    unsigned char* c_byte_compact;
    // Byte buffer for t2 precinct headers (CPU)
    unsigned char* c_byte_header;
    // Progression order (indices of precinct permutation, GPU)
    int* d_precinct_permutation;
    // Progression order (indices of precinct permutation, CPU)
    int* c_precinct_permutation;
    // Truncation point distortions (GPU)
    float* d_trunc_distortions;
    // Codeblock byte counts corresponding to distortions (GPU)
    unsigned int* d_trunc_sizes;
    
    // DATA, CXD and BYTE buffer sizes:
    // ---------------------------
    // Data buffer [d_data] size
    int data_size;
    // Data buffer [d_data_preprocessor] size
    int data_preprocessor_size;
    // Data buffer [d_data_dwt] size
    int data_dwt_size;
    // Data buffer [d_data_quantizer] size
    int data_quantizer_size;
    // CX,D buffer size
    int cxd_size;
    // Byte buffer size
    int byte_size;
    // Band buffer size
    int band_size;
    // Codeblock buffer size
    int cblk_size;
    
    // preprocessor for special formats
    struct j2k_fmt_preprocessor * fmt_preprocessor;
    
    // maximal byte count (for rate control, 0 == no limit)
    size_t max_byte_count;
    
    // CUDA device info
    struct cudaDeviceProp gpu_info;
    
    // nonzero if j2k_encoder_set_input can be called
    int need_input;
    
    // nonzero if j2k_encoder_get_output can be called
    int have_output;
    
    // error state of the encoder (0 == OK)
    int error;
    
    // encoding pipeline stages (loading, encoding, saving)
    struct j2k_pipeline_stream pipeline[3];
    
    // pointer to currently loading and saving streams
    struct j2k_pipeline_stream * in_stream;
    struct j2k_pipeline_stream * out_stream;
    
    // pointers to timers or NULLs if no time measurement required:
    struct j2k_gpu_timer * timer_h_to_d;
    struct j2k_gpu_timer * timer_preproc;
    struct j2k_gpu_timer * timer_dwt;
    struct j2k_gpu_timer * timer_quant;
    struct j2k_gpu_timer * timer_cxmod;
    struct j2k_gpu_timer * timer_mqc;
    struct j2k_gpu_timer * timer_rate;
    struct j2k_gpu_timer * timer_compact;
    struct j2k_gpu_timer * timer_d_to_h;
    struct j2k_cpu_timer * timer_t2;
    struct j2k_cpu_timer * timer_run;
};







// /**
//  * JPEG2000 Codec structure
//  */
// struct j2k_codec {
//     // Preprocessor
//     struct preprocessor* preprocessor;
// 
//     // Tier-1
//     struct t1* t1;
//     
//     // GPU device ID
//     int device_id;
// 
//     // Image parameters
//     struct j2k_param_image param_image;
// 
//     // Compression parameters
//     struct j2k_param_compress param_compress;
// };
// 
// /**
//  * Reset image parameters structure to default values
//  *
//  * @param param_image  Image parameters structure
//  * @return void
//  */
// void
// j2k_param_image_reset(struct j2k_param_image * param_image);
// 
// /**
//  * Reset JPEG2000 compress parameters structure to default values
//  *
//  * @param param_compress  Compress parameters structure
//  * @return void
//  */
// void
// j2k_param_compress_reset(struct j2k_param_compress * param_compress);
// 
// /**
//  * Initialize j2k codec structure
//  *
//  * @param device_id  GPU device ID
//  * @param param_image  Image parameters
//  * @param param_compress  Compress parameters
//  * @return J2K codec structure if OK, 0 otherwise
//  */
// struct j2k_codec* 
// j2k_codec_create(int device_id, struct j2k_param_image* param_image, struct j2k_param_compress* param_compress);
// 
// /**
//  * Clean up j2k codec structure
//  *
//  * @param j2k  J2K codec structure
//  * @return void
//  */
// void
// j2k_codec_destroy(struct j2k_codec* j2k_codec);
// 
// /**
//  * Register an image with the codec. Function allocates device buffers 
//  * for the image.
//  *
//  * @param j2k  J2K codec structure
//  * @param image  Image structure
//  * @return 0 if OK, nonzero otherwise
//  */
// int
// j2k_register_img(struct j2k_codec* j2k_codec, struct image* image);
// 
// /**
//  * Unregister an image from the codec. Function deallocates device buffers 
//  * for the image.
//  *
//  * @param j2k  J2K codec structure
//  * @param image  Image structure
//  * @return 0 if OK, nonzero otherwise
//  */
// int
// j2k_unregister_img(struct j2k_codec* j2k_codec, struct image* image);
// 
// /**
//  * Compress image
//  *
//  * @param j2k_codec  J2K codec structure
//  * @param image  Image structure
//  * @return 0 if OK, nonzero otherwise
//  */
// int
// j2k_compress(struct j2k_codec* j2k_codec, struct image* image, struct j2k_verify* j2k_verify);



#ifdef __cplusplus
} // end of extern "C"
#endif


#endif // J2K_H
