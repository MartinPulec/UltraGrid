/* 
 * Copyright (c) 2011, Martin Srom,
 *                     Martin Jirman
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
#include <stdlib.h>
#include <stdio.h>
#include "j2k.h"
#include "j2k_encoder_init.h"
#include "ebcot/mqc/mqc.h"
#include "quantizer/quantizer.h"
#include "j2k_encoder_progression.h"
#include "rate_control/rate_control.h"


/* Minimum of and maximum implementations. */
static int min(const int a, const int b) { return a < b ? a : b; }
static int max(const int a, const int b) { return a > b ? a : b; }


// /** Documented at declaration */
// int
// j2k_encoder_init_device(int device_id, int show_info)
// {
//     int dev_count;
// 
//     cudaGetDeviceCount(&dev_count);
//     cudaCheckError("Get device count");
// 
//     if ( dev_count == 0 ) {
//         printf("No CUDA enabled device\n");
//         return -1;
//     }
// 
//     if ( device_id < 0 || device_id >= dev_count ) {
//         printf("Selected device %d is out of bound. Devices on your system are in range %d - %d\n",
//                device_id, 0, dev_count - 1);
//         return -1;
//     }
// 
//     struct cudaDeviceProp devProp;
//     cudaGetDeviceProperties(&devProp, device_id);
//     cudaCheckError("Get device properties");
// 
//     if ( devProp.major < 1 ) {
//         printf("Device %d does not support CUDA\n", device_id);
//         return -1;
//     }
// 
//     if ( show_info ) {
//         printf("Setting device %d: %s\n", device_id, devProp.name);
//         int driverVersion;
//         if ( cudaSuccess == cudaDriverGetVersion(&driverVersion) ) {
//             printf("Driver version: %d\n", driverVersion);
//         } else {
//             printf("WARNING: Could not get griver version.\n");
//         }
//         int runtimeVersion;
//         if ( cudaSuccess == cudaRuntimeGetVersion(&runtimeVersion) ) {
//             printf("Runtime version: %d\n", runtimeVersion);
//         } else {
//             printf("WARNING: Could not get runtime version.\n");
//         }
//     }
//     cudaSetDevice(device_id);
//     cudaCheckError("Set selected device");
// 
//     return 0;
// }

/**
 * Frees buffers associated to given instance of J2K context.
 * @param ctx  pointer to instance of J2K context
 * @return  zero if successful, nonzero otherwise
 */
int
j2k_encoder_free_buffer(struct j2k_encoder * const encoder)
{
    if(encoder->component) { cudaFreeHost(encoder->component); encoder->component = 0; }
    if(encoder->resolution) { cudaFreeHost(encoder->resolution); encoder->resolution = 0; }
    if(encoder->precinct) { cudaFreeHost(encoder->precinct); encoder->precinct = 0; }
    if(encoder->tilepart) { cudaFreeHost(encoder->tilepart); encoder->tilepart = 0; }
    if(encoder->d_component) { cudaFree(encoder->d_component); encoder->d_component = 0; }
    if(encoder->d_resolution) { cudaFree(encoder->d_resolution); encoder->d_resolution = 0; }
    if(encoder->d_precinct) { cudaFree(encoder->d_precinct); encoder->d_precinct = 0; }
    if(encoder->d_tilepart) { cudaFree(encoder->d_tilepart); encoder->d_tilepart = 0; }
    if(encoder->d_data) { cudaFree(encoder->d_data); encoder->d_data = 0; }
    if(encoder->d_data_preprocessor) { cudaFree(encoder->d_data_preprocessor); encoder->d_data_preprocessor = 0; }
    if(encoder->d_data_dwt) { cudaFree(encoder->d_data_dwt); encoder->d_data_dwt = 0; }
    if(encoder->d_data_quantizer) { cudaFree(encoder->d_data_quantizer); encoder->d_data_quantizer = 0; }
    if(encoder->d_cxd) { cudaFree(encoder->d_cxd); encoder->d_cxd = 0; }
    if(encoder->d_byte) { cudaFree(encoder->d_byte); encoder->d_byte = 0; } // revert incrementation
    if(encoder->c_byte_compact) { cudaFreeHost(encoder->c_byte_compact); encoder->c_byte_compact = 0; }
    if(encoder->d_byte_compact) { cudaFree(encoder->d_byte_compact); encoder->d_byte_compact = 0; }
    if(encoder->d_byte_header) { cudaFree(encoder->d_byte_header); encoder->d_byte_header = 0; }
    if(encoder->c_byte_header) { cudaFreeHost(encoder->c_byte_header); encoder->c_byte_header = 0; }
    if(encoder->d_precinct_permutation) { cudaFree(encoder->d_precinct_permutation); encoder->d_precinct_permutation = 0; }
    if(encoder->c_precinct_permutation) { cudaFreeHost(encoder->c_precinct_permutation); encoder->c_precinct_permutation = 0; }
    if(encoder->d_trunc_distortions) { cudaFree(encoder->d_trunc_distortions); encoder->d_trunc_distortions = 0; }
    if(encoder->d_trunc_sizes) { cudaFree(encoder->d_trunc_sizes); encoder->d_trunc_sizes = 0; }
    if(encoder->d_source) { cudaFree(encoder->d_source); encoder->d_source = 0; }
    if(encoder->d_compact_size) {cudaFree(encoder->d_compact_size); encoder->d_compact_size = 0; }
    for(int i = 3; i--; ) {
        struct j2k_pipeline_stream * const stream = encoder->pipeline + i;
        if(stream->d_band) { cudaFree(stream->d_band); stream->d_band = 0; }
        if(stream->d_cblk) { cudaFree(stream->d_cblk); stream->d_cblk = 0; }
    }
    return 0;
}

/**
 * Divide and round up.
 * @param n numerator
 * @param d denominator
 * @return n/d rounded up to clossest integer
 */
static int
div_rnd_up(const int n, const int d) { return n / d + (n % d ? 1 : 0); }

/**
 * Adds codeblocks of some band into precinct info structure.
 * @param prec_ptr  pointer to updated precinct
 * @param band_idx  index of the source band of the precinct
 * @param prec_size  standard size of the precinct in the resolution
 * @param cblk_size  standard size of the codeblock
 * @param next_cblk_idx_ptr  pointer to index of next unused codeblock info structure
 * @param cblks_ptr  pointer to buffer with all codeblocks
 * @param cblk_count_ptr  output variable for saving number of codeblocks
 * @param bands_ptr  pointer to buffer with all bands
 */
static void
j2k_prec_add_band(const struct j2k_precinct * const prec_ptr,
                  const int const band_idx,
                  const struct j2k_size * const prec_size,
                  const struct j2k_size * const cblk_size,
                  int * const next_cblk_idx_ptr,
                  struct j2k_cblk * const cblks_ptr,
                  struct j2k_size * const cblk_count_ptr,
                  const struct j2k_band * const bands_ptr)
{
    // end of precinct along both axes
    const int prec_end_x = min(bands_ptr[band_idx].size.width, prec_ptr->position.x + prec_size->width);
    const int prec_end_y = min(bands_ptr[band_idx].size.height, prec_ptr->position.y + prec_size->height);
    
    // size of precinct
    const int prec_size_x = max(0, prec_end_x - prec_ptr->position.x);
    const int prec_size_y = max(0, prec_end_y - prec_ptr->position.y);
    
    // numbers of codeblocks along both axes
    cblk_count_ptr->width = div_rnd_up(prec_size_x, cblk_size->width);
    cblk_count_ptr->height = div_rnd_up(prec_size_y, cblk_size->height);
    
    // init all codeblocks
    for(int cblk_y = 0; cblk_y < cblk_count_ptr->height; cblk_y++) {
        for(int cblk_x = 0; cblk_x < cblk_count_ptr->width; cblk_x++) {
            // reserve next unused structuree for codeblock info and advance
            // index to next unused structure
            struct j2k_cblk * cblk_ptr = cblks_ptr + (*next_cblk_idx_ptr)++;
            
            // position of codeblock in the band
            const int cblk_pos_x = prec_ptr->position.x + cblk_x * cblk_size->width;
            const int cblk_pos_y = prec_ptr->position.y + cblk_y * cblk_size->height;
            
            // end of codeblock along both axes
            const int cblk_end_x = min(cblk_pos_x + cblk_size->width, bands_ptr[band_idx].size.width);
            const int cblk_end_y = min(cblk_pos_y + cblk_size->height, bands_ptr[band_idx].size.height);
            
            // initialize codeblock info
            cblk_ptr->band_index = band_idx;
            cblk_ptr->byte_count = 0; // set by MQ coder
            cblk_ptr->byte_index = 0; // initialized later
            cblk_ptr->cxd_count = 0; // set by context modeller
            cblk_ptr->cxd_index = 0; // initialized later
            cblk_ptr->data_index = bands_ptr[band_idx].data_index
                                 + cblk_pos_x
                                 + cblk_pos_y * bands_ptr[band_idx].size.width;
            cblk_ptr->pass_count = 0; // set by MQ coder
            cblk_ptr->size.width = cblk_end_x - cblk_pos_x;
            cblk_ptr->size.height = cblk_end_y - cblk_pos_y;
            cblk_ptr->bitplane_count = 0; // set by context modeller
        }
    }
}

/** 
 * Initializes bands, precincts and codeblocks of some resolution.
 * @param encoder_ptr  pointer to structure of encoder
 * @param comp_idx  index of resolution's source component
 * @param res_idx  index of the resolution
 * @param res_size_ptr  pointer to structure with size of all bands of the resolution
 * @param next_prec_idx_ptr  pointer to index of next unused codeblock to be updated
 * @param next_cblk_idx_ptr  pointer to index of next unused codeblock to be updated
 */
static int
j2k_init_resolutions(const struct j2k_encoder* const encoder_ptr,
                     const struct j2k_component* const comp_ptr,
                     const int res_size_x,
                     const int res_size_y,
                     int * const next_res_idx_ptr,
                     int * const next_band_idx_ptr,
                     int * const next_prec_idx_ptr,
                     int * const next_cblk_idx_ptr,
                     const int resolution)
{
    // scales origins of packets to get their absolute coordinates
    const int scale_shift = encoder_ptr->params.resolution_count + 1 - resolution;
    
    // lowest resolution?
    if(resolution) {
        // get sizes of bands in this resolution
        const int h_size_x = res_size_x / 2;
        const int h_size_y = res_size_y / 2;
        const int l_size_x = res_size_x - h_size_x;
        const int l_size_y = res_size_y - h_size_y;
        
        // not lowest resolution => proces lower resolutions first (returns 
        // offset of this resolution's band's pixels relative to pixels 
        // of all components)
        int pix_offset = j2k_init_resolutions(encoder_ptr, comp_ptr, l_size_x, l_size_y, next_res_idx_ptr, next_band_idx_ptr, next_prec_idx_ptr, next_cblk_idx_ptr, resolution - 1);
        
        // index of the resolution structure
        const int res_struct_idx = (*next_res_idx_ptr)++;
        
        // now inidices to next unused resolution, precinct and codeblock 
        // should be updated for this resolution => initialize it
        struct j2k_resolution * const res_ptr = encoder_ptr->resolution + res_struct_idx;
        res_ptr->band_count = 3;
        res_ptr->band_index = *next_band_idx_ptr;
        res_ptr->precinct_index = *next_prec_idx_ptr;
        res_ptr->precinct_size = encoder_ptr->params.precinct_size[resolution];
        res_ptr->component_index = comp_ptr - encoder_ptr->component;
        res_ptr->level = resolution;
        
        // prepare halved precinct size
        struct j2k_size prec_size;
        prec_size.width = res_ptr->precinct_size.width / 2;
        prec_size.height = res_ptr->precinct_size.height / 2;
        
        // get number of precincts along each axis and total precinct count
        const int prec_count_x = div_rnd_up(h_size_x, prec_size.width);
        const int prec_count_y = div_rnd_up(h_size_y, prec_size.height);
        res_ptr->precinct_count = prec_count_x * prec_count_y;
        
        // reserve 3 structures for info about 3 bands of the resolution
        struct j2k_band * const hl_ptr = encoder_ptr->band + (*next_band_idx_ptr)++;
        struct j2k_band * const lh_ptr = encoder_ptr->band + (*next_band_idx_ptr)++;
        struct j2k_band * const hh_ptr = encoder_ptr->band + (*next_band_idx_ptr)++;
        
        // initialize all 3 bands of the resolution
        // bit_depth set by quantization initialization for all 3 bands
        // stepsize set by quantization initialization for all 3 bands
        hl_ptr->data_index = pix_offset;
        lh_ptr->data_index = hl_ptr->data_index + h_size_x * l_size_y;
        hh_ptr->data_index = lh_ptr->data_index + l_size_x * h_size_y;
        hl_ptr->size.width = h_size_x;
        lh_ptr->size.width = l_size_x;
        hh_ptr->size.width = h_size_x;
        hl_ptr->size.height = l_size_y;
        lh_ptr->size.height = h_size_y;
        hh_ptr->size.height = h_size_y;
        hl_ptr->type = HL;
        lh_ptr->type = LH;
        hh_ptr->type = HH;
        lh_ptr->resolution_index = res_struct_idx;
        hl_ptr->resolution_index = res_struct_idx;
        hh_ptr->resolution_index = res_struct_idx;
        
        // initialize all precincts of the level
        for(int prec_y = 0; prec_y < prec_count_y; prec_y++) {
            for(int prec_x = 0; prec_x < prec_count_x; prec_x++) {
                // get next free structure for the precinct
                struct j2k_precinct * prec_ptr = encoder_ptr->precinct + (*next_prec_idx_ptr)++;
                prec_ptr->resolution_idx = res_struct_idx;
                prec_ptr->cblk_index = *next_cblk_idx_ptr;
                prec_ptr->byte_header_count = 0;  // set by T2
                prec_ptr->byte_header_index = 0;  // initialized later
                prec_ptr->position.x = prec_x * prec_size.width;
                prec_ptr->position.y = prec_y * prec_size.height;
                prec_ptr->abs_position.x = prec_ptr->position.x << (scale_shift + 1);
                prec_ptr->abs_position.y = prec_ptr->position.y << (scale_shift + 1);
                
                // add codeblocks from all 3 bands to the precinct
                j2k_prec_add_band(prec_ptr, hl_ptr - encoder_ptr->band, &prec_size, &encoder_ptr->params.cblk_size, next_cblk_idx_ptr, encoder_ptr->cblk, prec_ptr->cblk_counts + 0, encoder_ptr->band);
                j2k_prec_add_band(prec_ptr, lh_ptr - encoder_ptr->band, &prec_size, &encoder_ptr->params.cblk_size, next_cblk_idx_ptr, encoder_ptr->cblk, prec_ptr->cblk_counts + 1, encoder_ptr->band);
                j2k_prec_add_band(prec_ptr, hh_ptr - encoder_ptr->band, &prec_size, &encoder_ptr->params.cblk_size, next_cblk_idx_ptr, encoder_ptr->cblk, prec_ptr->cblk_counts + 2, encoder_ptr->band);
            }
        }
        
        // return index of first pixel of first of higher resolution
        return pix_offset + res_size_x * res_size_y - l_size_x * l_size_y;
    } else {
        // index of the resolution structure
        const int res_struct_idx = (*next_res_idx_ptr)++;
        
        // get pointer to the resolution info structure and update index to 
        // next unused resolution structure
        struct j2k_resolution * const res_ptr = encoder_ptr->resolution + res_struct_idx;
        
        // initialize the resolution structure
        res_ptr->band_count = 1;
        res_ptr->band_index = *next_band_idx_ptr;
        res_ptr->precinct_index = *next_prec_idx_ptr;
        res_ptr->precinct_size = encoder_ptr->params.precinct_size[resolution];
        res_ptr->component_index = comp_ptr - encoder_ptr->component;
        res_ptr->level = 0;
        
        // number of precincts along each axis and total number of precincts
        const int prec_count_x = div_rnd_up(res_size_x, res_ptr->precinct_size.width);
        const int prec_count_y = div_rnd_up(res_size_y, res_ptr->precinct_size.height);
        res_ptr->precinct_count = prec_count_x * prec_count_y;
        
        // get pointer to the structure for LL band and increase index of next
        // unused band structure
        struct j2k_band * const ll_ptr = encoder_ptr->band + (*next_band_idx_ptr)++;
        
        // initialize the band
        // ll_ptr->bit_depth set by quantization initialization
        // ll_ptr->stepsize set by quantization initialization
        ll_ptr->data_index = comp_ptr->data_index;
        ll_ptr->size.width = res_size_x;
        ll_ptr->size.height = res_size_y;
        ll_ptr->type = LL;
        ll_ptr->resolution_index = res_struct_idx;
        
        // initialize all precincts of the level
        for(int prec_y = 0; prec_y < prec_count_y; prec_y++) {
            for(int prec_x = 0; prec_x < prec_count_x; prec_x++) {
                // get next free structure for the precinct
                struct j2k_precinct * prec_ptr = encoder_ptr->precinct + (*next_prec_idx_ptr)++;
                prec_ptr->resolution_idx = res_struct_idx;
                prec_ptr->cblk_index = *next_cblk_idx_ptr;
                prec_ptr->byte_header_count = 0;  // set by T2
                prec_ptr->byte_header_index = 0;  // initialized later
                prec_ptr->position.x = prec_x * res_ptr->precinct_size.width;
                prec_ptr->position.y = prec_y * res_ptr->precinct_size.height;
                prec_ptr->abs_position.x = prec_ptr->position.x << scale_shift;
                prec_ptr->abs_position.y = prec_ptr->position.y << scale_shift;
                
                // add codeblocks from LL band to the precinct
                j2k_prec_add_band(prec_ptr,
                                  ll_ptr - encoder_ptr->band,  // gets index of the band
                                  &res_ptr->precinct_size,
                                  &encoder_ptr->params.cblk_size,
                                  next_cblk_idx_ptr,
                                  encoder_ptr->cblk,
                                  prec_ptr->cblk_counts,
                                  encoder_ptr->band);
                
                // other two bands are unused in this precinct
                prec_ptr->cblk_counts[1].width = 0;
                prec_ptr->cblk_counts[1].height = 0;
                prec_ptr->cblk_counts[2].width = 0;
                prec_ptr->cblk_counts[2].height = 0;
            }
        }
        
        // higher resolution's pixels begin immediately after this resolution
        return ll_ptr->data_index + res_size_x * res_size_y;
    }
}

/**
 * Allocates some bytes for heeader to all precincts. Expects codeblock count 
 * of all precincts to be set.
 * @param prec_ptr    pointer to array with precinct innfo structures
 * @param prec_count  total number of precincts
 * @return sum of header bytes of all precincts (size of  buffer for headers)
 */
static int
j2k_prec_header_bytes_allocate(struct j2k_precinct * const prec_ptr,
                               const int prec_count)
{
    int prec_headers_bytes = 0;
    for(int prec_idx = 0; prec_idx < prec_count; prec_idx++) {
        // save precinct's header's begin
        prec_ptr[prec_idx].byte_header_index = prec_headers_bytes;
        
        // get maximal number of precinct header bytes 
        // TODO: better estimate needed
        const int cblk_count = prec_ptr[prec_idx].cblk_counts[0].width
                             * prec_ptr[prec_idx].cblk_counts[0].height
                             + prec_ptr[prec_idx].cblk_counts[1].width
                             * prec_ptr[prec_idx].cblk_counts[1].height
                             + prec_ptr[prec_idx].cblk_counts[2].width
                             * prec_ptr[prec_idx].cblk_counts[2].height;
        const int prec_header_bytes = cblk_count * 32;
        
        // round to 16 and uppdate cumulative sum of precinct header bytes
        prec_headers_bytes += (prec_header_bytes + 15) & ~15;
    }
    return prec_headers_bytes;  // total sum of header bytes
}

/**
 * Assigns part of output CX,D buffer to each codeblock. Expects that 
 * dimensions and source band indices of all codelbocks are set and that 
 * all bands have their bit depths set.
 * @param cblk_ptr  pointer to array with all codeblocks
 * @param band_ptr  pointer to array with all bands
 * @param cblk_count  total count of all codeblocks
 * @return  maximal CX,D pairs sum in all codeblocks (size of CX,D buffer)
 */
static int
j2k_cblk_cxd_allocate(struct j2k_cblk * const cblk_ptr,
                      const struct j2k_band * const band_ptr,
                      const int cblk_count)
{
    // cumulative sum of CX,D pair maxima (from all codeblocks)
    int cblks_cxds = 0;
    
    // add CX,D maxima from all codeblocks
    for(int cblk_idx = 0; cblk_idx < cblk_count; cblk_idx++) {
        // set begin of CX,D pairs of the codeblock
        cblk_ptr[cblk_idx].cxd_index = cblks_cxds;
        
        // get parameters of the codeblock
        const int cblk_size_x = cblk_ptr[cblk_idx].size.width;
        const int cblk_size_y = cblk_ptr[cblk_idx].size.height;
        const int cblk_bpp = max(1, band_ptr[cblk_ptr[cblk_idx].band_index].bitplane_limit);
        
        // compute maximal count of CX,D pairs for the codeblock
        // worst case: first bitplane filled with 1s:
        // row begins in RLC mode and all bits are significant:
        // 1) first pixel: RLC mode termination + SC = 4
        // 2) other pixels: ZC + SC = 2
        const int cblk_max_cxds = (cblk_bpp - 1) * cblk_size_x * cblk_size_y
                                + (2 + 8 * cblk_size_x) * ((cblk_size_y + 3) / 4)
                                + cblk_bpp * 3 - 2;
                                
        // remember maximal count of CX,Ds temporarily
        cblk_ptr[cblk_idx].cxd_count = cblk_max_cxds;
        
        // use it to update cumulative maximal sum aligned to 16 bytes
        cblks_cxds += (cblk_max_cxds + 15) & ~15;
    }
    
    return cblks_cxds;
}

/**
 * Assigns output space to all codeblocks. For each codeblock, it reserves some 
 * space for all of its output bytes and truncation points.
 * @param enc  encoder info
 * @return size of buffer for encoded bytes of all codeblocks
 */
static int
j2k_cblk_byte_allocate(struct j2k_encoder * const encoder)
{   
    // cumulative sum of codeblock output bytes
    int cblks_bytes = 0;
    
    // add bytes of all codeblocks
    for(int cblk_idx = 0; cblk_idx < encoder->cblk_count; cblk_idx++) {
        // get info about the codeblock
        const int cblk_max_cxds = encoder->cblk[cblk_idx].cxd_count;
        
        // reserve some space for codeblock's output bytes rounded up to next multiple of 16
        const int cblk_max_bytes = (mqc_calculate_byte_count(cblk_max_cxds) + 15) & ~15;
        encoder->cblk[cblk_idx].byte_index = cblks_bytes + 16;
        cblks_bytes += cblk_max_bytes + 16;
        
        // up to 128 truncation points per codeblock 
        // (Slightly more than limit of pass count specified by standard.)
        encoder->cblk[cblk_idx].trunc_index = (cblk_idx + 1) * 128; 
    }
    
    // return sum of output bytes of all codeblocks
    return cblks_bytes;
}



/**
 * Initializes quantization coefficients of all bands.
 * @param encoder  encoder info
 */
static void
j2k_quantization_init(struct j2k_encoder * const encoder)
{    
    // Prepare bit_depth and stepsize for all bands in all components,
    // but in CPU buffers only. (GPU ones aren't allocated yet.)
    if( encoder->params.compression == CM_LOSSLESS ) {
        quantizer_setup_lossless(encoder);
    } else if ( encoder->params.compression == CM_LOSSY_FLOAT ) {
        quantizer_setup_lossy(encoder, encoder->params.quality_limit);
    } else {
        assert(0); // unknown compression mode
    }
    
    // use calculated bit depths to set the limit for bitplane counts
    for(int b = encoder->band_count; b--;) {
        encoder->band[b].bitplane_limit = encoder->band[b].bit_depth;
    }
}


/** Prints structure of image. */
void
j2k_encoder_structure_dump(const struct j2k_encoder * const encoder)
{
    for(int comp_idx = 0; comp_idx < encoder->params.comp_count; comp_idx++) {
        const struct j2k_component * const comp = encoder->component + comp_idx;
        const int res_begin_idx = comp->resolution_index;
        const int res_end_idx = res_begin_idx + encoder->params.resolution_count;
        printf("  Component #%d:\n", comp_idx);
        printf("    resolutions %d - %d (total %d):\n", res_begin_idx, res_end_idx - 1, res_end_idx - res_begin_idx);
        for(int res_idx = res_begin_idx; res_idx < res_end_idx; res_idx++) {
            const struct j2k_resolution * const res = encoder->resolution + res_idx;
            const int band_begin_idx = res->band_index;
            const int band_end_idx = band_begin_idx + res->band_count;
            const int prec_begin_idx = res->precinct_index;
            const int prec_end_idx = prec_begin_idx + res->precinct_count;
            printf("    Resolution #%d (level %d):\n", res_idx, res->level);
            printf("      Bands %d - %d (total %d):\n", band_begin_idx, band_end_idx - 1, res->band_count);
            const char * band_type_names[] = {"LL", "HL", "LH", "HH"};
            for(int band_idx = band_begin_idx; band_idx < band_end_idx; band_idx++) {
                const struct j2k_band * const band = encoder->band + band_idx;
                printf("        Band #%d: %s, %dx%d, %dbpp\n",
                       band_idx,
                       band_type_names[(int)band->type],
                       band->size.width,
                       band->size.height,
                       band->bitplane_limit);
            }
            printf("      Precincts %d - %d (total %d):\n", prec_begin_idx, prec_end_idx - 1, res->precinct_count);
            for(int prec_idx = prec_begin_idx; prec_idx < prec_end_idx; prec_idx++) {
                const struct j2k_precinct * const prec = encoder->precinct + prec_idx;
                const int cblk_count = prec->cblk_counts[0].width * prec->cblk_counts[0].height
                                     + prec->cblk_counts[1].width * prec->cblk_counts[1].height
                                     + prec->cblk_counts[2].width * prec->cblk_counts[2].height;
                const int cblk_begin_idx = prec->cblk_index;
                const int cblk_end_idx = cblk_begin_idx + cblk_count;
                printf("        Precinct #%d (at %d, %d), codeblocks %d - %d (total %d): \n",
                       prec_idx,
                       prec->position.x,
                       prec->position.y,
                       cblk_begin_idx,
                       cblk_end_idx - 1,
                       cblk_count);
                for(int cblk_idx = cblk_begin_idx; cblk_idx < cblk_end_idx; cblk_idx++) {
                    const struct j2k_cblk * const cblk = encoder->cblk + cblk_idx;
                    const struct j2k_band * const band = encoder->band + cblk->band_index;
                    const int band_relative_pix_begin = cblk->data_index - band->data_index;
                    const int cblk_x = band_relative_pix_begin % band->size.width;
                    const int cblk_y = band_relative_pix_begin / band->size.width;
                    printf("          Cblk #%d (%s, %dx%d at %d,%d) starting at %d, output at %d.\n",
                           cblk_idx,
                           band_type_names[(int)band->type],
                           cblk->size.width,
                           cblk->size.height,
                           cblk_x,
                           cblk_y,
                           cblk->data_index,
                           cblk->cxd_index);
                }
            }
        }
    }
    
    // Progression order dump:
    printf("\nProgression order: \n");
    for(int tpart_idx = 0; tpart_idx < encoder->tilepart_count; tpart_idx++) {
        printf("  Tile-part #%d:\n", tpart_idx);
        const int perm_begin_idx = encoder->tilepart[tpart_idx].precinct_index;
        const int perm_end_idx = encoder->tilepart[tpart_idx].precinct_count
                               + perm_begin_idx;
        for(int perm_idx = perm_begin_idx; perm_idx < perm_end_idx; perm_idx++) {
            const int prec_idx = encoder->c_precinct_permutation[perm_idx];
            const struct j2k_precinct * const prec = encoder->precinct + prec_idx;
            const struct j2k_resolution * const res = encoder->resolution + prec->resolution_idx;
            printf("    %04d -> %4dprec:  %2dcomp  %4d(%4d)x  %4d(%4d)y  %2dres\n",
                   perm_idx,
                   prec_idx,
                   res->component_index,
                   prec->position.x,
                   prec->abs_position.x,
                   prec->position.y,
                   prec->abs_position.y,
                   res->level);
        }
    }
    printf("\n");
}



/** Documented at declaration */
int
j2k_encoder_init_buffer(struct j2k_encoder* encoder)
{
    // Get parameters
    struct j2k_encoder_params* params = &encoder->params;
    
    // calculate number of bands in single component
    encoder->comp_band_count = (params->resolution_count * 3 - 2);
    
    // first determine counts of bands, precincts, codeblocks ...
    const int comp_count = params->comp_count;
    const int res_count = params->resolution_count * comp_count;
    const int band_count = encoder->comp_band_count * comp_count;
    int prec_count = 0;
    int cblk_count = 0;
    
    // examine all resolutions, starting with the highest one
    int l_size_x = params->size.width;
    int l_size_y = params->size.height;
    for(int r = params->resolution_count - 1; r > 0; r--) {
        // update sizes of this level's bands
        const int h_size_x = l_size_x / 2;
        const int h_size_y = l_size_y / 2;
        l_size_x -= h_size_x;
        l_size_y -= h_size_y;
        
        // count codeblocks and precincts of all 3 bands of the level
        prec_count += div_rnd_up(h_size_x, params->precinct_size[r].width / 2)
                    * div_rnd_up(h_size_y, params->precinct_size[r].height / 2);
        const int h_cblks_x = div_rnd_up(h_size_x, params->cblk_size.width);
        const int l_cblks_x = div_rnd_up(l_size_x, params->cblk_size.width);
        const int h_cblks_y = div_rnd_up(h_size_y, params->cblk_size.height);
        const int l_cblks_y = div_rnd_up(l_size_y, params->cblk_size.height);
        cblk_count += h_cblks_x * h_cblks_y
                    + l_cblks_x * h_cblks_y
                    + h_cblks_x * l_cblks_y;
    }
    
    // add precinct and codeblock counts for LL band
    prec_count += div_rnd_up(l_size_x, params->precinct_size[0].width)
                * div_rnd_up(l_size_y, params->precinct_size[0].height);
    cblk_count += div_rnd_up(l_size_x, params->cblk_size.width)
                * div_rnd_up(l_size_y, params->cblk_size.height);
    
    // finally multiply cblk and precinct counts by number of components
    prec_count *= params->comp_count;
    cblk_count *= params->comp_count;
    
    // set all pointers to 0 to know what to free if somethng goes wrong
    encoder->component = 0;
    encoder->resolution = 0;
    encoder->band = 0;
    encoder->precinct = 0;
    encoder->cblk = 0;
    encoder->tilepart = 0;
    encoder->d_component = 0;
    encoder->d_resolution = 0;
    encoder->d_band = 0;
    encoder->d_precinct = 0;
    encoder->d_cblk = 0;
    encoder->d_tilepart = 0;
    encoder->d_data = 0;
    encoder->d_data_preprocessor = 0;
    encoder->d_data_dwt = 0;
    encoder->d_data_quantizer = 0;
    encoder->d_cxd = 0;
    encoder->d_byte = 0;
    encoder->d_byte_compact = 0;
    encoder->c_byte_compact = 0;
    encoder->d_byte_header = 0;
    encoder->c_byte_header = 0;
    encoder->d_precinct_permutation = 0;
    encoder->c_precinct_permutation = 0;
    encoder->fmt_preprocessor = 0;
    encoder->d_trunc_distortions = 0;
    encoder->d_trunc_sizes = 0;
    
    // return value of this function
    int result = 0;
    
    // allocate all common buffers
    const int components_size = sizeof(struct j2k_component) * comp_count;
    const int resolutions_size = sizeof(struct j2k_resolution) * res_count;
    encoder->band_size = sizeof(struct j2k_band) * band_count;
    const int precincts_size = sizeof(struct j2k_precinct) * prec_count;
    encoder->cblk_size = sizeof(struct j2k_cblk) * cblk_count;
    if(cudaSuccess != cudaMallocHost((void**)&encoder->component, components_size)) { result |= 1 << 0; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_component, components_size)) { result |= 1 << 1; }
    if(cudaSuccess != cudaMallocHost((void**)&encoder->resolution, resolutions_size)) { result |= 1 << 2; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_resolution, resolutions_size)) { result |= 1 << 3; }
    if(cudaSuccess != cudaMallocHost((void**)&encoder->band, encoder->band_size)) { result |= 1 << 4; }
    if(cudaSuccess != cudaMallocHost((void**)&encoder->cblk, encoder->cblk_size)) { result |= 1 << 5; }
    if(cudaSuccess != cudaMallocHost((void**)&encoder->precinct, precincts_size)) { result |= 1 << 6; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_precinct, precincts_size)) { result |= 1 << 7; }
    
    // allocate all pipeline-stream-specific buffers
    for(int i = 3; i--; ) {
        struct j2k_pipeline_stream * const stream = encoder->pipeline + i;
        if(cudaSuccess != cudaMalloc((void**)&stream->d_band, encoder->band_size)) { result |= 1 << 8; }
        if(cudaSuccess != cudaMalloc((void**)&stream->d_cblk, encoder->cblk_size)) { result |= 1 << 9; }
    }
    
    // use buffers of pipeline stream #0 for initialization (will be copied to other streams later)
    encoder->d_band = encoder->pipeline[0].d_band;
    encoder->d_cblk = encoder->pipeline[0].d_cblk;
    
    // check allocation result
    if(result) {
        // free all buffers if some buffers not allocated
        j2k_encoder_free_buffer(encoder);
        return result;
    }
    
    // remember counts of codeblocks, precincts and bands
    encoder->cblk_count = cblk_count;
    encoder->precinct_count = prec_count;
    encoder->band_count = band_count;
    
    // indices of next free components, bands, resolutions, and other stuff
    int next_res_idx = 0;
    int next_band_idx = 0;
    int next_prec_idx = 0;
    int next_cblk_idx = 0;
    
    // initialize all structures
    for(int c_idx = 0; c_idx < params->comp_count; c_idx++) {
        // initialize component attributes
        encoder->component[c_idx].data_index = c_idx * params->size.width * params->size.height;
        encoder->component[c_idx].resolution_index = next_res_idx;
        
        // initialize component's resolutions
        j2k_init_resolutions(
            encoder,
            encoder->component + c_idx,
            params->size.width,
            params->size.height,
            &next_res_idx,
            &next_band_idx,
            &next_prec_idx,
            &next_cblk_idx,
            params->resolution_count - 1
        );
    }
    
    // reserve space for precinct headers
    const int prec_headers_bytes = j2k_prec_header_bytes_allocate(encoder->precinct, prec_count);
    
    // number of input pixels
    const int pixel_count = params->comp_count * params->size.width * params->size.height;
    
    // setup data size for current buffer to zero (it will be set when d_data pointer is set to d_data_preprocessor | d_data_dwt | d_data_quantizer)
    encoder->data_size = 0;
    
    // setup data size for d_data_preprocessor and d_data_dwt, it depends on data_type (integer or float)
    encoder->data_preprocessor_size = pixel_count * 4; // 4 == size of int of float
    encoder->data_dwt_size = pixel_count * 4;

    // setup data size for d_data_quantizer (always integer buffer)
    encoder->data_quantizer_size = pixel_count * 4;
    
    // initialize quantization
    j2k_quantization_init(encoder);
    
    // reserve space for codeblock CX,D pairs
    encoder->cxd_size = j2k_cblk_cxd_allocate(encoder->cblk, encoder->band, cblk_count);
    
    // reserve space for encoded codeblock bytes
    encoder->byte_size = j2k_cblk_byte_allocate(encoder) + 16;
    
    // number of tile-parts
    switch(params->capabilities) {
        case J2K_CAP_DEFAULT:
            encoder->tilepart_count = 1;
            break;
        case J2K_CAP_DCI_2K_48:
        case J2K_CAP_DCI_2K_24:
            encoder->tilepart_count = 3;
            break;
        case J2K_CAP_DCI_4K:
            encoder->tilepart_count = 6;
            break;
        default:
            encoder->tilepart_count = 1;
            result |= 1 << 31;
            break;
    }
    
    // allocate remaining buffers
    const int tileparts_size = encoder->tilepart_count * sizeof(struct j2k_tilepart);
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_byte, encoder->byte_size)) { result |= 1 << 10; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_byte_compact, encoder->byte_size)) { result |= 1 << 11; }
    if(cudaSuccess != cudaMallocHost((void**)&encoder->c_byte_compact, encoder->byte_size)) { result |= 1 << 11; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_byte_header, prec_headers_bytes)) { result |= 1 << 12; }
    if(cudaSuccess != cudaMallocHost((void**)&encoder->c_byte_header, prec_headers_bytes)) { result |= 1 << 13; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_data_dwt, encoder->data_dwt_size)) { result |= 1 << 14; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_data_preprocessor, encoder->data_preprocessor_size)) { result |= 1 << 15; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_data_quantizer, encoder->data_quantizer_size)) { result |= 1 << 16; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_cxd, encoder->cxd_size)) { result |= 1 << 17; }
    if(cudaSuccess != cudaMallocHost((void**)&encoder->c_precinct_permutation, prec_count * sizeof(int))) { result |= 1 << 18; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_precinct_permutation, prec_count * sizeof(int))) { result |= 1 << 19; }
    if(cudaSuccess != cudaMallocHost((void**)&encoder->tilepart, tileparts_size)) { result |= 1 << 20; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_tilepart, tileparts_size)) { result |= 1 << 21; }

    // check allocation result
    if(result) {
        // free all buffers if some buffers not allocated
        j2k_encoder_free_buffer(encoder);
        return result;
    }
    
    // try to initialize progression
    if(j2k_encoder_progression_init(encoder)) {
        result = 1 << 23;
    }
    
    // check counts of structures
    assert(res_count == next_res_idx);
    assert(band_count == next_band_idx);
    assert(prec_count == next_prec_idx);
    assert(cblk_count == next_cblk_idx);
    
    // initialize visual weights
    j2k_rate_control_init_weights(encoder);
    
    // copy required structure buffers into GPU memory
    if(cudaSuccess != cudaMemcpy(encoder->d_component, encoder->component, components_size, cudaMemcpyHostToDevice)) { result |= 1 << 24; }
    if(cudaSuccess != cudaMemcpy(encoder->d_resolution, encoder->resolution, resolutions_size, cudaMemcpyHostToDevice)) { result |= 1 << 25; }
    if(cudaSuccess != cudaMemcpy(encoder->d_band, encoder->band, encoder->band_size, cudaMemcpyHostToDevice)) { result |= 1 << 26; }
    if(cudaSuccess != cudaMemcpy(encoder->d_precinct, encoder->precinct, precincts_size, cudaMemcpyHostToDevice)) { result |= 1 << 27; }
    if(cudaSuccess != cudaMemcpy(encoder->d_cblk, encoder->cblk, encoder->cblk_size, cudaMemcpyHostToDevice)) { result |= 1 << 28; }
    if(cudaSuccess != cudaMemcpy(encoder->d_tilepart, encoder->tilepart, tileparts_size, cudaMemcpyHostToDevice)) { result |= 1 << 29; }
    if(cudaSuccess != cudaMemcpy(encoder->d_precinct_permutation, encoder->c_precinct_permutation, sizeof(int) * prec_count, cudaMemcpyHostToDevice)) { result |= 1 << 30; }
    
    // allocate buffers for truncation points
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_trunc_distortions, sizeof(encoder->d_trunc_distortions) * 128 * (1 + cblk_count))) { result |= 1 << 31; }
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_trunc_sizes, sizeof(encoder->d_trunc_sizes) * 128 * (1 + cblk_count))) { result |= 1 << 31; }
    
    // allocate buffers for input
    const size_t bytes_per_sample = params->bit_depth > 16 ? 4 : (params->bit_depth > 8 ? 2 : 1);
    encoder->default_source_size = params->size.width * params->size.height * params->comp_count * bytes_per_sample;
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_source, encoder->default_source_size)) { result |= 1 << 31; }
    
    // allocate buffer for compact output size
    if(cudaSuccess != cudaMalloc((void**)&encoder->d_compact_size, sizeof(*encoder->d_compact_size))) { result |= 1; }
    
    // copy initialized contents of buffers from pipeline stage #0 to other stages
    const struct j2k_pipeline_stream * const src_stream = encoder->pipeline;
    for(int i = 1; i < 3; i++) {
        const struct j2k_pipeline_stream * const dest_stream = encoder->pipeline + i;
        if(cudaSuccess != cudaMemcpy(dest_stream->d_band, src_stream->d_band, encoder->band_size, cudaMemcpyDeviceToDevice)) { result |= 1; }
        if(cudaSuccess != cudaMemcpy(dest_stream->d_cblk, src_stream->d_cblk, encoder->cblk_size, cudaMemcpyDeviceToDevice)) { result |= 1; }
    }
    
    // finally check mempcy status
    if(result) {
        // free all buffers if anything is wrong
        j2k_encoder_free_buffer(encoder);
        return result;
    }
    
    //j2k_encoder_structure_dump(encoder);

    return result;
}