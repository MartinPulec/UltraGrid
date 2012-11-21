/**
 * @file    dwt.c
 * @author  Martin Jirman (martin.jirman@cesnet.cz)
 * @brief   Implementation of helper functions for GPU DWT.
 * @date    2011-07-21 11:27
 *
 *
 * Copyright (c) 2011 Martin Jirman
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
 *
 */


#include "dwt.h"



/**
 * Initializes structure for band info format so that all strides are set to 1
 * and all pointer are set 0 (indicating that corresponding attribute is 
 * not needed).
 * @param format  strucure with format of the band info
 */
void dwt_init_band_info_format(dwt_band_info_format_t * format) {
    format->levels_ptr = 0;
    format->levels_stride = 1;
    format->orientations_ptr = 0;
    format->orientations_stride = 1;
    format->offsets_ptr = 0;
    format->offsets_stride = 1;
    format->x_sizes_ptr = 0;
    format->x_sizes_stride = 1;
    format->y_sizes_ptr = 0;
    format->y_sizes_stride = 1;
}



/**
 * Gets number of output bands after forward transformation of some image
 * with specified number of DWT levels.
 * @param levels  number of forward transform levels
 * @return total number of bands after transform with given number of levels
 */
int dwt_get_band_count(int levels) {
    return 1 + levels * 3;
}



/**
 * Gets info about transformed bands. Can be usefull either for locating
 * individual transformed bands in forward DWT output buffer or for arranging
 * input bands for reverse DWT. Note that forward DWT output band layout
 * exactly matches reverese DWT input band layout. Number of bands for some
 * number of DWT levels can be queried by 'dwt_get_band_count' function.
 * The order of band infos written by this function matches the order
 * of bands in the forward DWT output buffer. (See header of this file.)
 * @param fmt     info about which individual info elements should be saved
 *                and where they should be saved
 * @param size_x  width of input image (in pixels)
 * @param size_y  height of input image (in pixels)
 * @param levels  number of DWT levels
 */
void dwt_get_band_info(const dwt_band_info_format_t * fmt,
                       int size_x, int size_y, int levels) {
    /* write info about LH, HL and HH bands, starting with level 1 */
    int remaining_levels = levels;
    while(remaining_levels--) {
        /* compute sizes of subbands in this level */
        const int h_size_x = size_x / 2;
        const int h_size_y = size_y / 2;
        const int l_size_x = size_x - h_size_x;
        const int l_size_y = size_y - h_size_y;
        
        /* get offsets of infos for the three bands */
        const int hl_index = 1 + remaining_levels * 3;
        const int lh_index = 2 + remaining_levels * 3;
        const int hh_index = 3 + remaining_levels * 3;
        
        /* write all info elements */
        if(fmt->levels_ptr) {
            const int current_level = levels - remaining_levels;
            fmt->levels_ptr[fmt->levels_stride * hl_index] = current_level;
            fmt->levels_ptr[fmt->levels_stride * lh_index] = current_level;
            fmt->levels_ptr[fmt->levels_stride * hh_index] = current_level;
        }
        if(fmt->offsets_ptr) {
            const int hl_offset = l_size_x * l_size_y;
            const int lh_offset = hl_offset + h_size_x * l_size_y;
            const int hh_offset = lh_offset + l_size_x * h_size_y;
            fmt->offsets_ptr[fmt->offsets_stride * hl_index] = hl_offset;
            fmt->offsets_ptr[fmt->offsets_stride * lh_index] = lh_offset;
            fmt->offsets_ptr[fmt->offsets_stride * hh_index] = hh_offset;
        }
        if(fmt->orientations_ptr) {
            fmt->orientations_ptr[fmt->orientations_stride * hl_index] = 1;
            fmt->orientations_ptr[fmt->orientations_stride * lh_index] = 2;
            fmt->orientations_ptr[fmt->orientations_stride * hh_index] = 3;
        }
        if(fmt->x_sizes_ptr) {
            fmt->x_sizes_ptr[fmt->x_sizes_stride * hl_index] = h_size_x;
            fmt->x_sizes_ptr[fmt->x_sizes_stride * lh_index] = l_size_x;
            fmt->x_sizes_ptr[fmt->x_sizes_stride * hh_index] = h_size_x;
        }
        if(fmt->y_sizes_ptr) {
            fmt->y_sizes_ptr[fmt->y_sizes_stride * hl_index] = l_size_y;
            fmt->y_sizes_ptr[fmt->y_sizes_stride * lh_index] = h_size_y;
            fmt->y_sizes_ptr[fmt->y_sizes_stride * hh_index] = h_size_y;
        }
        
        /* adjust size for the next level */
        size_x = l_size_x;
        size_y = l_size_y;
    }
    
    /* finally write info about deepest LL level */
    if(fmt->levels_ptr) {
        *fmt->levels_ptr = levels;
    }
    if(fmt->offsets_ptr) {
        *fmt->offsets_ptr = 0;
    }
    if(fmt->orientations_ptr) {
        *fmt->orientations_ptr = 0;
    }
    if(fmt->x_sizes_ptr) {
        *fmt->x_sizes_ptr = size_x;
    }
    if(fmt->y_sizes_ptr) {
        *fmt->y_sizes_ptr = size_y;
    }
}

