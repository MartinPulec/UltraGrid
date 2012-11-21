/**
 * @file    dwt.h
 * @author  Martin Jirman (207962@mail.muni.cz)
 * @brief   Interface for CUDA implementaion of 9/7 and 5/3 DWT.
 * @date    2011-01-20 11:41
 *
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
 *
 *
 * Following conditions are common for all DWT functions:
 * - Absolutely nothing is done if 0 levels of DWT are specified. Don't expect 
 *   the input to be copied into the output buffer as a result of 0 DWT levels.
 *   However, helper functions (dwt_get_band_count and dwt_get_band_info)
 *   work correctly even for 0 levels of DWT.
 * - Both input and output images are stored in GPU memory with no padding
 *   of lines or interleaving of pixels.
 * - DWT coefficients are stored as follows: Each band is saved as one
 *   consecutive chunk (no padding/stride/interleaving). Deepest level bands
 *   (smallest ones) are stored first (at the beginning of the input/output
 *   buffers), less deep bands follow. There is no padding between stored
 *   bands in the buffer. Order of bands of the same level in the buffer is
 *   following: Low-low (LL) band (or deeper level subbands) is stored first.
 *   Horizontal-high/vertical-low (HL) follows. Horizonal-low/vertical-high 
 *   band (LH) is saved next and finally, the high-high (HH) band is saved. 
 *   Out of all low-low bands, only the deepest one is saved (right at the 
 *   beginning of the buffer), others are replaced with deeper level subbands.
 * - Input images of all functions won't be preserved (will be overwritten)
 *   if more than one DWT level is required. If only single level is specified,
 *   input buffer remains untouched.
 * - Input and output buffers can't overlap.
 * - Size of output buffer must be greater or equal to size of input buffer.
 *
 */


#ifndef DWT_CUDA_H
#define DWT_CUDA_H


#ifdef __cplusplus
extern "C" {
#endif



/** Format of info written by getBandInfo function. */
typedef struct dwt_band_info_format {
    /**
     * Pointer to place, where level of the first band should be saved,
     * or null if no band levels should be saved at all. Levels equal to number
     * of DWT transformations after which the band was computed. So the deepest
     * (smallest) bands have highest level. Note that this is opposite to what
     * is called 'resolution'.
     */
    int * levels_ptr;

    /** Difference between indices of levels of two consecutive bands. */
    int levels_stride;

    /**
     * Pointer to place, where orientation of the first band should be saved,
     * or null if band orientations should not be saved. For LL bands, 0 is
     * written, HL bands get 1, LH bands 2 and HH bands 3. Those numbers can
     * be interpreted as two bits, where lsb is 1 iff band is horizontal-high
     * band and the other bit is 1 iff band is vertical-high band.
     */
    int * orientations_ptr;

    /**
     * Difference between indices of orientations of two consecutive bands.
     */
    int orientations_stride;

    /**
     * Pointer to integer, where offset (relative to beginning of the buffer)
     * of the first band should be saved, or null if such offsets not needed.
     */
    int * offsets_ptr;

    /** Difference between indices of offsets of two consecutive bands. */
    int offsets_stride;

    /**
     * Pointer to place, where width of the first band should be saved
     * or null if band widths not needed.
     */
    int * x_sizes_ptr;

    /** Difference between indices of widths of two consecutive bands. */
    int x_sizes_stride;

    /** Pointer to height of the first band or null if heights not needed. */
    int * y_sizes_ptr;

    /** Difference between indices of heights of two consecutive bands. */
    int y_sizes_stride;
} dwt_band_info_format_t;



/**
 * Initializes structure for band info format so that all strides are set to 1
 * and all pointer are set 0 (indicating that corresponding attribute is 
 * not needed).
 * @param format  strucure with format of the band info
 */
void dwt_init_band_info_format(dwt_band_info_format_t * format);



/**
 * Gets number of output bands after forward transformation of some image
 * with specified number of DWT levels.
 * @param levels  number of forward transform levels
 * @return total number of bands after transform with given number of levels
 */
int dwt_get_band_count(int levels);



/**
 * Gets info about transformed bands. Can be usefull either for locating
 * individual transformed bands in forward DWT output buffer or for arranging
 * input bands for reverse DWT. Note that forward DWT output band layout
 * exactly matches reverese DWT input band layout. Number of bands for some
 * number of DWT levels can be queried by 'dwt_get_band_count' function.
 * The order of band infos written by this function matches the order
 * of bands in the forward DWT output buffer. (See header of this file.)
 * @param format  info about which individual info elements should be saved
 *                and where they should be saved
 * @param sizeX   width of input image (in pixels)
 * @param sizeY   height of input image (in pixels)
 * @param levels  number of DWT levels
 */
void dwt_get_band_info(
    const dwt_band_info_format_t *format,
    int sizeX,
    int sizeY,
    int levels
);



/**
 * Forward 5/3 2D DWT. See common rules (above) for more details.
 * @param in      input buffer
 * @param out     output buffer on GPU
 * @param sizeX   width of input image (in pixels)
 * @param sizeY   height of input image (in pixels)
 * @param levels  number of recursive DWT levels
 * @param stream  pointer to CUDA stream to run in, or NULL for default stream
 */
void dwt_forward_53(
    int *in,
    int *out,
    int sizeX,
    int sizeY,
    int levels,
    const void * stream
);



/**
 * Reverse 5/3 2D DWT. See common rules (above) for more details.
 * @param in      input DWT coefficients (format described in common rules)
 * @param out     output buffer on GPU - will contain original image
 * @param sizeX   width of input image (in pixels)
 * @param sizeY   height of input image (in pixels)
 * @param levels  number of recursive DWT levels
 * @param stream  pointer to CUDA stream to run in, or NULL for default stream
 */
void dwt_reverse_53(
    int *in,
    int *out,
    int sizeX,
    int sizeY,
    int levels,
    const void * stream
);



/**
 * Forward 9/7 2D DWT. See common rules (above) for more details.
 * @param in      input DWT coefficients
 * @param out     output buffer on GPU - format specified in common rules
 * @param sizeX   width of input image (in pixels)
 * @param sizeY   height of input image (in pixels)
 * @param levels  number of recursive DWT levels
 * @param stream  pointer to CUDA stream to run in, or NULL for default stream
 */
void dwt_forward_97(
    float *in,
    float *out,
    int sizeX,
    int sizeY,
    int levels,
    const void * stream
);



/**
 * Reverse 9/7 2D DWT. See common rules (above) for more details.
 * @param in      input DWT coefficients (format described in common rules)
 * @param out     output buffer on GPU - will contain original image
 * @param sizeX   width of input image (in pixels)
 * @param sizeY   height of input image (in pixels)
 * @param levels  number of recursive DWT levels
 * @param stream  pointer to CUDA stream to run in, or NULL for default stream
 */
void dwt_reverse_97(
    float *in,
    float *out,
    int sizeX,
    int sizeY,
    int levels,
    const void * stream
);


/**
 * Copies partially transformed data from buffer back into input buffer.
 * @param dest  destination for the data (aligned to 4 bytes)
 * @param src  source pointer  (aligned to 4 bytes)
 * @param byteCount  number of bytes to be copied
 * @param stream  pointer to CUDA stream to run in or NULL for default stream
 */
void dwt_cuda_copy(
    void * dest,
    const void * src,
    int byteCount,
    const void * stream
);



#ifdef __cplusplus
} /* end of extern "C" */
#endif


#endif /* DWT_CUDA_H */

