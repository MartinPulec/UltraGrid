// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Copyright 2026 CESNET, zájmové sdružení právnických osob

/**
 * @file   from_planar.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file contains various conversions from planar pixel formats as used
 * by libavcodec or jpegxs to packed pixel formats.
 */

#ifndef FROM_PLANAR_H_E96E5C5D_6F9C_480D_AD6E_D0AD5149378F
#define FROM_PLANAR_H_E96E5C5D_6F9C_480D_AD6E_D0AD5149378F

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

enum {
        FROM_PLANAR_THREADS_AUTO = 0,
        FROM_PLANAR_MAX_COMP = 4,
};

// defined also in pixfmt_conv.h
#define DEFAULT_R_SHIFT  0
#define DEFAULT_G_SHIFT  8
#define DEFAULT_B_SHIFT 16

struct from_planar_data {
        int width;
        int height;
        unsigned char *__restrict out_data;
        unsigned out_pitch;
        const unsigned char *__restrict in_data[FROM_PLANAR_MAX_COMP];
        unsigned in_linesize[FROM_PLANAR_MAX_COMP];
        // the following fields are not always needed
        int in_depth; ///< needed just for XX (generic) conversions
        /// set to 1 for Y420 and decode_planar_parallel, otherwise unused
        int log2_chroma_h; ///< semantic same as in FF (420 - 1, otherwise 0)
        int rgb_shift[3]; ///< relevant only for output RGBA
};

/// functions to decode whole buffer of packed data to planar or packed
typedef void         decode_planar_func_t(struct from_planar_data d);
/**
 * run the @ref decode_planar_func_t in parallel
 * @param dec         fn to run, must be to packed format (or set num_threads=1
 *                    if the conversion is to a planar format)
 * @param num_threads number of threads or FROM_PLANAR_THREADS_AUTO to use the
 *                    number of logical cores)
 */
void decode_planar_parallel(decode_planar_func_t   *dec,
                            struct from_planar_data d, int num_threads);

// XX in the conversion names below indicates that in_depth must be set.
// If =8, byte per input sample is used, if >8, 16 bits.
decode_planar_func_t gbrap_to_rgb;
decode_planar_func_t gbrap_to_rgba;
decode_planar_func_t gbrp10le_to_rgb;
decode_planar_func_t gbrp10le_to_rgba;
decode_planar_func_t gbrp10le_to_rg48;
decode_planar_func_t gbrp10le_to_r10k;
decode_planar_func_t gbrp12le_to_rgb;
decode_planar_func_t gbrp12le_to_rgba;
decode_planar_func_t gbrp12le_to_rg48;
decode_planar_func_t gbrp12le_to_r10k;
decode_planar_func_t gbrp12le_to_r12l;
decode_planar_func_t gbrp16le_to_rgb;
decode_planar_func_t gbrp16le_to_rgba;
decode_planar_func_t gbrp16le_to_rg48;
decode_planar_func_t gbrp16le_to_r10k;
decode_planar_func_t gbrp16le_to_r12l;
decode_planar_func_t rgbpXX_to_rgb;
decode_planar_func_t rgbpXXle_to_rg48;
decode_planar_func_t rgbpXXle_to_r10k;
decode_planar_func_t rgbpXXle_to_r12l;
decode_planar_func_t yuv420p_to_uyvy;
decode_planar_func_t yuv420_to_i420;
decode_planar_func_t yuv422p_to_uyvy;
decode_planar_func_t yuv422p_to_yuyv;
decode_planar_func_t yuv422pXX_to_uyvy;
decode_planar_func_t yuv422p10le_to_uyvy;
decode_planar_func_t yuv422p10le_to_v210;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // definedFROM_PLANAR_H_E96E5C5D_6F9C_480D_AD6E_D0AD5149378F
