// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Copyright 2019-2026 CESNET, zájmové sdružení právnických osob

/**
 * @file   to_planar.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * This file contains various conversions from planar pixel formats as used
 * by libavcodec or jpegxs to packed pixel formats.
 */

#ifndef TO_PLANAR_H_999A77F3_85E4_4666_880B_1DE038CDE8C6
#define TO_PLANAR_H_999A77F3_85E4_4666_880B_1DE038CDE8C6

#ifdef __cplusplus
extern "C" {
#endif

enum {
        TO_PLANAR_THREADS_AUTO = 0,
        TO_PLANAR_MAX_COMP     = 4,
};

struct to_planar_data {
        int width;
        int height;
        unsigned char *__restrict out_data[TO_PLANAR_MAX_COMP];
        unsigned out_linesize[TO_PLANAR_MAX_COMP];
        const unsigned char *__restrict in_data;
};

/// functions to decode whole buffer of packed data to planar or packed
typedef void
decode_buffer_func_t(struct to_planar_data d);

decode_buffer_func_t v210_to_p010le;
decode_buffer_func_t y216_to_p010le;
decode_buffer_func_t uyvy_to_nv12;
decode_buffer_func_t rgba_to_bgra;
// other packed->planar convs are histaorically in video_codec.[ch]
decode_buffer_func_t uyvy_to_i420;
decode_buffer_func_t r12l_to_gbrp12le;
decode_buffer_func_t r12l_to_gbrp16le;
decode_buffer_func_t r12l_to_rgbp12le;

/**
 * run the @ref decode_buffer_func_t from packed format  in parallel
 * @param dec         fn to run
 * @param src_linesize source linesize (vc_get_linesize(width, in_pixfmt))
 * @param num_threads number of threads or DECODE_TO_THREADS_AUTO to use the
 *                    number of logical cores)
 * @note no support for horizontal subsampling for now
 * @sa decode_to_planar_parallel
 */
void decode_to_planar_parallel(decode_buffer_func_t *dec,
                               struct to_planar_data d, int src_linesize,
                               int num_threads);

#ifdef __cplusplus
}
#endif

#endif // defined TO_PLANAR_H_999A77F3_85E4_4666_880B_1DE038CDE8C6
