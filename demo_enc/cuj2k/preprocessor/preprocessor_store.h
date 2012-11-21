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

#ifndef J2K_PREPROCESSOR_STORE_H
#define J2K_PREPROCESSOR_STORE_H

/**
 * Normalize component input value.
 *
 * @template bit_depth  Input value bit depth
 * @template is_signed  Flag if input value is signed integer (default: unsigned)
 */
template<int bit_depth, bool is_signed = false>
struct j2k_value_normalize {
    /**
     * Default implementation is to do no normalization.
     *
     * @param value  Value to be normalized
     * @return normalized value
     */
    static __device__ void perform(int & value) {
    }
};

/** Specialization [is_signed = false] */
template<int bit_depth>
struct j2k_value_normalize<bit_depth, false> {
    /**
     * If input value is unsigned integer, normalize component input value to n-bit 
     * positive/negative integer (n = bit_depth) where middle value is 0.
     *
     * For instance (bit_depth = 8):
     * 1) value_normalize<8>::perform(0)   = -128
     * 2) value_normalize<8>::perform(128) = 0
     * 3) value_normalize<8>::perform(255) = 127
     */
    static __device__ void perform(int & value) {
        value = value - (1 << (bit_depth - 1));
    }
};

/**
 * Store n component values to n buffers and perform normalization
 *
 * @template bit_depth  Input value bit depth
 * @template is_signed  Input value is signed integer
 * @template tranform  Color component transformation
 */
template<int bit_depth, bool is_signed = false, enum j2k_component_transform transform = CT_NONE>
struct j2k_store_component {

    /**
     * Store 1 component value to 1 buffer and perform normalization
     *
     * @param d_c  Component buffer
     * @param c  Component value
     * @param pos  Component value position in buffer
     */
    template<class data_type>
    static __device__ void perform(data_type* d_c, int c, int pos)
    {
        j2k_value_normalize<bit_depth, is_signed>::perform(c);
        d_c[pos] = static_cast<data_type>(c);
    }

    /**
     * Store 3 component values to 3 buffers and perform normalization and component transformation
     *
     * @param d_c1  First component buffer
     * @param d_c2  Second component buffer
     * @param d_c3  Third component buffer
     * @param c1  First component value
     * @param c2  Second component value
     * @param c3  Third component value
     * @param pos  Component value position in buffer
     */
    template<class data_type>
    static __device__ void perform(data_type* d_c1, data_type* d_c2, data_type* d_c3, int c1, int c2, int c3, int pos)
    {
        j2k_value_normalize<bit_depth, is_signed>::perform(c1);
        j2k_value_normalize<bit_depth, is_signed>::perform(c2);
        j2k_value_normalize<bit_depth, is_signed>::perform(c3);

        data_type r1 = static_cast<data_type>(c1);
        data_type r2 = static_cast<data_type>(c2);
        data_type r3 = static_cast<data_type>(c3);

        j2k_component_transform_forward<transform>::perform(r1, r2, r3);

        d_c1[pos] = r1;
        d_c2[pos] = r2;
        d_c3[pos] = r3;
    }
};

#endif // J2K_PREPROCESSOR_STORE_H

