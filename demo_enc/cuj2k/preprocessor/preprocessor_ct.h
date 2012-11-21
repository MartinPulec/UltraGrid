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

#ifndef J2K_PREPROCESSOR_CT_H
#define J2K_PREPROCESSOR_CT_H

#include "preprocessor_ct_type.h"

/**
 * Forward component transformation
 *
 * @param transform
 */
template<enum j2k_component_transform transform>
struct j2k_component_transform_forward {
    /** 
     * Default implementation do nothing [transform = CT_NONE]
     *
     * @param c1  First color component 
     * @param c2  Second color component 
     * @param c3  Third color component 
     */
    template<class data_type>
    static __device__ void perform(data_type & c1, data_type & c2, data_type & c3) {
    }
};

/** Specialization [transform = CT_REVERSIBLE] */
template<>
struct j2k_component_transform_forward<CT_REVERSIBLE> {
    /** Reversible RGB -> YUV transform */
    static __device__ void perform(int & c1, int & c2, int & c3) {
        int r1 = (c1 + 2 * c2 + c3) >> 2;
        int r2 = c3 - c2;
        int r3 = c1 - c2;
        c1 = r1;
        c2 = r2;
        c3 = r3;
    }
};

/** Specialization [transform = CT_IRREVERSIBLE] */
template<>
struct j2k_component_transform_forward<CT_IRREVERSIBLE> {
    /** Irreversible RGB -> YCbCr transform */
    static __device__ void perform(float & c1, float & c2, float & c3) {
        float r1 =  c1 * 0.29900f + c2 * 0.58700f + c3 * 0.11400f;
		float r2 = -c1 * 0.16875f - c2 * 0.33126f + c3 * 0.50000f;
		float r3 =  c1 * 0.50000f - c2 * 0.41869f - c3 * 0.08131f;
        c1 = r1;
        c2 = r2;
        c3 = r3;
    }
};

/**
 * Inverse component transformation
 *
 * @param transform
 */
template<enum j2k_component_transform transform>
struct j2k_component_transform_inverse {
    /** 
     * Default implementation do nothing [transform = CT_NONE]
     *
     * @param c1  First color component 
     * @param c2  Second color component 
     * @param c3  Third color component 
     */
    template<class data_type>
    static __device__ void perform(data_type & c1, data_type & c2, data_type & c3) {
    }
};

/** Specialization [transform = CT_REVERSIBLE] */
template<>
struct j2k_component_transform_inverse<CT_REVERSIBLE> {
    /** Reversible YUV -> RGB transform */
    static __device__ void perform(int & c1, int & c2, int & c3) {
        int r2 = c1 - ((c2 + c3) >> 2);
        int r1 = c3 + r2;
        int r3 = c2 + r2;
        c1 = r1;
        c2 = r2;
        c3 = r3;
    }
};

/** Specialization [transform = CT_IRREVERSIBLE] */
template<>
struct j2k_component_transform_inverse<CT_IRREVERSIBLE> {
    /** Irreversible YCbCr -> RGB transform */
    static __device__ void perform(float & c1, float & c2, float & c3) {
        float r1 = c1 * 1.00000f + c2 * 0.00000f + c3 * 1.40200f;
		float r2 = c1 * 1.00000f - c2 * 0.34413f - c3 * 0.71414f;
		float r3 = c1 * 1.00000f + c2 * 1.77200f + c3 * 0.00000f;
        c1 = r1;
        c2 = r2;
        c3 = r3;
    }
};

#endif // J2K_PREPROCESSOR_CT_H
