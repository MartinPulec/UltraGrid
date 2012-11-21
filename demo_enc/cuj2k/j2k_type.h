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

#ifndef J2K_TYPE_H
#define J2K_TYPE_H

#include "j2k_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif





/**
 * Code-block style for the SPcod and SPcoc parameters
 * 
 * @see http://www.jpeg.org/public/fcd15444-1.pdf, page 31
 */
#define CBLK_STYLE_BYPASS    1
#define CBLK_STYLE_RESET     2
#define CBLK_STYLE_RESTART   4
#define CBLK_STYLE_CAUSAL    8
#define CBLK_STYLE_ERTERM   16
#define CBLK_STYLE_SEGMARK  32


/**
 * Band type string
 */
static const char j2k_band_type_string[4][3] = { "LL", "HL", "LH", "HH" };


#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif // J2K_TYPE_H

