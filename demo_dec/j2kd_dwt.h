/**
 * @file    j2kd_dwt.h (originally dwt.h)
 * @author  Martin Jirman (207962@mail.muni.cz)
 * @brief   Interface for CUDA implementaion of 9/7 and 5/3 reverse DWT.
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
 */


#ifndef J2KD_DWT_H
#define J2KD_DWT_H


#include "j2kd_type.h"
#include "j2kd_image.h"
#include "j2kd_cuda.h"


namespace cuj2kd {


/// High level DWT implementation for JPEG 2000 decoder.
class DWT {
private:
    /// Buffer for correctly mirrored line pointers and coordinates
    GPUBuffer mirror;
    
public:
    /// Standard constructor - configures DWT kernels
    DWT();
    
    /// Transforms all bands in all tiles in given images in given stream.
    void transform(
        Image & image,
        IOBufferGPU<u8> & working,
        GPUBuffer & tempBuffer,
        cudaStream_t stream
    );
}; // end of class DWT


} // end of namespace cuj2kd


#endif // J2KD_DWT_H

