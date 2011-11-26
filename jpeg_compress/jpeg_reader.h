/**
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

#ifndef JPEG_READER
#define JPEG_READER

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/** JPEG decoder structure predeclaration */
struct jpeg_decoder;

/** JPEG reader structure */
struct jpeg_reader
{
};

/**
 * Create JPEG reader
 * 
 * @return reader structure if succeeds, otherwise NULL
 */
struct jpeg_reader*
jpeg_reader_create();

/**
 * Destroy JPEG reader
 * 
 * @param reader  Reader structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_reader_destroy(struct jpeg_reader* reader);

/**
 * Read JPEG image from data buffer
 * 
 * @param image  Image data
 * @param image_size  Image data size
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_reader_read_image(struct jpeg_decoder* decoder, uint8_t* image, int image_size);

#ifdef __cplusplus
} // END extern "C"
#endif

#endif // JPEG_WRITER
