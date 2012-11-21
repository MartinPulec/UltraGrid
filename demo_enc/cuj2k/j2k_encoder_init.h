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

#ifndef J2K_ENCODER_INIT_H
#define J2K_ENCODER_INIT_H

#include "j2k.h"

// /**
//  * Initialize GPU device
//  *
//  * @param device_id
//  * @param show_info  nonzero = print message if initialized OK
//  * @return 0 if OK, nonzero otherwise
//  */
// int
// j2k_encoder_init_device(int device_id, int show_info);

/**
 * Allocates buffers and initializes structures according to coding parameters
 * in given context
 * @param ctx  context with initialized coding parameters and without buffers 
 *             and uninitialized image strucutre
 * @return 0 if successfully initialized, nonzero otherwise
 */
/** Documented at declaration */
int
j2k_encoder_init_buffer(struct j2k_encoder* encoder);

/**
 * Dumps structure of the image.
 */
void
j2k_encoder_structure_dump(const struct j2k_encoder * const encoder);

/**
 * Frees buffers associated to given instance of J2K context.
 * @param ctx  pointer to instance of J2K context
 * @return  zero if successful, nonzero otherwise
 */
int
j2k_encoder_free_buffer(struct j2k_encoder * const encoder);

#endif // J2K_ENCODER_INIT_H

