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

#ifndef MQC_GPU_REGISTER_H
#define MQC_GPU_REGISTER_H

#include "../mqc.h"

/**
 * Create gpu implementation of MQ-Coder
 *
 * @param configuration  Configuration of MQ-Coder
 * @return MQ-Coder handle or 0 if fails
 */
void*
mqc_gpu_register_create(struct mqc_configuration * configuration);

/**
 * Encode code-blocks by gpu implementation of MQ-Coder
 *
 * @param mqc  MQ-Coder handle
 * @param d_block  Array of definitions for blocks
 * @param block_count  Count of blocks in d_block
 * @param d_cxd  Array of input CX,D pairs
 * @param d_byte  Array of output bytes
 * @return 0 if OK, nonzero otherwise
 */
int
mqc_gpu_register_encode(
    void* mqc,
    struct j2k_cblk* d_cblk,
    int cblk_count,
    unsigned char * d_cxd, 
    unsigned char * d_byte
);

/**
 * Destroy gpu implementation of MQ-Coder
 * 
 * @param mqc  MQ-Coder handle
 * @return 0 if OK, nonzero otherwise
 */ 
int
mqc_gpu_register_destroy(void* mqc);

#endif // MQC_GPU_REGISTER_H
