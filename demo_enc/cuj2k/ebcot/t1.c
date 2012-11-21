/* 
 * Copyright (c) 2009, Jiri Matela
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

#include <stdlib.h>
#include <assert.h>
#include "../common.h"
#include "t1.h"
#include "cxmod/gpu/cxmod_interface.h"
#include "mqc/mqc.h"
#include "../dwt/gpu/dwt.h"

/** Documented at declaration */
int
j2k_t1_encode(struct j2k_encoder* encoder)
{
//     // data of timer (if timing info should be shown)
//     struct cuda_timer_t timer;
//     if ( encoder->params.print_info ) { timer = cuda_timer_start(); }
//     
//     // Run context-modeller
//     int result = cxmod_encode(
//         encoder->cxmod,
//         encoder->cblk_count,
//         encoder->d_cblk,
//         encoder->band,
//         (int*)encoder->d_data,
//         encoder->d_cxd,
//         NULL
//     );
//     if ( result != 0 )
//         return -1;
//     
//     // possibly show info
//     if ( encoder->params.print_info ) {
//         printf("Tier-1 CXMOD:      %f ms\n", cuda_timer_stop(timer));
//         timer = cuda_timer_start();
//     }
// 
//     // Run MQ-Coder
//     result = mqc_encode(
//         encoder->mqc, 
//         encoder->cblk_count, 
//         encoder->d_cblk, 
//         encoder->d_cxd, 
//         encoder->d_byte,
//         encoder->d_trunc_sizes
//     );
//     if ( result != 0 )
//         return -2;
//     
//     // possibly show info
//     if ( encoder->params.print_info ) {
//         printf("Tier-1 MQ-Coder:   %f ms\n", cuda_timer_stop(timer));
//     }
    
    return 0;
}
