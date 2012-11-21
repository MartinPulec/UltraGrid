/* 
 * Copyright (c) 2011, Martin Jirman (martin.jirman@cesnet.cz)
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

#ifndef J2K_TIMER_H
#define J2K_TIMER_H

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif



/** Type of CPU timer. */
struct j2k_cpu_timer;


/** Type of GPU timer. */
struct j2k_gpu_timer;


/**
 * Creates new CPU timer
 * @return either new timer instance pointer or NULL for error
 */
struct j2k_cpu_timer * j2k_cpu_timer_create();


/**
 * Releases all resources associated with the timer instance.
 * @param cpu_timer  pointer to CPU timer instance or NULL to do nothing
 */
void j2k_cpu_timer_destroy(struct j2k_cpu_timer * cpu_timer);


/**
 * Starts the time measurement.
 * @param cpu_timer  pointer to CPU timer instance or NULL to do nothing
 */
void j2k_cpu_timer_start(struct j2k_cpu_timer * cpu_timer);


/**
 * Stops the time measurement.
 * @param cpu_timer  pointer to CPU timer instance or NULL to do nothing
 */
void j2k_cpu_timer_stop(struct j2k_cpu_timer * cpu_timer);


/**
 * Gets time in milliseconds between last start and stop.
 * @param cpu_timer  pointer to CPU timer instance or NULL to do nothing
 * @return  time in milliseconds between last instance start and stop
 *          or 0 if pointer is NULL
 */
double j2k_cpu_timer_time_ms(struct j2k_cpu_timer * cpu_timer);


/**
 * Creates new GPU timer
 * @return either new timer instance pointer or NULL for error
 */
struct j2k_gpu_timer * j2k_gpu_timer_create();


/**
 * Releases all resources associated with the timer instance.
 * @param gpu_timer  pointer to instance of GPU timer or NULL to do nothing
 */
void j2k_gpu_timer_destroy(struct j2k_gpu_timer * gpu_timer);


/**
 * Starts the time measurement.
 * @param gpu_timer  instance of GPU timer or NULL to do nothing
 * @param stream  CUDA stream containing stuff to be measured
 */
void j2k_gpu_timer_start(struct j2k_gpu_timer * gpu_timer, cudaStream_t stream);


/**
 * Stops the time measurement.
 * @param gpu_timer  instance of GPU timer or NULL to do nothing
 * @param stream  CUDA stream containing stuff to be measured
 */
void j2k_gpu_timer_stop(struct j2k_gpu_timer * gpu_timer, cudaStream_t stream);


/**
 * Gets time in milliseconds between last start and stop.
 * @param gpu_timer  pointer to instance of GPU timer or NULL to do nothing
 * @return number of milliseconds or 0 for NULL pointer
 */
double j2k_gpu_timer_time_ms(struct j2k_gpu_timer * gpu_timer);



#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* J2K_TIMER_H */

