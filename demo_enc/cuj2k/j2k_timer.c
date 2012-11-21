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

#include <cuda_runtime_api.h>
#include <sys/time.h>
#include <stdlib.h>
#include "j2k_timer.h"


/** Type of CPU timer. */
struct j2k_cpu_timer {
    /* time of last start call */
    double last_start;
    
    /* time of last stop */
    double last_stop;
};


/** Type of GPU timer. */
struct j2k_gpu_timer {
    /* start event */
    cudaEvent_t start_event;
    
    /* end event */
    cudaEvent_t end_event;
};


/**
 * @return current time in milliseconds
 */
static double get_time_ms() {
    struct timeval t;
    
    gettimeofday(&t, 0);
    return t.tv_sec * 1000.0 + t.tv_usec * 0.001;
}


/**
 * Creates new CPU timer
 * @return either new timer instance pointer or NULL for error
 */
struct j2k_cpu_timer * j2k_cpu_timer_create() {
    struct j2k_cpu_timer * t;
    
    if(t = (struct j2k_cpu_timer*)malloc(sizeof(struct j2k_cpu_timer))) {
        t->last_start = 0.0;
        t->last_stop = 0.0;
    }
    return t;
}


/**
 * Releases all resources associated with the timer instance.
 * @param cpu_timer  pointer to CPU timer instance or NULL to do nothing
 */
void j2k_cpu_timer_destroy(struct j2k_cpu_timer * cpu_timer) {
    if(cpu_timer) {
        free(cpu_timer);
    }
}


/**
 * Starts the time measurement.
 * @param cpu_timer  pointer to CPU timer instance or NULL to do nothing
 */
void j2k_cpu_timer_start(struct j2k_cpu_timer * cpu_timer) {
    if(cpu_timer) {
        cpu_timer->last_start = get_time_ms();
    }
}


/**
 * Stops the time measurement.
 * @param cpu_timer  pointer to CPU timer instance or NULL to do nothing
 */
void j2k_cpu_timer_stop(struct j2k_cpu_timer * cpu_timer) {
    if(cpu_timer) {
        cpu_timer->last_stop = get_time_ms();
    }
}


/**
 * Gets time in milliseconds between last start and stop.
 * @param cpu_timer  pointer to CPU timer instance or NULL to do nothing
 * @return  time in milliseconds between last instance start and stop
 *          or 0 if pointer is NULL
 */
double j2k_cpu_timer_time_ms(struct j2k_cpu_timer * cpu_timer) {
    return cpu_timer ? cpu_timer->last_stop - cpu_timer->last_start : 0.0;
}


/**
 * Creates new GPU timer
 * @return either new timer instance pointer or NULL for error
 */
struct j2k_gpu_timer * j2k_gpu_timer_create() {
    struct j2k_gpu_timer * t;
    
    if(t = (struct j2k_gpu_timer*)malloc(sizeof(struct j2k_gpu_timer))) {
        if(cudaSuccess != cudaEventCreate(&t->start_event)) {
            free(t);
            return 0;
        }
        if(cudaSuccess != cudaEventCreate(&t->end_event)) {
            cudaEventDestroy(t->start_event);
            free(t);
            return 0;
        }
    }
    return t;
}


/**
 * Releases all resources associated with the timer instance.
 * @param gpu_timer  pointer to instance of GPU timer or NULL to do nothing
 */
void j2k_gpu_timer_destroy(struct j2k_gpu_timer * gpu_timer) {
    if(gpu_timer) {
        cudaEventDestroy(gpu_timer->end_event);
        cudaEventDestroy(gpu_timer->start_event);
        free(gpu_timer);
    }
}


/**
 * Starts the time measurement.
 * @param gpu_timer  instance of GPU timer or NULL to do nothing
 * @param str  CUDA stream containing stuff to be measured
 */
void j2k_gpu_timer_start(struct j2k_gpu_timer * gpu_timer, cudaStream_t str) {
    if(gpu_timer) {
        cudaEventRecord(gpu_timer->start_event, str);
    }
}


/**
 * Stops the time measurement.
 * @param gpu_timer  instance of GPU timer or NULL to do nothing
 * @param str  CUDA stream containing stuff to be measured
 */
void j2k_gpu_timer_stop(struct j2k_gpu_timer * gpu_timer, cudaStream_t str) {
    if(gpu_timer) {
        cudaEventRecord(gpu_timer->end_event, str);
    }
}


/**
 * Gets time in milliseconds between last start and stop.
 * @param gpu_timer  pointer to instance of GPU timer or NULL to do nothing
 * @return number of milliseconds or 0 for NULL pointer
 */
double j2k_gpu_timer_time_ms(struct j2k_gpu_timer * gpu_timer) {
    float t = 0.0f;
    
    if(gpu_timer) {
        cudaEventSynchronize(gpu_timer->start_event);
        cudaEventSynchronize(gpu_timer->end_event);
        cudaEventElapsedTime(&t, gpu_timer->start_event, gpu_timer->end_event);
    }
    return t;
}

