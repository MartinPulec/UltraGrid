// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob
/**
 * @file
 * macOS doesn't have pthread_condattr_setclock() therefore CLOCK_MONOTONIC
 * cannot be set. As CLOCK_REALTIME is the default, it is not optimal
 * (eventually non-continuous), use workaround on macOS using sleep with timeout
 * in relative time (duration).
 */
#include "utils/pthread_cond.h"

#include <errno.h>   // for ETIMEDOUT
#include <stdio.h>   // for perror
#include <stdlib.h>  // for abort
#include <time.h>    // for timespec, CLOCK_MONOTONIC, clock_gettime

#include "tv.h"      // for NS_IN_SEC

void
ug_pthread_cond_init(pthread_cond_t *cv)
{
        pthread_condattr_t attr;
        pthread_condattr_init(&attr);
#ifndef __APPLE__
        pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
#endif
        int ret = pthread_cond_init(cv, &attr);
        if (ret != 0) {
                perror(__func__);
                abort();
        }
        pthread_condattr_destroy(&attr);
}


int
ug_pthread_cond_timedwait(pthread_cond_t *cv, pthread_mutex_t *lock,
                          time_ns_t *timeout_ns)
{
        struct timespec tmout = { 0, 0 };
#ifndef __APPLE__
        clock_gettime(CLOCK_MONOTONIC, &tmout);
#endif
        unsigned long long nsec = tmout.tv_nsec + *timeout_ns;
        tmout.tv_sec += nsec / NS_IN_SEC;
        tmout.tv_nsec = nsec % NS_IN_SEC;
#ifdef __APPLE__
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int ret = pthread_cond_timedwait_relative_np(cv, lock, &tmout);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        // relative time on macOS - decrease for the case of spurious wake-up
        *timeout_ns -=
            (t0.tv_sec - t1.tv_sec) * NS_IN_SEC + (t1.tv_nsec - t0.tv_nsec);
#else
        int ret = pthread_cond_timedwait(cv, lock, &tmout);
#endif
        if (ret != 0 && ret != ETIMEDOUT) {
                perror(__func__);
        }
        return ret;
}

