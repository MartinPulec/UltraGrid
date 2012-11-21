///
/// @file    cxmod_util.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Common utility functions for context modeller.
///

#ifndef CXMOD_UTIL_H
#define CXMOD_UTIL_H

#include "cxmod_device_types.h"

namespace cxmod_cuda {
    
    
    /// Divide and round up.
    __device__ __host__ inline int div_rnd_up(const int dividend,
                                              const int divisor) {
        return (dividend / divisor) + ((dividend % divisor) ? 1 : 0);
    }
    
    
    /// Compile time binary operations template.
    template<int X, int Y> struct COMPILE_TIME {
        enum {
            MIN = (X < Y) ? X : Y,
            MAX = (X > Y) ? X : Y,
            NUM_GROUPS = (X + 1) * (Y / 4 + 2) + 1,
            SHMEM_BYTES = NUM_GROUPS * 20,
            DIV_RND_UP = (X + Y - 1) / Y
        };
    };


} // end of namespace cxmod_cuda

#endif // CXMOD_UTIL_H
