/// 
/// @file    cxmod_border_init.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Initialization of border pixel groups.
///

#ifndef CXMOD_BORDER_INIT_H
#define CXMOD_BORDER_INIT_H

#include "cxmod_device_types.h"

namespace cxmod_cuda {

    
    /// Initializes border groups to expected values (zeros).
    /// @tparam CBLK_SX  width of codeblock
    /// @tparam CBLK_SY  height of codeblock
    /// @tparam GPT      number of pixel groups processed by each thread
    /// @param bits      buffer for magnitudes
    /// @param spps      buffer for SPP sigma precomputing
    /// @param xchg      buffer for signs and later for exchanges
    template <int CBLK_SX, int CBLK_SY, int GPT>
    __device__ static inline void clear_border_groups(u64 * const bits, 
                                                      u64 * const spps, 
                                                      u32 * const xchg) {
        // compile time constants
        enum {
            // number of groups along y axis
            NUM_GROUPS_Y = CBLK_SY / 4,
            
            // number of left/right border groups
            NUM_SIDE_BORDER_GROUPS = 3 + CBLK_SY / 4,
            
            // total number of threads in this threadblock
            NUM_THREADS = (CBLK_SX * CBLK_SY) / (GPT * 4),
            
            // offset of first bottom border group
            BOTTOM_BORDER_OFFSET = (1 + NUM_GROUPS_Y) * (1 + CBLK_SX) + 1,
            
            // offset of first top border group
            TOP_BORDER_OFFSET = 1
        };

        // initialize horizontal groups
        if(0 == threadIdx.y) {
            bits[TOP_BORDER_OFFSET + threadIdx.x] = 0;
            spps[TOP_BORDER_OFFSET + threadIdx.x] = 0;
            xchg[TOP_BORDER_OFFSET + threadIdx.x] = 0;
            bits[BOTTOM_BORDER_OFFSET + threadIdx.x] = 0;
            spps[BOTTOM_BORDER_OFFSET + threadIdx.x] = 0;
            xchg[BOTTOM_BORDER_OFFSET + threadIdx.x] = 0;
        }
        
        // initialize side border groups
        for (int grp_index = threadIdx.x + threadIdx.y * blockDim.x;
             grp_index < NUM_SIDE_BORDER_GROUPS;
             grp_index += NUM_THREADS) {
            const int idx = grp_index * (CBLK_SX + 1);
            bits[idx] = 0;
            spps[idx] = 0;
            xchg[idx] = 0;
        }
        
    }
    
    
} // end of namespace cxmod_cuda

#endif // CXMOD_BORDER_INIT_H
