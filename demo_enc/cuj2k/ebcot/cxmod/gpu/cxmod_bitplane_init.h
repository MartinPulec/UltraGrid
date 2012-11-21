///
/// @file    cxmod_bitplane_init.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Updating state of codeblock before encoding each bitplane.
///

#ifndef CXMOD_BITPLANE_INIT_H
#define CXMOD_BITPLANE_INIT_H

#include "cxmod_device_types.h"
#include "cxmod_util.h"

namespace cxmod_cuda {
    
    
    /// Adds uper and lower exchanges to given group's exchange.
    /// @tparam G_STRIDE_Y  difference between indices of two vertically 
    ///                     neighboring exchanges
    /// @param exchanges    buffer with all exchanges
    /// @param group_idx    index of group whose exchange is modified
    /// @return combined exchange of the group
    template <int G_STRIDE_Y>
    __device__ static inline u32 combine_exchanges(u32 * const exchanges,
                                                   const int group_idx) {
        const u32 upper_xchg = exchanges[group_idx - G_STRIDE_Y];
        const u32 lower_xchg = exchanges[group_idx + G_STRIDE_Y];
        return exchanges[group_idx] |= (upper_xchg >> 16) | (lower_xchg << 16);
    }
    
    
    
    /// Updates history and initializes exchange of some group.
    /// @tparam COMPLETE   true if codeblock is complete (all pixels in band)
    /// @param exchanges   buffer with exchanges of all groups
    /// @param magnitudes  buffer with magnitudes of all groups
    /// @param group_idx   index of group's magnitude and exchange in buffers
    /// @param bitplane    number of current bitplane
    /// @param flags       packed group's flags (info about 4 group's pixels)
    /// @param history     packed history of state of sigmas of group's pixels
    /// @return  new state of group's exchange
    template <bool COMPLETE>
    __device__ static inline u32 update_group(u32 * const exchanges,
                                              const u64 * const magnitudes,
                                              const int group_idx,
                                              const int bitplane,
                                              const u16 flags,
                                              u16 & history) {
        // index and shift for getting right magnitude bits
        const int bits_shift = (bitplane >> 2) * 16 + (bitplane & 3);
        
        // new state of group's exchange
        u32 exchange = 0;
        
        // skip the group if completely out of the band (all 'out' flags == 1)
        if(COMPLETE || (0x1111 & ~flags)) {
            // get packed pixel values for current bitplane
            const u16 bits = 0x1111 & u16(magnitudes[group_idx] >> bits_shift);
            
            // update history of sigmas of the group ...
            //          after:     before:
            //    1) sigma'                <-- old sigma before bpln.
            //    2) sigma before bpln.    <-- ald sigma after bpln.
            //    3) sigma after bitplane  <-- old sigma after OR magnitude bit
            //    4) current magnitude bit <-- magnitude bit
            // ... and get sigmas after spp:
            //    5) sigma after SPP == 1 iff:
            //       5a) either both new sigma after bpln. and sigma diff are 1 
            //       5b) or sigma already was 1 before the bitplane
            u16 spp_sigmas = history;
            history = (history & 0x6666) | bits;
            history |= history << 1;
            spp_sigmas |= history & flags;
        
            // exchange for the group: for each pixel, get Z0, Z1 and Zv 
            // from history:
            exchange = u32(history & 0x7777) << 8;
            // ... and Zs from spp_sigmas:
            exchange += u32(spp_sigmas & 0x2222) * (1 << 10);
        }
        
        // publish exchange into the shared memory for neighboring groups
        return exchanges[group_idx] = exchange;
    }
    
    
} // end of namespace cxmod_cuda

#endif // CXMOD_BITPLANE_INIT_H
