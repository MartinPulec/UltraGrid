/// 
/// @file    cxmod_sc_precomp.h
/// @author  Martin Jirman  (martin.jirman@cesnet.cz)
/// @brief   Precomputing of SC CX,D pairs for all pixels.
///

#ifndef CXMOD_SC_PRECOMP_H
#define CXMOD_SC_PRECOMP_H

#include "cxmod_device_types.h"
#include "cxmod_lut_interfaces.h"
#include "cxmod_util.h"

namespace cxmod_cuda {
    
    
    
    /// Treating given values each as 4 packed 16bit values, it returns 1s in 
    /// each value from 'a', which is greater than or equal to 
    /// corresponding value in 'b'. Other bits are zero.
    __device__ static inline u64 packed_ge_mask(const u64 a, const u64 b) {
        u64 result = 0;
        const u64 mask_3 = 0xFFFF000000000000LL;
        if((a & mask_3) >= (b & mask_3)) {
            result |= mask_3;
        }
        const u64 mask_2 = 0x0000FFFF00000000LL;
        if((a & mask_2) >= (b & mask_2)) {
            result |= mask_2;
        }
        const u64 mask_1 = 0x00000000FFFF0000LL;
        if((a & mask_1) >= (b & mask_1)) {
            result |= mask_1;
        }
        const u64 mask_0 = 0x000000000000FFFFLL;
        if((a & mask_0) >= (b & mask_0)) {
            result |= mask_0;
        }
        return result;
    }
    
    
    
    /// Treating given value as 4 packed 16bit unsigned numbers, it returns 
    /// 1 in place of each nonzero value and 0 in place of each zero value.
    __device__ static inline u64 packed_nonzero(const u64 value,
                                                const u64 msbs,
                                                const u64 lsbs) {
        return (((value & lsbs) + lsbs) | value) & msbs;
    }

    
    
    /// Precomputes SCs.
    /// @tparam CB_SX   codeblock width
    /// @param groups   pointer to topmost-leftmost non-border group
    template <int GRP_STRIDE_Y>
    __device__ static void precompute_scs(const u64 * const magnitudes_buffer, 
                                          const u64 * const spp_sigmas_buffer, 
                                          u32 * const scs_buffer,
                                          const int group_idx) {
        // Assumes that group->signs contains only 4 raw sign bits in 
        // each byte's lsb. Other bits must be 0.
        
        const u64 lsbs = 0x7FFF7FFF7FFF7FFFLL;
        const u64 msbs = ~lsbs;
        
        // Load packed magnitudes
        const u64 magnitudes = magnitudes_buffer[group_idx];
        
        // load packed sigma flags and extract packed sigmas after SPP
        const u64 sigma_flags = spp_sigmas_buffer[group_idx];
        const u64 spp_sigmas = sigma_flags & lsbs;
        
        // packed values of sigmas for all 4 pixels after each bitplane 
        const u64 sigmas = spp_sigmas | magnitudes;
        
        // create packed mask for determining significancy of neighbors 
        // of pixels in bitplanes, where pixels become significant
        const u64 signif_mask = ~((sigmas >> 1) & lsbs);
        
        // determine whether each pixel is sign coded in SPP or in CUP
        const u64 sc_in_spp = packed_ge_mask(spp_sigmas, magnitudes);
        const u64 sc_in_cup = ~sc_in_spp;
        const u64 sc_in_spp_without_msbs = sc_in_spp & lsbs;
        
        const u64 shift48 = 1LL << 48;
        const u64 shift16 = 1LL << 16;
        
        // load state of uppe rand lower neighboring pixels
        const u16 upper_magnitude = ((const u16*)(magnitudes_buffer + group_idx - GRP_STRIDE_Y))[3];
        const u16 lower_magnitude = ((const u16*)(magnitudes_buffer + group_idx + GRP_STRIDE_Y))[0];
        const u16 upper_spp_sigmas = ((const u16*)(spp_sigmas_buffer + group_idx - GRP_STRIDE_Y))[3];
        const u16 lower_spp_sigmas = ((const u16*)(spp_sigmas_buffer + group_idx + GRP_STRIDE_Y))[0];
        
        
        // load sign flags of neighbors
        const u64 upper_sigma_flags = upper_spp_sigmas + sigma_flags * shift16;
        const u64 left_sigma_flags = spp_sigmas_buffer[group_idx - 1];
        const u64 right_sigma_flags = spp_sigmas_buffer[group_idx + 1];
        const u64 lower_sigma_flags = (sigma_flags >> 16) + u64(lower_spp_sigmas) * shift48;
        const u64 lower_magnitudes = (magnitudes >> 16) + u64(lower_magnitude) * shift48;
        
        // load packed neighbor significancy:
        const u64 upper_signif_spp = upper_sigma_flags;
        const u64 upper_signif_cup = upper_magnitude + magnitudes * shift16;
        const u64 left_signif_spp = left_sigma_flags;
        const u64 left_signif_cup = magnitudes_buffer[group_idx - 1];
        const u64 right_signif_spp = magnitudes_buffer[group_idx + 1] >> 1;
        const u64 right_signif_cup = right_sigma_flags;
        const u64 lower_signif_spp = lower_magnitudes >> 1;
        const u64 lower_signif_cup = lower_sigma_flags;
        
        // select those neighbor significancy flags, which represent the pass 
        // in which each corresponding pixel is encoded
        const u64 signif_mask_without_msbs = signif_mask & lsbs;
        const u64 upper_signif = ((upper_signif_spp & sc_in_spp_without_msbs) | (upper_signif_cup & sc_in_cup)) & signif_mask;
        const u64 lower_signif = ((lower_signif_spp & sc_in_spp_without_msbs) | (lower_signif_cup & sc_in_cup)) & signif_mask_without_msbs;
        const u64 right_signif = ((right_signif_spp & sc_in_spp_without_msbs) | (right_signif_cup & sc_in_cup)) & signif_mask_without_msbs;
        const u64 left_signif = ((left_signif_spp & sc_in_spp_without_msbs) | (left_signif_cup & sc_in_cup)) & signif_mask;
        
        // obtain sign masks for neighboring pixels
        const u64 left_signs_raw = left_sigma_flags & msbs;
        const u64 right_signs_raw = right_sigma_flags & msbs;
        const u64 upper_signs_raw = upper_sigma_flags & msbs;
        const u64 lower_signs_raw = lower_sigma_flags & msbs;
        const u64 left_signs_neg = left_signs_raw - (left_signs_raw >> 15);
        const u64 right_signs_neg = right_signs_raw - (right_signs_raw >> 15);
        const u64 upper_signs_neg = upper_signs_raw >> 15;
        const u64 lower_signs_neg = lower_signs_raw >> 15;
        const u64 left_signs_pos = ~left_signs_neg;
        const u64 right_signs_pos = ~right_signs_neg;
        const u64 upper_signs_pos = ~upper_signs_neg;
        const u64 lower_signs_pos = ~lower_signs_neg;
        
        // In contrast to the standard, both vertical and horizontal 
        // neighbor sign contributions will be expressed as a number 
        // ranging from 0 to 4. Values 0 and 1 mean negative neighbors,
        // 2 means neutral and 3 and 4 mean positive neighbors in each 
        // direction. This will save some max/min and other operations.
        // Moreover, the horizontal contribution is scaled by 5 to get 
        // the final sign coding index in range 0-24 (both inclusive).
        u64 sc_indices = 0x000C000C000C000CLL; // 12 = no neighbor contribution
        
        // add contributions of those neighbors, which are significant
        // at the time of coding this pixel
        const u64 vertical_contrib_weight = 0x0005000500050005LL;
        const u64 upper_contrib = packed_nonzero(upper_signif, msbs, lsbs) >> 15;
        sc_indices += upper_contrib & upper_signs_pos;
        sc_indices -= upper_contrib & upper_signs_neg;
        const u64 lower_contrib = packed_nonzero(lower_signif, msbs, lsbs) >> 15;
        sc_indices += lower_contrib & lower_signs_pos;
        sc_indices -= lower_contrib & lower_signs_neg;
        u64 right_contrib = packed_nonzero(right_signif, msbs, lsbs);
        right_contrib -= right_contrib >> 15;
        right_contrib &= vertical_contrib_weight;
        sc_indices += right_contrib & right_signs_pos;
        sc_indices -= right_contrib & right_signs_neg;
        u64 left_contrib = packed_nonzero(left_signif, msbs, lsbs);
        left_contrib -= left_contrib >> 15;
        left_contrib &= vertical_contrib_weight;
        sc_indices += left_contrib & left_signs_pos;
        sc_indices -= left_contrib & left_signs_neg;
        
        // All indices are ready now - use them to obtain SCs
        u32 scs = 0;
        const u32 byte_mask = 0xFF;
        if(magnitudes & 0xFFFF) { // is pixel A nonzero?
            scs = gpu_luts::sc_lookup(u32(sc_indices) & byte_mask);
        }
        if(magnitudes & 0xFFFF0000) { // is pixel B nonzero?
            scs += gpu_luts::sc_lookup(u32((sc_indices >> 16) & byte_mask)) * (1 << 8);
        }
        if(magnitudes & 0xFFFF00000000LL) {
            scs += gpu_luts::sc_lookup(u32((sc_indices >> 32) & byte_mask)) * (1 << 16);
        }
        if(magnitudes & 0xFFFF000000000000LL) {
            scs += gpu_luts::sc_lookup(u32((sc_indices >> 48) & byte_mask)) * (1 << 24);
        }
        
        // Perform final XOR on all SCs at once, combine resulting SCs with raw
        // signs and in-flags and write it all back into 'signs' attribute of
        // the group.
        scs_buffer[group_idx] ^= scs;
    }
    

} // end of namespace cxmod_cuda

#endif // CXMOD_SC_PRECOMP_H

