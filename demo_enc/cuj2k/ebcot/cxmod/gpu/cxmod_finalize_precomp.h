///
/// @file    cxmod_finalize_precomp.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Transfroms shared memory state of codeblock for easier encoding.
///


#ifndef CXMOD_FINALIZE_PRECOMP_H
#define CXMOD_FINALIZE_PRECOMP_H

#include "cxmod_device_types.h"


namespace cxmod_cuda {
    
    
    /// Reorders magnitude bits so that each 16 consecutive bits are related 
    /// to 4 bitplanes and each 4 consecutive bits belong to one pixel.
    /// @param bits  raw magnitude bits
    /// @return  reordered magnitude bits (permutation of input)
    __device__ static inline u64 reorder_bits(const u64 bits) {
        // select some bits from each shift - whole output is covered this way
        u64 reordered = (0xF0000F0000F0000FLL & bits);
        reordered    |= (0x0F0000F0000F0000LL & bits) >> 12;
        reordered    |= (0x00F0000F00000000LL & bits) >> 24;
        reordered    |= (0x000F000000000000LL & bits) >> 36;
        reordered    += (0x000000000000F000LL & bits) * (1LL << 36);
        reordered    += (0x00000000F0000F00LL & bits) * (1LL << 24);
        reordered    += (0x0000F0000F0000F0LL & bits) * (1LL << 12);
        return reordered;
    }
    
    
    
    /// Composes flags which include (for one group):
    ///  - difference between sigma after and before SPP in bitplane, where 
    ///    corresponding pixel is sign coded
    ///  - for incomplete codeblocks also bits representing validity of pixels
    /// @tparam COMPLETE   is codeblock is complete? (All pixels in band?)
    /// @param magnitudes  packed raw magnitudes
    /// @param spp_sigmas  packed raw SPP sigmas
    /// @param sign_flags  packed sign flags (containing in/out info)
    /// @param lsbs        0s at positions before multiples of 16, 1s elsewhere
    /// @return packed flags for 4 pixels of some group
    template <bool COMPLETE>
    __device__ static inline u16 get_flags(const u64 magnitudes,
                                                const u64 spp_sigmas,
                                                const u32 sign_flags,
                                                const u64 lsbs) {
        // Represents 4 packed 16bit value, each is nonzero iff corresponding 
        // pixel is NOT sign coded in SPP.
        const u64 not_spp = (spp_sigmas | magnitudes) ^ spp_sigmas;
        
        // 4 packed 16 bit values - 2nd bit of each 16bit value is 1 iff 
        // corresponding pixel is sign coded in SPP. Other bits are 0.
        const u64 in_spp = ~((not_spp & lsbs) + lsbs | lsbs | not_spp) >> 14;
        
        // Simmilar as above, but reordered into 32bits (pixel order is DBCA).
        // 2nd bit of each byte is 1 iff corresponding pixel is sign coded 
        // in SPP. Other pixels are 0.
        u32 flags = u32(in_spp | in_spp >> 24);
        
        // Possibly add 'out flags' for all 4 pixels. (Representing whether 
        // corresponding pixel is in the band or outside the band and thus 
        // should not be encoded.) If codeblock is known to be complete, flags 
        // aren't used. Following bits are added to flags:
        //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        //   | | | | | | | |d| | | | | | | |b| | | | | | | |c| | | | | | | |a|,
        //   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        // where bits d, b, c and a are 1 iff corrsponding pixels are out 
        // of the band, zero otherwise. Other pixels are left unaffected.
        if(!COMPLETE) {
            flags |= 0x01010101 & ~__byte_perm(sign_flags >> 1, 0, 0x3120);
        }
        
        // Return SPP sigma diffs and in/out flags in DCBA pixel order 
        // and at required bit positions.
        return u16(flags | flags >> 12);
    }
    
    
    
    /// Reorders magnitude bits and composes flags for all thread's groups.
    /// @param magnitudes       raw magnitudes of all groups in shared memory
    /// @param spps             raw SPP sigmas of all groups in shared memory
    /// @param scs              precomputed SCs of all groups in shared memory
    /// @param first_group_idx  index of first group's stuff in shared buffers
    /// @param flags_out        registers output for flags of thread's groups
    /// @param scs_out          registers output for SCs of thread's groups
    template <bool COMPLETE, int GPT>
    __device__ static inline void finalize(u64 * const magnitudes, 
                                           const u64 * const spps, 
                                           const u32 * const scs,
                                           const int first_group_idx,
                                           u16 flags_out[GPT],
                                           u8x4 scs_out[GPT]) {
        const u64 lsbs = 0x7FFF7FFF7FFF7FFFLL;
        
        #pragma unroll
        for(int g = 0; g < GPT; g++) {
            // load magnitude bits, reorder them and same them
            const u64 bits = magnitudes[first_group_idx + g];
            magnitudes[first_group_idx + g] = reorder_bits(bits);
            
            // load raw SPP sigmas and clear raw signs in msbs
            const u64 spp_sigmas = spps[first_group_idx + g] & lsbs;
            
            // copy precomputed SCs into registers before overwriting them
            scs_out[g] = ((u8x4*)scs)[first_group_idx + g];
            
            // compose flags including pixel invalidity and SPP sigma diff
            const u32 invld = scs[first_group_idx + g];
            flags_out[g] = get_flags<COMPLETE>(bits, spp_sigmas, invld, lsbs);
        }
    }
    
    
} // end of namespace cxmod_cuda

#endif // CXMOD_FINALIZE_PRECOMP_H
