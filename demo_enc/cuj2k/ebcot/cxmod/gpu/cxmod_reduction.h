///
/// @file    cxmod_reduction.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Stuff related to direct writing of reduced output CX,Ds.
///


#ifndef CXMOD_REDUCTION_H
#define CXMOD_REDUCTION_H

#include "cxmod_device_types.h"
#include "cxmod_prefix_sum.h"


namespace cxmod_cuda {
    
    
    /// Computes counts of CX,D pairs from given group in each of 3 passes.
    /// @tparam COMPLETE      true if codeblock is complete (no pixels out)
    /// @param flags          packed flags for 4 pixels (in/out)
    /// @param history        packed history of sigmas of the group
    /// @param c_xchg         group's exchanges
    /// @param l_xchg         group's left neighbor's exchanges
    /// @param r_xchg         group's right neighbor's exchanges
    /// @param spp_cxd_count  group's SPP CX,D count will be added to this
    /// @param mrp_cxd_count  group's MRP CX,D count will be added to this
    /// @param cup_cxd_count  group's CUP CX,D count will be added to this
    template <bool COMPLETE>
    __device__ static inline void cxd_count(const u16 flags,
                                            const u16 history,
                                            const u32 c_xchg,
                                            const u32 l_xchg,
                                            const u32 r_xchg,
                                            u16 & spp_cxd_count,
                                            u16 & mrp_cxd_count,
                                            u16 & cup_cxd_count) {
        // Not in RLC mode at the beginning of coding the group?
        // (Contains 0 iff group begins in RLC mode, 1 otherwise)
        u32 not_rlc = 0x08888820 & (r_xchg | c_xchg) | 0x08222220 & l_xchg;
        if(!COMPLETE) { not_rlc |= 0x1111 & flags; }
        not_rlc = not_rlc + 0x0FFFFFFF >> 28;
        
        // 1 iff groups starts in RLC mode, 0 othewise
        const u32 rlc = 1 - not_rlc;
        
        // packed magnitude bits for this group in this bitplane
        const u32 bits = c_xchg & 0x00111100;
        
        // 1 if any magnitude bit set, 0 otherwise
        const int bits_set = bits + 0x00FFFFFF >> 24;
        
        // contains 1 iff whole group not coded in RLC mode (=iff no rlc mode 
        // at all or if RLC mode terminated by some set magnitude bit)
        const u32 rlc_terminated = not_rlc | bits_set;
        
        // initialize CX,D count for CUP: either 1 CX,D iff no bits set and in 
        // RLC mode, or 2 CX,Ds iff RLC and some bits set or 0 if not in RLC
        cup_cxd_count += rlc * (1 + bits_set);
        
        // bits representing CX,D pairs from MRP
        const u32 mrp = 0x00444400 & c_xchg;
        
        // flags for determining whether prefered neighborhood is nonzero
        const u32 pn = (r_xchg & 0x04444480 | l_xchg & 0x04888880)
                     + (c_xchg & 0x0CCCCCC0) * 4;

        // bits (at same positions) for pixels coded in MRP or SPP
        u32 mrp_or_spp = 0;
        if(COMPLETE) {
            // first and last pixel at once
            mrp_or_spp = (pn & 0x15EDDE80) + 0x1FFDFF80 & 0x20020000;
        } else {
            // first pixel
            if(!(flags & 0x0001)) {
                mrp_or_spp = (pn & 0x0001DE80) + 0x0001FF80 & 0x00020000;
            }
            // last pixel
            if(!(flags & 0x1000)) {
                mrp_or_spp |= (pn & 0x15EC0000) + 0x1FFC0000 & 0x20000000;
            }
        }
        // second pixel
        if(COMPLETE || !(flags & 0x0010)) {
            mrp_or_spp |= (pn & 0x001DEC00) + 0x001FFC00 & 0x00200000;
        }
        // third pixel
        if(COMPLETE || !(flags & 0x0100)) {
            mrp_or_spp |= (pn & 0x01DEC000) + 0x01FFC000 & 0x02000000;
        }
        mrp_or_spp >>= 7;
        
        // bits at pixels which will be sign coded in this bitplane
        const int sigma_diff = c_xchg ^ (c_xchg << 1);
        
        // bits for pixels coded in SPP
        const u32 spp = mrp_or_spp ^ mrp;
        
        // sign codes in SPP
        const u32 spp_scs = spp & sigma_diff;
        
        // determine which pixels get coded in cup
        u32 cup = 0x00444400 ^ mrp_or_spp;
        
        // don't count pixels skipped with RLC
        const u32 rlc_coded_mask = (bits - 1 ^ bits) * rlc;
        
        // remove bits for pixels coded in RLC mode
        cup &= ~rlc_coded_mask;
        
        // possibly remove invalid pixels
        if(!COMPLETE) {
            cup &= u32(~flags) << 10;
        }
        
        // sign codes in CUP
        const u32 cup_scs = cup & sigma_diff;
        
        // finally count bits = CX,Ds (but only if not in RLC mode now)
        cup_cxd_count += rlc_terminated * __popc(cup_scs + 2 * cup);
        mrp_cxd_count += rlc_terminated * __popc(mrp);
        spp_cxd_count += rlc_terminated * __popc(spp_scs + 2 * spp);
    }



    /// Computes offsets of thread's CX,D outputs in each pass. Assumes that 
    /// thread processes 'GPT' consecutive groups starting with the one at 
    /// index #xchg_idx.
    /// @tparam CB_SX          width of codeblock in pixels
    /// @tparam CB_SY          height of codeblock in pixels
    /// @tparam COMPLETE       true if codeblock complete (all pixels in band)
    /// @tparam GPT            number of groups processed by each thread
    /// @param thread_idx      1D index of this thread
    /// @param temp            temporary buffer big enough for prefix sum
    /// @param scs             precomputed SCs for each thread's group
    /// @param flags           packed flags for each thread's group
    /// @param histories       sigma histories of all thread's groups 
    /// @param exchanges       buffer with exchanges of all groups
    /// @param xchg_idx        index of exchange of first thread's group
    /// @param offset          offset of first SPP CX,D pair
    /// @param spp_out_offset  (out) offset of thread's first SPP CX,D pair
    /// @param mrp_out_offset  (out) offset of thread's first MRP CX,D pair
    /// @param cup_out_offset  (out) offset of thread's first CUP CX,D pair
    /// @return  total number of all output CX,D pairs of all groups in 
    ///          all 3 passes of current bitplane
    template <int CB_SX, int CB_SY, bool COMPLETE, int GPT>
    __device__ static inline u16 output_offsets(const int idx,
                                                u64x2 * const temp,
                                                const u8x4 scs[GPT],
                                                const u16 flags[GPT],
                                                const u16 histories[GPT],
                                                const u32 * const exchanges,
                                                const int xchg_idx,
                                                const int offset,
                                                int & spp_out_offset,
                                                int & mrp_out_offset,
                                                int & cup_out_offset) {
        // compile time constants
        enum { NUM_ITEMS = (CB_SX * CB_SY) / (4 * GPT) };
        
        // count output CX,D pairs from all thread's groups in each pass
        u16x4 counts = {spp_out_offset, mrp_out_offset, cup_out_offset, 0};
        #pragma unroll
        for(int g = 0; g < GPT; g++) {
            const u32 c_xchg = exchanges[xchg_idx + g];
            const u32 l_xchg = exchanges[xchg_idx + g - 1];
            const u32 r_xchg = exchanges[xchg_idx + g + 1];
            cxd_count<COMPLETE>(flags[g], histories[g], c_xchg, l_xchg,
                                r_xchg, counts.x, counts.y, counts.z);
        }
        
        // get offsets
        const u16x4 offsets
                = packed_prefix_sum<NUM_ITEMS>(idx, counts, temp, offset);
        
        // distribute results to output variables and return total sum
        spp_out_offset = offsets.x;
        mrp_out_offset = offsets.y;
        cup_out_offset = offsets.z;
        return offsets.w;
    }
    
    
} // end of namespace cxmod_cuda


#endif // CXMOD_REDUCTION_H
