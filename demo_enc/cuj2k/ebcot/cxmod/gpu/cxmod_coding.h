///
/// @file    cxmod_coding.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Coding core of context modeller.
///


#ifndef CXMOD_CODING_H
#define CXMOD_CODING_H

#include "cxmod_device_types.h"
#include "cxmod_lut_interfaces.h"


namespace cxmod_cuda {

    
    /// MRC and ZC table lookup wrappers.
    template <j2k_band_type ORIENTATION>
    __device__ static inline u8 cxd_lut(const u32 index);
    
    template <>
    __device__ static inline u8 cxd_lut<LL>(const u32 index) {
        return gpu_luts::zc_and_mrc_ll_lh_lookup(index);
    }
    
//     template <>
//     __device__ static inline u8 cxd_lut<LH>(const u32 index) {
//         return gpu_luts::zc_and_mrc_ll_lh_lookup(index);
//     }
    
    template <>
    __device__ static inline u8 cxd_lut<HL>(const u32 index) {
        return gpu_luts::zc_and_mrc_hl_lookup(index);
    }

    template <>
    __device__ static inline u8 cxd_lut<HH>(const u32 index) {
        return gpu_luts::zc_and_mrc_hh_lookup(index);
    }
    
    
    
    /// Composes lookup table index for SPP coding for the right pixxel.
    /// @param pixel_idx  index of the encoded pixel (0, 1, 2 or 3)
    /// @param sigmas_0   packed state of sigmas before SPP
    /// @param sigmas_s   packed state of sigmas after SPP
    /// @return combined states representing prefered neighborhood of the pixel
    __device__ static inline u32 spp_lut_idx(const int pixel_idx,
                                             const u32 sigmas_0,
                                             const u32 sigmas_s) {
        if(0 == pixel_idx) {
            return (sigmas_0 & 0x0000A200) | (sigmas_s & 0x000045E0);
        } else if(1 == pixel_idx) {
            return (sigmas_0 & 0x000A2200) | (sigmas_s & 0x00045C00);
        } else if(2 == pixel_idx) {
            return (sigmas_0 & 0x00A22000) | (sigmas_s & 0x0045C000);
        } else {
            return (sigmas_0 & 0x0E220000) | (sigmas_s & 0x005C0000);
        }
    }
    
    
    
    /// Composes lookup table index for coding some pixel in CUP.
    /// @param pixel_idx  index of the pixel (0, 1, 2 or 3)
    /// @param sigmas_s   packed state of sigmas before CUP
    /// @param sigmas_1   packed state of sigmas after CUP
    /// @return combined states containing sigmas in pixel's neighborhood
    __device__ static inline u32 cup_lut_idx(const int pixel_idx,
                                             const u32 sigmas_s,
                                             const u32 sigmas_1) {
        if(0 == pixel_idx) {
            return (sigmas_s & 0x0000A300) | (sigmas_1 & 0x000044E0);
        } else if(1 == pixel_idx) {
            return (sigmas_s & 0x000A3200) | (sigmas_1 & 0x00044C00);
        } else if(2 == pixel_idx) {
            return (sigmas_s & 0x00A32000) | (sigmas_1 & 0x0044C000);
        } else {
            return (sigmas_s & 0x0E320000) | (sigmas_1 & 0x004C0000);
        }
    }
    
    
    
    /// Composes lookup table index for MRP.
    /// @param pixel_idx  index of coded pixel
    /// @param sigmas_0   packed state of old pixel sigmas
    /// @param sigmas_s   packed state of sigmas before MRP
    /// @return packed state of sigmas needed for MRC
    __device__ static inline u32 mrp_lut_idx(const int pixel_idx,
                                             const u32 sigmas_0,
                                             const u32 sigmas_s) {
        const u32 mask_0 = 0x1800 << (pixel_idx * 4);
        const u32 mask_s = 0xE7E0 << (pixel_idx * 4);
        return (sigmas_0 & mask_0) | (sigmas_s & mask_s);
    }


    
    /// Composes output code of one pixel.
    /// @tparam PX_IDX         pixel index (relative to group: 0, 1, 2 or 3)
    /// @tparam BAND           codeblock's origin band oriantation
    /// @param output          pointer to output buffer for all CX,D pairs
    /// @param spp_out_offset  offset of next SPP CX,D pair in the buffer
    /// @param mrp_out_offset  offset of next MRP CX,D pair in the buffer
    /// @param cup_out_offset  offset of next CUP CX,D pair in the buffer
    /// @param sc              preencoded sign code for the pixel
    /// @param rlc_mode        true if in RLC mode (may be changed here)
    /// @param invld           packed invalid flags for all 4 pixels
    /// @param sigmas_0        packed state of sigmas of the group before SPP
    /// @param sigmas_s        packed state of sigmas of the group after SPP
    /// @param sigmas_1        packed state of sigmas of the group after CUP
    /// @param sigma_diff      sigma difference before and after bitplane
    template <int PX_IDX, j2k_band_type BAND>
    __device__ static inline void encode_pixel(u8 * const output,
                                               int & spp_out_offset,
                                               int & mrp_out_offset,
                                               int & cup_out_offset,
                                               const u8 sc,
                                               bool & rlc_mode,
                                               const u16 invld,
                                               const u32 sigmas_0,
                                               const u32 sigmas_s,
                                               const u32 sigmas_1, 
                                               const u32 sigma_diff) {
        // some compile time masks:
        enum { 
            SIGMA_BIT = 0x0800 << (PX_IDX * 4), ///< sigma bit of pixel
            VALUE_BIT = 0x0100 << (PX_IDX * 4), ///< value bit of pixel
            PREF_NBHD_BITS = 0xE6E0 << (PX_IDX * 4), ///< bits of pref neighb
            INVALID_BIT = 0x0001 << (PX_IDX * 4)  ///< pixel's in/out bit
        };
        
        // Is group encoder still in run-length mode?
        if(rlc_mode) {
            // If value of encoded pixel is 1 ...
            if(sigmas_s & VALUE_BIT) {
                // terminate run-length mode
                rlc_mode = false;
                
                // put RLC,1 + UNI,x + UNI,y into pixel's output byte
                // (xy is 2bit number representing index of pixel)
                output[0 + cup_out_offset] = 0x47;
                output[1 + cup_out_offset] = (PX_IDX & 2) ? 0x4B : 0x4A;
                output[2 + cup_out_offset] = (PX_IDX & 1) ? 0x4B : 0x4A;
                output[3 + cup_out_offset] = sc;
                
                // advance CUP out offset
                cup_out_offset += 4;
            }
        } else if (!(invld & INVALID_BIT)) { 
            // (else if not in run length mode and not invalid)
            // will contain index to lookup table for MRC or ZC CX,D pair
            u32 lut_index;
            
            // will contait destination index for the MRC or ZC pair
            // (initially assume that pixel is coded in MRC)
            int cxd_out_offset = mrp_out_offset;
            
            // true if sign code will be written into the output
            const bool sc_written = bool(SIGMA_BIT & sigma_diff);
            
            // In which pass will the pixel be encoded?
            // Is it in MRP? Or - is the sigma of the pixel set?
            if(sigmas_0 & SIGMA_BIT) {
                // compose index to lookup table for MRP
                lut_index = mrp_lut_idx(PX_IDX, sigmas_0, sigmas_s);
                
                // advance MRP output offset
                mrp_out_offset++;
            } else {
                // number of CX,D pairs written
                const int num_cxds_written = sc_written ? 2 : 1;
                
                // compose LUT index as if the pixel was encoded in SPP
                // It will contain all info needed to determine whether
                // the pixel will be encoded in SPP.
                lut_index = spp_lut_idx(PX_IDX, sigmas_0, sigmas_s);
                
                // if any sigma belonging to prefered neighborhood is 1
                if(lut_index & PREF_NBHD_BITS) {
                    // encoding in SPP - index is ready, set the output:
                    cxd_out_offset = spp_out_offset;
                    spp_out_offset += num_cxds_written;
                } else {
                    // set output index and advance it
                    cxd_out_offset = cup_out_offset;
                    cup_out_offset += num_cxds_written;
                    
                    // encoding in CUP - discard the index and make new
                    lut_index = cup_lut_idx(PX_IDX, sigmas_s, sigmas_1);
                }
            }
            
            // use the index to look up encoded CX,D pair
            lut_index >>= PX_IDX * 4 + 5; // align to bit #0
            
            // put the CX,D pair into the right place
            output[cxd_out_offset] = cxd_lut<BAND>(lut_index);
            
            // possibly add sign
            if(sc_written) {
                output[1 + cxd_out_offset] = sc;
            }
        }
    }
    
    
    
    /// Reduced encoding of one pixel group.
    /// @tparam BAND           orientation of codeblock's origin band
    /// @tparam COMPLETE       true if codeblock is complete (no pixels out)
    /// @param output          pointer to output buffer for all CX,D pairs
    /// @param flags           packed flags of the group
    /// @param history         packed history of the group
    /// @param scs             preencoded sign codes of the group
    /// @param c_xchg          encoded group's packed state
    /// @param l_xchg          group's left neighbor's packed state
    /// @param r_xchg          group's right neighbor's packed state
    /// @param spp_out_offset  output offset of group's first SPP CX,D pair
    /// @param mrp_out_offset  output offset of group's first MRP CX,D pair
    /// @param cup_out_offset  output offset of group's first CUP CX,D pair
    template <j2k_band_type BAND, bool COMPLETE>
    __device__ static inline void encode_group(u8 * const output,
                                               const u16 flags,
                                               const u16 history,
                                               const u8x4 scs,
                                               const u32 c_xchg,
                                               const u32 l_xchg,
                                               const u32 r_xchg,
                                               int & spp_out_offset,
                                               int & mrp_out_offset,
                                               int & cup_out_offset) {
        // unpack invalidity of pixels (in/out of the band)
        // if codeblock is complete, all pixels are valid
        const u16 invalid = COMPLETE ? 0 : (flags & 0x1111);
            
        // if this group has at least one pixel in the band
        if(invalid ^ 0x1111) {
            // combine neighboring exchanges to get all encoding info
            const u32 lxs = l_xchg >> 1;
            const u32 rxs = r_xchg >> 2;
            const u32 sigmas_0 = 2 * (rxs & 0x01111100 | lxs & 0x02000000 |
                                      c_xchg & 0x04000000)
                               + 512 * (history & 0xCCCC);
            const u32 sigmas_s = lxs & 0x04444440 | c_xchg & 0x08999980 |
                                 rxs & 0x02222220;
            const u32 sigmas_1 = 4 * (lxs & 0x00111110 | rxs & 0x00000008 |
                                      c_xchg & 0x00222220);

            // True if coding in run-lenght mode. Depends on:
            // 1) no pixel is out (invalid) = all pixels are valid
            // 2) sigmas after SPP of right, central and lower pixels are 0
            // 3) sigmas after CUP of upper and left pixels are 0
            bool rlc_mode = !(invalid | sigmas_s & 0x0EAAAA00
                                      | sigmas_1 & 0x004444E0);
            
            // if in run lenght mode and all values are 0, then RLC,0 
            // is output of this group in clean-up pass
            if(rlc_mode && !(sigmas_s & 0x00111100)) {
                output[cup_out_offset++] = 0x46; // 0x46 = RLC,0 = 17.0
            } else {
                // packed difference between sigmas before and after bitplane
                const u32 sgm_diff = sigmas_1 ^ sigmas_0;
                
                // encode all 4 pixels
                encode_pixel<0, BAND>(output, spp_out_offset, mrp_out_offset,
                                      cup_out_offset, scs.x, rlc_mode, invalid,
                                      sigmas_0, sigmas_s, sigmas_1, sgm_diff);
                encode_pixel<1, BAND>(output, spp_out_offset, mrp_out_offset,
                                      cup_out_offset, scs.y, rlc_mode, invalid,
                                      sigmas_0, sigmas_s, sigmas_1, sgm_diff);
                encode_pixel<2, BAND>(output, spp_out_offset, mrp_out_offset,
                                      cup_out_offset, scs.z, rlc_mode, invalid,
                                      sigmas_0, sigmas_s, sigmas_1, sgm_diff);
                encode_pixel<3, BAND>(output, spp_out_offset, mrp_out_offset,
                                      cup_out_offset, scs.w, rlc_mode, invalid,
                                      sigmas_0, sigmas_s, sigmas_1, sgm_diff);
            }
        }
    }

    

} // end of namespace cxmod_cuda

#endif // CXMOD_CODING_H

