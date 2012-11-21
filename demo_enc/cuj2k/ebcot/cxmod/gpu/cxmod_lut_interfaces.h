///
/// @file    cxmod_lut_interfaces.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Headers of functions for table lookup.
/// 


#ifndef CXMOD_LUT_INTERFACES_H
#define CXMOD_LUT_INTERFACES_H

#include "cxmod_device_types.h"

namespace cxmod_cuda {
    namespace gpu_luts {
        
        
        /// Sign coding table lookup.
        __device__ static u8 sc_lookup(const int sc_index);
        
        /// Lookup table for MRC and ZC in LL and LH bands.
        __device__ static u8 zc_and_mrc_ll_lh_lookup(const u32 lut_index);
        
        /// Lookup table for MRC and ZC in HL bands.
        __device__ static u8 zc_and_mrc_hl_lookup(const u32 lut_index);
        
        /// Lookup table for MRC and ZC in HH bands.
        __device__ static u8 zc_and_mrc_hh_lookup(const u32 lut_index);
        
    
    } // end of namespace gpu_luts
} // end of namespace cxmod_cuda

#endif // CXMOD_LUT_INTERFACES
