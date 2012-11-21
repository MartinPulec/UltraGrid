///
/// @file    cxmod_specialized.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Accessing specialized instantces of context modeller.
///

#ifndef CXMOD_SPECIALIZED_H
#define CXMOD_SPECIALIZED_H

#include "cxmod_device_types.h"
#include "cxmod_kernel.h"

namespace cxmod_cuda {
    

    /// Sets L1 cache for specialized kernel and returns its launcher.
    template<int CB_SX, int CB_SY>
    cxmod_launcher_fn_t set_cache_and_get_launcher() {
        // compile time: number of groups per thread
        enum { GPT = 1 << (CB_SX * CB_SY > 256) };
        cxmod_set_cache<CB_SX, CB_SY, GPT>();
        return cxmod_launcher<CB_SX, CB_SY, GPT>;
    }
    
    
    
    /// Gets pointer to specialized context modeller launcher function.
    /// @param cblk_sx  width of codeblock
    /// @param cblk_sy  height of codeblock
    /// @return  pointer to function for launching specified context modeller
    ///          kernels or null if some parameters invalid/unsupported
    inline cxmod_launcher_fn_t cxmod_find_launcher(const int cblk_sx,
                                                   const int cblk_sy) {
        #warning "many valid (but weird) codeblock dimensions commented out"
        switch(cblk_sx) {
            //case 4:
            //   switch(cblk_sy) {
            //       case 4:    return set_cache_and_get_launcher<4, 4>();
            //        case 8:    return set_cache_and_get_launcher<4, 8>();
            //        case 16:   return set_cache_and_get_launcher<4, 16>();
            //        case 32:   return set_cache_and_get_launcher<4, 32>();
            //        case 64:   return set_cache_and_get_launcher<4, 64>();
            //        case 128:  return set_cache_and_get_launcher<4, 128>();
            //        case 256:  return set_cache_and_get_launcher<4, 256>();
            //       case 512:  return set_cache_and_get_launcher<4, 512>();
            //        case 1024: return set_cache_and_get_launcher<4, 1024>();
            //   }
            //   break;
            //case 8:
            //    switch(cblk_sy) {
            //        case 4:   return set_cache_and_get_launcher<8, 4>();
            //        case 8:   return set_cache_and_get_launcher<8, 8>();
            //        case 16:  return set_cache_and_get_launcher<8, 16>();
            //        case 32:  return set_cache_and_get_launcher<8, 32>();
            //        case 64:  return set_cache_and_get_launcher<8, 64>();
            //        case 128: return set_cache_and_get_launcher<8, 128>();
            //        case 256: return set_cache_and_get_launcher<8, 256>();
            //        case 512: return set_cache_and_get_launcher<8, 512>();
            //    }
            //    break;
            case 16:
                switch(cblk_sy) {
                    //case 4:   return set_cache_and_get_launcher<16, 4>();
                    //case 8:   return set_cache_and_get_launcher<16, 8>();
                    case 16:  return set_cache_and_get_launcher<16, 16>();
                    case 32:  return set_cache_and_get_launcher<16, 32>();
                    //case 64:  return set_cache_and_get_launcher<16, 64>();
                    //case 128: return set_cache_and_get_launcher<16, 128>();
                    //case 256: return set_cache_and_get_launcher<16, 256>();
                }
                break;
            case 32:
                switch(cblk_sy) {
                    //case 4:   return set_cache_and_get_launcher<32, 4>();
                    //case 8:   return set_cache_and_get_launcher<32, 8>();
                    case 16:  return set_cache_and_get_launcher<32, 16>();
                    case 32:  return set_cache_and_get_launcher<32, 32>();
                    case 64:  return set_cache_and_get_launcher<32, 64>();
                    //case 128: return set_cache_and_get_launcher<32, 128>();
                }
                break;
            case 64:
                switch(cblk_sy) {
                    //case 4:  return set_cache_and_get_launcher<64, 4>();
                    //case 8:  return set_cache_and_get_launcher<64, 8>();
                    //case 16: return set_cache_and_get_launcher<64, 16>();
                    case 32: return set_cache_and_get_launcher<64, 32>();
                    case 64: return set_cache_and_get_launcher<64, 64>();
                }
                break;
            //case 128:
            //    switch(cblk_sy) {
            //        case 4:  return set_cache_and_get_launcher<128, 4>();
            //        case 8:  return set_cache_and_get_launcher<128, 8>();
            //        case 16: return set_cache_and_get_launcher<128, 16>();
            //        case 32: return set_cache_and_get_launcher<128, 32>();
            //    }
            //    break;
            //case 256:
            //    switch(cblk_sy) {
            //        case 4:  return set_cache_and_get_launcher<256, 4>();
            //        case 8:  return set_cache_and_get_launcher<256, 8>();
            //        case 16: return set_cache_and_get_launcher<256, 16>();
            //    }
            //    break;
            //case 512:
            //    switch(cblk_sy) {
            //        case 4: return set_cache_and_get_launcher<512, 4>();
            //        case 8: return set_cache_and_get_launcher<512, 8>();
            //    }
            //    break;
            //case 1024:
            //    if(cblk_sy == 4) {
            //        return set_cache_and_get_launcher<1024, 4>();
            //    }
            //    break;
        }
        return 0; // no suitable codeblock size found
    }
    
    
    
} // end of namespace cxmod_cuda

#endif // CXMOD_SPECIALIZED_H
