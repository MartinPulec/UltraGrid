/// 
/// @file    cxmod_types.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Data types used by context modeller GPU implementation.
///

#ifndef CXMOD_TYPES_H
#define CXMOD_TYPES_H

#include "cxmod_interface.h"

namespace cxmod_cuda {
    

    /// 8bit unsigned integer type
    typedef unsigned char u8;


    /// 16bit unsigned integer type
    typedef unsigned short u16;


    /// 32bit unsigned integer type
    typedef unsigned int u32;


    /// 64bit unsigned integer type
    typedef unsigned long long int u64;
    
    
    /// structure of 4 unsigned 16bit values
    typedef ushort4 u16x4;
    
    
    /// structure of 4 unsigned 8bit values
    typedef uchar4 u8x4;
    
    
    /// structure of 4 unsigned 32bit values
    typedef uint4 u32x4;
    
    
    /// structure of 2 unsigned 64bit values
    typedef ulonglong2 u64x2;
    
    
    
    /// Info about one input codeblock needed in CUDA kernel
    struct cxmod_cblk_info_t {
        // begin of codeblock's CX,D output relative to CX,D buffer for all 
        // codeblocks
        int cxd_out_begin;
        
        // offset of first inut pixel relative to begin of buffer for all pixels
        int pix_in_begin;
        
        // difference between indices of two vertically neighboring pixels
        int pix_stride_y;
        
        // actual width of codeblock in pixels
        int size_x;
        
        // actual height of codeblock in pixels
        int size_y;
        
        // orientation of the codeblock
        j2k_band_type orientation;
        
        // maximal number of most significant bitplanes to be encoded
        int max_bplns;
    
        // CX,D pairs count in code-block (including pass end special pairs)
        int * cxd_count_out_ptr;

        // Bitplanes count (filled by cxmod)
        int * bplns_count_out_ptr;
    }; // end of struct cxmod_cblk_info_t

    
    
    /// All parameters needed by context modeller kernel.
    struct cxmod_kernel_params_t {
        /// output buffer for CX,D pairs (GPU memory pointer)
        u32x4 * cxd_out_ptr;
        
        /// Pointer to input pixels (or coefficients)
        const int * pixels_in_ptr;
        
        /// number of codeblocks
        int num_cblks;
        
        /// pointer to info about input codeblocks
        struct j2k_cblk * cblks_ptr;
        
        /// pointer to info about input bands
        const struct j2k_band * bands_ptr;
    }; //end of struct cxmod_params_t
    
    
    
    /// Type of function which launches context modeller.
    /// @param gsize             size of the grid
    /// @param stream            cuda stream in for context modeller kernel
    /// @param params            parameters for the context modeller kernel
    /// @return zero if successfully launched, nonzero otherwise
    typedef int (*cxmod_launcher_fn_t)(
        const dim3 & gsize,
        const cudaStream_t & stream,
        const cxmod_kernel_params_t & params
    );
    
    
    
}; // end of namespace cxmod_cuda

#endif // CXMOD_TYPES_H

