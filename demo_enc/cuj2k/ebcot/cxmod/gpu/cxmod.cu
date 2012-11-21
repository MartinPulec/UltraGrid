/// 
/// @file    cxmod.cu
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @date    2011-06-03 12:06
/// @brief   Implementation of interface of CUDA JPEG2000 context modeller.
///


#include <cstdlib>
#include <cmath>

#include "cxmod_device_types.h"
#include "cxmod_specialized.h"
#include "cxmod_interface.h"
#include "cxmod_util.h"

#include "lookup_tables/cxmod_sc_lut.h"
#include "lookup_tables/cxmod_zc_mrc_ll_lh_lut.h"
#include "lookup_tables/cxmod_zc_mrc_hl_lut.h"
#include "lookup_tables/cxmod_zc_mrc_hh_lut.h"



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////                                        ///////////////////
////////////////////  GPU LOOKUP TABLES AND THEIR WRAPPERS  ///////////////////
////////////////////                                        ///////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


namespace cxmod_cuda {
    namespace gpu_luts {

        
        /// Sign coding lookup table.
        __constant__ u8 sc_lut[25];

        /// MRC and ZC lookup table in LL and LH bands.
        __device__ u8 mrc_zc_ll_lh_lut[1 << 11];
        
        /// MRC and ZC lookup table in HL bands.
        __device__ u8 mrc_zc_hl_lut[1 << 11];
        
        /// MRC and ZC lookup table in HH bands.
        __device__ u8 mrc_zc_hh_lut[1 << 11];

        
        
        /// Sign coding table lookup.
        __device__ u8 sc_lookup(const int sc_index) {
            return sc_lut[sc_index];
        }
        
        
        /// Lookup table for MRC and ZC in LL and LH bands.
        __device__ u8 zc_and_mrc_ll_lh_lookup(const u32 lut_index) {
            return mrc_zc_ll_lh_lut[lut_index];
        }
        
        
        /// Lookup table for MRC and ZC in HL bands.
        __device__ u8 zc_and_mrc_hl_lookup(const u32 lut_index) {
            return mrc_zc_hl_lut[lut_index];
        }
        
        
        /// Lookup table for MRC and ZC in HH bands.
        __device__ u8 zc_and_mrc_hh_lookup(const u32 lut_index) {
            return mrc_zc_hh_lut[lut_index];
        }
    
    
    } // end of namespace gpu_luts
} // end of namespace cxmod_cuda


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//////////////////////////                            /////////////////////////
//////////////////////////  INTERFACE IMPLEMENTATION  /////////////////////////
//////////////////////////                            /////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////



// often using stuff from namespace cxmod_cuda
using namespace cxmod_cuda;



/// Loads coding tables into GPU memory.
static int cxmod_load_tables() {
    // Load lookup table for SC.
    if(cudaMemcpyToSymbol(gpu_luts::sc_lut, lookup_tables::sc_lut,
                          sizeof(lookup_tables::sc_lut),
                          0, cudaMemcpyHostToDevice) != cudaSuccess) {
        return -1;
    }
    
    // Load lookup table for ZC and MRC in LL and LH bands.
    if(cudaMemcpyToSymbol(gpu_luts::mrc_zc_ll_lh_lut,
                          lookup_tables::zc_mrc_ll_lh_lut,
                          sizeof(lookup_tables::zc_mrc_ll_lh_lut),
                          0, cudaMemcpyHostToDevice) != cudaSuccess) {
        return -2;
    }
    
    // Load lookup table for ZC and MRC in HL bands.
    if(cudaMemcpyToSymbol(gpu_luts::mrc_zc_hl_lut,
                          lookup_tables::zc_mrc_hl_lut,
                          sizeof(lookup_tables::zc_mrc_hl_lut),
                          0, cudaMemcpyHostToDevice) != cudaSuccess) {
        return -3;
    }
    
    // Load lookup table for ZC and MRC in HH bands.
    if(cudaMemcpyToSymbol(gpu_luts::mrc_zc_hh_lut,
                          lookup_tables::zc_mrc_hh_lut,
                          sizeof(lookup_tables::zc_mrc_hh_lut),
                          0, cudaMemcpyHostToDevice) != cudaSuccess) {
        return -4;
    }
        
    return 0;  // indicate success
}



/**
 * Creates a new instance of context modeller for given parameteers.
 * @param p initialized parameters of J2K encoder
 * @return null = error, non-null = new valid instance of context modeller
 */
void * cxmod_create(const struct j2k_encoder_params * const p) {
    // check pointer and initialize lookup tables
    if(p == 0 || cxmod_load_tables()) { return 0; }
    
    // try to find context modeller for required codeblock size
    return (void*)cxmod_find_launcher(p->cblk_size.width, p->cblk_size.height);
}



/**
 * Releases all resources associated with given context modeller instance.
 * @param cxmod_ptr  pointer to internal stuff of context modeller in main 
 *                   system memory returned by 'cxmod_create'
 * @return zero if destroyed OK, nonzero otherwise
 */
int cxmod_destroy(void * const cxmod_ptr) {
    return 0; // nothing to do - it's just for possibility of changes in future
}



/// Finds grid size with valid dimensions and minimal 
/// number of useless threadblocks for given number of threadblocks.
inline dim3 cxmod_grid_dim_for(const int num_cblks) {
    int size_x = num_cblks;
    int size_y = 1;
    while(num_cblks >= (1 << 16)) {
        size_x = (size_x + 1) / 2;
        size_y *= 2;
    }
    return dim3(size_x, size_y, 1);
}



/**
 * Uses given context modeller structure to encode codeblocks in specified 
 * buffer, to save output CX,D pairs to the other specified buffer 
 * and to write info about output codeblocks and passes with specified 
 * formatting.
 * @param cxmod_ptr  main system memory pointer to context modeller 
 *        internal data (returned by 'cxmod_create')
 * @param cblk_count  number of codeblocks to be encoded
 * @param cblks_gpu_ptr  GPU memory pointer to array with info about codeblocks
 * @param bands_gpu_ptr  GPU memory pointer to array with info about bands 
 * @param in_pixels_gpu_ptr  GPU memory pointer to all input pixels
 * @param out_cxd_gpu_ptr  GPU memory pointer to buffer for output CX,D pairs
 *        NOTE: must be aligned to 16byte boundary!
 * @param cuda_stream_ptr  pointer to variable of type cudaStream_t in main 
 *        system memory, specifying CUDA stream, in which the context modeller
 *        kernel should be launched (can be NULL for default stream)
 * @return zero if successfully launched, nonzero otherwise
 */
int cxmod_encode(
    void * const cxmod_ptr,
    const int cblk_count,
    struct j2k_cblk * const cblks_gpu_ptr,
    const struct j2k_band * const bands_gpu_ptr,
    const int * const in_pixels_gpu_ptr,
    unsigned char * const out_cxd_gpu_ptr,
    const void * const cuda_stream_ptr
) {
    // check parameters
    if(0 == cxmod_ptr) { return -1; }
    if(0 == cblks_gpu_ptr) { return -2; }
    if(0 == bands_gpu_ptr) { return -3; }
    if(0 == in_pixels_gpu_ptr) { return -4; }
    if(0 == out_cxd_gpu_ptr) { return -5; }
    if(15 & (size_t)out_cxd_gpu_ptr) { return -6; } // alignment check
    if(0 > cblk_count) { return -7; }

    // parameters OK, compose kernel parameters structure
    cxmod_kernel_params_t params;
    params.bands_ptr = bands_gpu_ptr;
    params.cblks_ptr = cblks_gpu_ptr;
    params.cxd_out_ptr = (u32x4*)out_cxd_gpu_ptr;
    params.pixels_in_ptr = in_pixels_gpu_ptr;
    params.num_cblks = cblk_count;
    
    // CUDA stream ID for context modeller kernel
    const cudaStream_t stream = cuda_stream_ptr
            ? *((const cudaStream_t*)cuda_stream_ptr) : 0;
    
    // grid size for giiven number of codeblocks
    const dim3 grid_dim = cxmod_grid_dim_for(cblk_count);
    
    // launch the kernel
    return ((cxmod_launcher_fn_t)cxmod_ptr)(grid_dim, stream, params);
}


