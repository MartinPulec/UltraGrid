///
/// @file    cxmod_kernel.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Main GPU entry point for context modeller.
///

#ifndef CXMOD_KERNEL_H
#define CXMOD_KERNEL_H

#include "cxmod_device_types.h"
#include "cxmod_loading.h"
#include "cxmod_border_init.h"
#include "cxmod_spp_sigma_precomp.h"
#include "cxmod_sc_precomp.h"
#include "cxmod_finalize_precomp.h"
#include "cxmod_bitplane_init.h"
#include "cxmod_coding.h"
#include "cxmod_interface.h"
#include "cxmod_reduction.h"


namespace cxmod_cuda {
    
        
    
    /// Encoded all thread's groups in given number of bitplanes using 
    /// precomputed stuff.
    /// @tparam BAND          orientation of codeblock's origin band
    /// @tparam COMPLETE      true if codeblock complete (all pixels in band)
    /// @tparam CB_SX         width of codeblock
    /// @tparam CB_SY         height of codeblock
    /// @tparam GPT           number of groups processed by each thread
    /// @param params         parameters for whole grid
    /// @param cblk           input info about processed codeblock
    /// @param scs            precomputed sign codes for thread's group
    /// @param flags          precomputed flags for thread's groups
    /// @param exchanges      buffer for exchanges with cleared borders
    /// @param temp           buffer for composign reduced output
    /// @param magnitudes     buffer with magnitudes of all groups
    /// @param num_bps        number of bitplanes to be encoded
    /// @param grp_idx        index of thread's first group
    template <j2k_band_type BAND, bool COMPLETE, int CB_SX, int CB_SY, int GPT>
    __device__ static inline void encode_cblk(
            const cxmod_kernel_params_t & params,
            const cxmod_cblk_info_t & cblk,
            const u8x4 scs[GPT],
            const u16 flags[GPT],
            u32 * const exchanges,
            u32x4 * const temp,
            const u64 * const magnitudes,
            int num_bps,
            const int grp_idx
    ) {
        // total thread count
        enum { NUM_THREADS = (CB_SX * CB_SY) / (4 * GPT) };
        
        // index of this thread in threadblock
        const int thread_idx = threadIdx.x + threadIdx.y * CB_SX;
        
        // last thread remembers, whether this is first bitplane or not
        __shared__ bool not_first_bpln;
        if(thread_idx == NUM_THREADS - 1) {
            not_first_bpln = false;
        }
        
        // initialize history of sigmas of groups
        u16 histories[GPT];
        #pragma unroll
        for(int g = 0; g < GPT; g++) { histories[g] = 0; }
        
        // offset of next output CX,D (relative to begin of output of this 
        // codeblock; in bytes)
        u32 out_offset = cblk.cxd_out_begin >> 4;  // in 16B chunks
        
        // number of remaining CX,D pairs in the buffer from previous iteration
        int remaining_cxds = 0;
        
        // for each bitplane (beginning with most significant nonzero one) ...
        while(num_bps--) {
            __syncthreads();
            // ... update state of groups for current bitplane
            #pragma unroll
            for(int g = 0; g < GPT; g++) {
                update_group<COMPLETE> (exchanges, magnitudes,
                                        grp_idx + g, num_bps, flags[g],
                                        histories[g]);
            }
            __syncthreads();
            #pragma unroll
            for(int g = 0; g < GPT; g++) {
                combine_exchanges<CB_SX + 1>(exchanges, grp_idx + g);
            }
            __syncthreads();
            // initialize offsets of outputs of output of this thread
            int spp_out_offset = 0,
                mrp_out_offset = 0,
                cup_out_offset = 0;
            
            // last thread needs extra space for end-of-pass markers
            if(thread_idx == NUM_THREADS - 1) {
                if(not_first_bpln) {
                    spp_out_offset = 1;
                    mrp_out_offset = 1;
                }
                cup_out_offset = 1;
            }
            
            // all threads compute prefix sum to get their offsets
            const u16 bpln_cxds = output_offsets<CB_SX, CB_SY, COMPLETE, GPT>
                                  (thread_idx, (u64x2*)(temp + 1), scs, flags,
                                   histories, exchanges, grp_idx, remaining_cxds,
                                   spp_out_offset, mrp_out_offset, cup_out_offset);
            
            // encode current bitplane
            __syncthreads();
            #pragma unroll
            for(int g = 0; g < GPT; g++) {
                const u32 c_xchg = exchanges[grp_idx + g];
                const u32 l_xchg = exchanges[grp_idx + g - 1];
                const u32 r_xchg = exchanges[grp_idx + g + 1];
                encode_group<BAND, COMPLETE>((u8*)temp, flags[g],
                        histories[g], scs[g], c_xchg, l_xchg, r_xchg,
                        spp_out_offset, mrp_out_offset, cup_out_offset);
            }
            
            // last thread writes end-of-pass markers
            if(thread_idx == NUM_THREADS - 1) {
                if(not_first_bpln) {
                    ((u8*)temp)[spp_out_offset] = 0xFF;
                    ((u8*)temp)[mrp_out_offset] = 0xFF;
                } else {
                    not_first_bpln = true; // first bitplane has just finished
                }
                ((u8*)temp)[cup_out_offset] = 0xFF;
            }
            
            // update count of remaining CX,D pairs and compute number of 
            // output chunks for this iteration
            remaining_cxds += bpln_cxds;
            const int num_complete_chunks = remaining_cxds >> 4;
            remaining_cxds &= 0xF;
            __syncthreads();
            
            // write all complete chunks into the global memory output buffer
            enum {
                MAX_CXDS = (CB_SX * CB_SY + 1) * 2 + 15,
                CXDS_PER_WRITE = NUM_THREADS * sizeof(*temp),
                MAX_WRITES = COMPILE_TIME<MAX_CXDS, CXDS_PER_WRITE>::DIV_RND_UP
            };
            for(int w = 0; w < MAX_WRITES; w++) {
                const int chunk_index = w * NUM_THREADS + thread_idx;
                if(chunk_index < num_complete_chunks) {
                    params.cxd_out_ptr[out_offset + chunk_index]
                            = temp[chunk_index];
                }
            }
            out_offset += num_complete_chunks;
            if(remaining_cxds && !thread_idx) {
                // possibly move last incomplete chunk to the begin (no need 
                // for sync as thread #0 overwrites chunk flushed by itself)
                temp[0] = temp[num_complete_chunks];
            }
        }
        
        if(thread_idx == 0) {
            // possibly flush last chunk
            if(remaining_cxds) {
                params.cxd_out_ptr[out_offset] = temp[0];
            }
            
            // write number of CX,D pairs
            const int cxd_out_end = out_offset * 16 + remaining_cxds;
            const int cxd_count = cxd_out_end - cblk.cxd_out_begin;
            *(cblk.cxd_count_out_ptr) = cxd_count;
        }
    }
    
    
        
    /// Loads pixels, precomputes some stuff and the calls actual encoding.
    /// @tparam COMPLETE  true if codeblock complete (all pixels in band)
    /// @tparam CB_SX     width of codeblock
    /// @tparam CB_SY     height of codeblock
    /// @tparam GPT       number of groups processed by each thread
    /// @param params     parameters for whole grid
    /// @param cblk       input info about processed codeblock
    /// @param bits       buffer for magnitude bits of all groups
    /// @param spps       buffer for SPP sigmas and reduced output composition
    /// @param xchg       buffer for precomputing SCs and later for exchanges
    template <int CB_SX, int CB_SY, bool COMPLETE, int GPT>
    __device__ static inline void cxmod_run(
            const cxmod_kernel_params_t & params,
            const cxmod_cblk_info_t & cblk,
            u64 * const bits,
            u64 * const spps,
            u32 * const xchg
    ) {
        // index (to buffers of groups) of first group procesed by this thread
        const int v_grp_idx = (1 + threadIdx.x)
                            + (1 + threadIdx.y * GPT) * (CB_SX + 1);
        
        // loads data and returns number of bitplanes to be encoded
        int num_bps = load<CB_SX, CB_SY, COMPLETE, GPT>(params, cblk, bits,
                                                        spps, xchg, v_grp_idx, 0);
        
        // save number of encoded bitplanes
        if(threadIdx.y == 0 && threadIdx.x == 0) {
            *(cblk.bplns_count_out_ptr) = num_bps;
        }
        
        // Have too much bitplanes? Load again with discarding lower ones!
        if(num_bps > cblk.max_bplns) {
            load<CB_SX, CB_SY, COMPLETE, GPT>(params, cblk, bits, spps, xchg,
                                              v_grp_idx, num_bps - cblk.max_bplns);
            num_bps = cblk.max_bplns;
        }
        
        // clear borders, precompute sigmas after each SPP and precompute SCs
        clear_border_groups<CB_SX, CB_SY, GPT>(bits, spps, xchg);
        __syncthreads();
        precompute_spp_sigmas<CB_SX, CB_SY, GPT>(bits, spps);
        __syncthreads();
        for(int g = 0; g < GPT; g++) {
            precompute_scs<CB_SX + 1>(bits, spps, xchg,
                                      v_grp_idx + g * (CB_SX + 1));
        }
        __syncthreads();
        
        // from now, change indexing of groups so that each thread processes
        // some number of vertically neighboring groups
        const int thread_idx = threadIdx.x + threadIdx.y * CB_SX;
        const int h_grp_x = thread_idx % (CB_SX / GPT);
        const int h_grp_y = thread_idx / (CB_SX / GPT);
        const int h_grp_idx = 1 + h_grp_x * GPT + (1 + h_grp_y) * (CB_SX + 1);
        
        // finalize state of each thread's group
        u8x4 scs[GPT];
        u16 flags[GPT];
        finalize<COMPLETE, GPT>(bits, spps, xchg, h_grp_idx, flags, scs);
                
        // run the right version of encoding according to band orientation
        if(HL == cblk.orientation) {
            encode_cblk<HL, COMPLETE, CB_SX, CB_SY, GPT>
                    (params, cblk, scs, flags, xchg, 
                     (u32x4*)spps, bits, num_bps, h_grp_idx);
        } else if (HH == cblk.orientation) {
            encode_cblk<HH, COMPLETE, CB_SX, CB_SY, GPT>
                    (params, cblk, scs, flags, xchg, 
                     (u32x4*)spps, bits, num_bps, h_grp_idx);
        } else { // LL or LH
            encode_cblk<LL, COMPLETE, CB_SX, CB_SY, GPT>
                    (params, cblk, scs, flags, xchg, 
                     (u32x4*)spps, bits, num_bps, h_grp_idx);
        }
    }



    /// finds the right band and identifies codeblock processed by this 
    /// threadblock in the band.
    /// @param bands  array with info structures about all bands
    /// @param num_bands  total number of bands
    /// @param initial_binary_search_step  initial step size of the binary 
    ///        search (precomputed, depends on number of codeblocks)
    /// @param cblk  output structure for info about required codeblock
    /// @param cblk_idx  index of the required codeblock (among all bands)
    __device__ static inline void get_cblk_info(j2k_cblk * const cblks,
                                                const j2k_band * const bands,
                                                cxmod_cblk_info_t & cblk,
                                                const int cblk_idx) {
        // only thread #0 loads the info
        if(threadIdx.y == 0 && threadIdx.x == 0) {
            const int band_idx = cblks[cblk_idx].band_index;
            cblk.cxd_count_out_ptr = &cblks[cblk_idx].cxd_count;
            cblk.bplns_count_out_ptr = &cblks[cblk_idx].bitplane_count;
            cblk.cxd_out_begin = cblks[cblk_idx].cxd_index;
            cblk.max_bplns = min(bands[band_idx].bitplane_limit, 16);
            cblk.orientation = bands[band_idx].type;
            cblk.pix_stride_y = bands[band_idx].size.width;
            cblk.pix_in_begin = cblks[cblk_idx].data_index;
            cblk.size_x = cblks[cblk_idx].size.width;
            cblk.size_y = cblks[cblk_idx].size.height;
        }
    }


    
    /// Main context modeller entry point. Each threadblock processes one 
    /// codeblock.
    /// @tparam CB_SX    width of the codeblock
    /// @tparam CB_SY    height of the codeblock
    /// @tparam GPT      number of pixel groups processed by each thread
    /// @param params    all other non-constant parameters
    template <int CB_SX, int CB_SY, int GPT>
    __launch_bounds__(CB_SX * CB_SY / (4 * GPT),
                      COMPILE_TIME<49152 / COMPILE_TIME<CB_SX, CB_SY>::SHMEM_BYTES, 6>::MIN)
    __global__ void cxmod_kernel(const cxmod_kernel_params_t params) {
        // some compile time constants
        enum {
            // number of pixel groups
            NUM_GROUPS = COMPILE_TIME<CB_SX, CB_SY>::NUM_GROUPS,
            
            // number of bytes needed for SPP sigma precomputation buffer
            SPP_BYTES = sizeof(u64) * NUM_GROUPS,
            
            // number of bytes needed for prefix sum and output reduction
            TEMP_BYTES = 16 + SPP_BYTES * (1 + (GPT < 2))
        };
        
        // buffer for magnitudes sharing
        __shared__ u64 bits[NUM_GROUPS];
        
        // buffer for precomputing SPP sigmas and later for reducing output
        __shared__ u64x2 spp[(TEMP_BYTES + sizeof(u64x2) - 1) / sizeof(u64x2)];
        
        // buffer for sharing signs during precomputation 
        // and for sharing states of groups when encoding
        __shared__ u32 xchg[NUM_GROUPS];
        
        // info about this threadblock's codeblock
        __shared__ cxmod_cblk_info_t cblk;
        
        // terminate if threadblock's index is not less than number of cblks
        const int cblk_idx = blockIdx.x + blockIdx.y * gridDim.x;
        if(cblk_idx >= params.num_cblks) { return; }
        
        // otherwise load the info about codeblock, wait for info to be loaded 
        // and decide which template version of encoder to use
        get_cblk_info(params.cblks_ptr, params.bands_ptr, cblk, cblk_idx);
        __syncthreads();
        
        // if codeblock is complete, run faster version without checking 
        // validity of pixels
        if((cblk.size_x == CB_SX) && (cblk.size_y == CB_SY)) {
            cxmod_run<CB_SX, CB_SY, true, GPT>
                (params, cblk, bits, (u64*)spp, xchg);
        } else {
            cxmod_run<CB_SX, CB_SY, false, GPT>
                (params, cblk, bits, (u64*)spp, xchg);
        }
    }
    
    
    
    /// Sets L1 cache preference for the kernel for specified codeblock size.
    /// @tparam CB_SIZE_X  width of codeblock
    /// @tparam CB_SIZE_Y  height of codelbock
    /// @tparam GPT        number of groups processed by each thread
    template <int CB_SIZE_X, int CB_SIZE_Y, int GPT>
    void cxmod_set_cache() {
        cudaFuncSetCacheConfig(cxmod_kernel<CB_SIZE_X, CB_SIZE_Y, GPT>,
                               cudaFuncCachePreferShared);
    }
    
    
    
    /// Launches context modeller kernel with fixed size, orientation and 
    /// output format.
    /// @tparam CB_SIZE_X     codeblock width
    /// @tparam CB_SIZE_Y     codeblock height
    /// @tparam GPT           number of groups processed by each thread
    /// @param gsize          size of the CUDA grid
    /// @param stream         CUDA stream in which the kernel should run
    /// @param params         parameters for context modeller kernel
    /// @return 0 if succeded, nonzero otherwise
    template <int CB_SX, int CB_SY, int GPT>
    int cxmod_launcher(const dim3 & gsize,
                       const cudaStream_t & stream,
                       const cxmod_kernel_params_t & params) {
        // size of threadblock
        const dim3 tsize = dim3(CB_SX, CB_SY / (4 * GPT), 1);
        
        // launch the kernel
        cxmod_kernel<CB_SX, CB_SY, GPT><<<gsize, tsize, 0, stream>>>(params);
        
        // nothing checked => nothing goes wrong :)
        return 0;
    }
    
    
    
} // end of namespace cxmod_cuda

#endif // CXMOD_KERNEL_H
