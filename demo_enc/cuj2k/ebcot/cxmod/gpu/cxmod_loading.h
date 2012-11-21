/// 
/// @file    cxmod_loading.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Loading of context modeller input and finding number of bitplanes.
/// 

#ifndef CXMOD_LOADING_H
#define CXMOD_LOADING_H

#include "cxmod_device_types.h"
#include "cxmod_util.h"

namespace cxmod_cuda {
    
    
    /// Returns number, where all less significant bits than the highest
    /// significant 1 in input, are set to 1, others to 0.
    __device__ static inline u16 bit_right_propagate(u16 n) {
        if(n) {            // n must be nonzero for following 4 steps to work:
            n = __brev(n); // reverse bits
            n &= -n;       // isolate least significant 1
            n = __brev(n); // reverse back (only most significant 1 is set now)
            n--;           // set all bits right to most significant 1
        }
        return n;          // if n was 0, output is 0 too, which is correct
    }
    


    /// Finds number of bitplanes which must be encoded in this codeblock.
    /// @param thread_max maximal magnitude found by this thread
    /// @return number of bitplanes which have to be encoded
    __device__ static inline int find_number_of_bitplanes(u32 thread_max) {
        int num_bitplanes = 0;  // initialized to 'no significant bitplane'
        
        if(__syncthreads_or(thread_max)) {
            // at least one 1 found => initialize number of bitplanes to 1
            num_bitplanes = 1;
        
            // Is most significant 1 in upper word?
            if(__syncthreads_or(thread_max & 0xFFFF0000)) {
                thread_max &= 0xFFFF0000;  // mask out lower word
                num_bitplanes += 16;       // more than 16 bitplanes
            }
            
            // Is most significant 1 in upper bytes?
            if(__syncthreads_or(thread_max & 0xFF00FF00)) {
                thread_max &= 0xFF00FF00; // mask out lower bytes
                num_bitplanes += 8;    // more than 8 bitplanes to be encoded
            }
            
            // Is most significant 1 in odd nibbles?
            if(__syncthreads_or(thread_max & 0xF0F0F0F0)) {
                thread_max &= 0xF0F0F0F0;  // mask out lower nibbles
                num_bitplanes += 4;    // 4 more bitplanes need to be encoded
            }
            
            // Is most significant 1 in odd pairs of bits?
            if(__syncthreads_or(thread_max & 0xCCCCCCCC)) {
                thread_max &= 0xCCCCCCCC;  // mask out even bit pairs
                num_bitplanes += 2;    // 2 more bitplanes to be encoded
            }
            
            // Is most significant 1 odd bit?
            if(__syncthreads_or(thread_max & 0xAAAAAAAA)) {
                num_bitplanes += 1;    // most significant bitplane discovered
            }
        }
        
        return num_bitplanes;
    }
    
    
    
    /// Loads input of context modeller into given buffer of pixel groups.
    /// @tparam CB_SX       height of codeblock
    /// @tparam CB_SY       width of codeblock
    /// @tparam COMPLETE    is codeblock complete? (not crossing band boundary)
    /// @tparam GPT         number of pixel groups processed by one thread
    /// @param params           common context modeller parameters
    /// @param cblk             info about processed codeblock
    /// @param magnitudes_out   array for output magnitudes
    /// @param spp_sigmas_out   array for initial outpur spp sigma states
    /// @param signs_out        array for output signs flags
    /// @param first_group_idx  index of first group to be loaded by thread
    /// @param discard_lsb      number of least significat bits to be discarded
    /// @return number of least significant bitplanes which must be encoded
    template <int CB_SX, int CB_SY, bool COMPLETE, int GPT>
    __device__ static inline int load(const cxmod_kernel_params_t & params,
                                      const cxmod_cblk_info_t & cblk,
                                      u64 * const magnitudes_out, 
                                      u64 * const spp_sigmas_out, 
                                      u32 * const signs_out,
                                      const int first_group_idx,
                                      const int discard_lsb) {
        // y-coordinate of loaded pixel (relative to codeblock top-left pixel)
        int pixel_y = threadIdx.y * 4 * GPT;
        
        // Pointer to next pixel to be loaded, or past the band end if next 
        // pixel is out of band. (Skip the test it codeblock is complete.)
        const int * next_pixel = params.pixels_in_ptr  // all data
                               + cblk.pix_in_begin     // first pixel of cblk
                               + threadIdx.x           // x-offset and ...
                               + pixel_y * cblk.pix_stride_y; // ... y-offset
        
        // disjunction of all loaded magnitudes - most significant loaded 
        // bit will be set here
        u32 max_bits = 0;
        
        // load all groups
        for(int g = 0; g < GPT; g++) {
            // output variables for the group
            u8 flags[4];
            u16 bits[4];
            u16 spps[4];
            
            // possibly initialize input if codeblock not complete
            if(!COMPLETE) {
                #pragma unroll
                for(int p = 0; p < 4; p++) {
                    flags[p] = 0;
                    bits[p] = 0;
                    spps[p] = 0;
                }
            }
            
            // is the x-coordinate in the band?
            if(COMPLETE || (threadIdx.x < cblk.size_x)) {
                // load all 4 pixels
                #pragma unroll
                for(int p = 0; p < 4; p++) {
                    // check y-coordinate and possibly load the pixel
                    if(COMPLETE || (pixel_y++ < cblk.size_y)) {
                        const int value = *next_pixel;
                        const u32 magnitude = abs(value) >> discard_lsb;
                        bits[p] = magnitude;
                        
                        // update maximal magnitude
                        max_bits |= magnitude; 
                        
                        // init sigmas after spp with sigmas from previous bitplane
                        spps[p] = bit_right_propagate(bits[p]); 
                        
                        // set sign flags: set bit #1 and if value is negative, 
                        // set also bits #0 and #4
                        if(value < 0) {
                            flags[p] = 0x03;
                            spps[p] |= 0x8000;
                        } else {
                            flags[p] = 0x02;
                        }
                        
                        // advance to next pixel in the band
                        next_pixel += cblk.pix_stride_y;
                    }
                }
            }
            
            // save the loaded group status
            const int group_idx = first_group_idx + g * (CB_SX + 1);
            ((u16x4*)magnitudes_out)[group_idx]
                    = make_ushort4(bits[0], bits[1], bits[2], bits[3]);
            ((u16x4*)spp_sigmas_out)[group_idx]
                    = make_ushort4(spps[0], spps[1], spps[2], spps[3]);
            ((u8x4*)signs_out)[group_idx]
                    = make_uchar4(flags[0], flags[1], flags[2], flags[3]);
        }
        
        // find position of most significant magnitude bit among all threads 
        // which process the same codeblock
        return find_number_of_bitplanes(max_bits);
    }

    
} // end of namespace cxmod_cuda


#endif // CXMOD_LOADING_H

