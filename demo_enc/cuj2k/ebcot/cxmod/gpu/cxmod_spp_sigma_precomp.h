///
/// @file    cxmod_spp_sigma_precomp.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Precomputing of states of sigmas after each SPP.
///


#ifndef CXMOD_SPP_SIGMA_PRECOMP_H
#define CXMOD_SPP_SIGMA_PRECOMP_H

#include "cxmod_device_types.h"
#include "cxmod_util.h"

namespace cxmod_cuda {
    
    
    
    /// Precomputes sigmas after SPP for each pixel in each bitplane.
    /// Assumes that for each non-boundary pixel, its magnitude is set and 
    /// its SPP sigmas are initialized with sigmas after previous CUP.
    /// Also assumes that SPP sigma for all pixels in border groups is 0.
    /// @tparam CB_SX    width of codeblock
    /// @tparam CB_SY    height of codeblock
    /// @tparam GPT      number of groups processed by each thread
    /// @param magnitudes_buffer  magnitudes of all groups
    /// @param spp_sigmas_buffer  initialized SPP sigmas of all groups
    template <int CB_SX, int CB_SY, int GPT>
    __device__ static void precompute_spp_sigmas(
            const u64 * const magnitudes_buffer, 
            u64 * const spp_sigmas_buffer) {
        // some compile time constants
        enum {
            NUM_THREADS_Y = CB_SY / (4 * GPT),
            PRECOMP_ROWS = COMPILE_TIME<NUM_THREADS_Y / 1, 1>::MAX,
            ALL_THREADS_PRECOMPUTE = NUM_THREADS_Y == PRECOMP_ROWS,
            PRECOMP_ITERATIONS = CB_SY / (4 * PRECOMP_ROWS)
        };
        
        // true if this thread participates on precomputing
        const bool this_thread_precomputes = threadIdx.y < PRECOMP_ROWS;
        
        // index of the group processed by this thread
        int group_idx = (ALL_THREADS_PRECOMPUTE || this_thread_precomputes)
                ? (1 + threadIdx.x + (1 + threadIdx.y) * (CB_SX + 1))
                : 0;
        
        // During SPP sigma precomputing, codeblock is horizontally divided 
        // into stripes. Each stripe gets precomputed at once in parallel 
        // and after that the next (bottom) one is being precomputed. Not 
        // all threads are participating in precomputing. Only those which 
        // have threadIdx.y lower than PRECOMP_ROWS are precomputing SPP 
        // sigmas of one group of 4 pixels in each stripe. Others wait. 
        // Following loop iterates through y-axis offsets of all stripes.
        
        // Now iterate through all the stripes:
        for(int strp = 0; strp < PRECOMP_ITERATIONS; strp++) {
            // Constant parts of prefered neighborhood sigmas. These can be 
            // loaded only once per each group of 4 pixels. Others need to 
            // be re-loaded before each iteration.
            u64 const_pref_neighb = 0;
            
            // current state of SPP sigmas of all 4 pixels
            u64 sigmas = 0;
            
            // magnitude bits (values) of all 4 pixels
            u64 bits = 0;
            
            // if this thread is one of precomputing threads ...
            if(ALL_THREADS_PRECOMPUTE || this_thread_precomputes) {
                // load all initial sigmas from this and right group
                const u64 r_sigmas = spp_sigmas_buffer[group_idx + 1];
                sigmas = spp_sigmas_buffer[group_idx];
                
                // add right neighbor sigmas to prefered neighborhood of 
                // all 4 pixels of the group
                const_pref_neighb = r_sigmas;
                
                // add sigmas of upper right neighbors
                const_pref_neighb |= r_sigmas << 16;
                
                // add sigmas of lower and lower-right neighbors
                const_pref_neighb |= (r_sigmas | sigmas) >> 16;
                
                // add all 3 lower neighbors of last pixel
                const u16x4 * const lower = (const u16x4*)(spp_sigmas_buffer + group_idx + CB_SX);
                const_pref_neighb |= u64(lower[0].x | lower[1].x | lower[2].x) << 48;
                
                // load all magnitude bits
                bits = magnitudes_buffer[group_idx] & 0x7FFF7FFF7FFF7FFFL;
            }
            
            // will be nonzero after each iteration if thread's group's 
            // SPP sigmas change in the iteration
            int changes = 0;
            
            // propagate sigmas as long as there are any changes
            do {
                // if this thread is one of precomputing threads ...
                if(ALL_THREADS_PRECOMPUTE || this_thread_precomputes) {
                    // Constant sigmas are alredy preloaded for all 
                    // iterations of this propagation loop. For each of
                    // 4 pixels,pixel, load also variable prefered 
                    // neighborhood sigmas and update sigma.
                    
                    // Preload varying prefered neighborhood which is 
                    // constant for this iteration.
                    const u64 l_sigmas = spp_sigmas_buffer[group_idx - 1];
                    
                    // Initialize this iteration's variable PN with const
                    // PN and left neighbor sigmas
                    u64 pref_neighb = const_pref_neighb | l_sigmas;
                    
                    // add upper and upper-left neighbor sigmas 
                    pref_neighb |= (sigmas | l_sigmas) << 16;
                    
                    // add lower left sigmas
                    pref_neighb |= l_sigmas >> 16;
                    
                    // for first pixel, upper sigmas must be selected 
                    // from upper group
                    const u16x4 * const upper = (const u16x4*)(spp_sigmas_buffer + group_idx - CB_SX - 2);
                    pref_neighb |= upper[0].w | upper[1].w | upper[2].w;
                    
                    // update spp sigmas of all pixes at once
                    const u64 new_sigmas = sigmas | (bits & pref_neighb);
                    
                    // are there any differences?
                    changes = (new_sigmas != sigmas);
                    
                    // replace old sigmas with new ones (both in registers 
                    // and shared memory)
                    sigmas = new_sigmas;
                }
                
                // wait for all threads to precompute their sigmas and replace
                // them with new ones
                __syncthreads();
                if(ALL_THREADS_PRECOMPUTE || this_thread_precomputes) {
                    spp_sigmas_buffer[group_idx] = sigmas;
                }
                
                // if any thread's group's SPP sigmas changed, all threads 
                // continue with next iteration:
            } while(__syncthreads_or(changes));
            
            // No changes in last sigma propagation iteration => current 
            // stripe is precomputed => advance group pointer to this
            // thread's group in next stripe!
            group_idx += PRECOMP_ROWS * (CB_SX + 1);
        }
    } // end of void precompute_spp_sigmas()
    
    
    
} // end of namespace cxmod_cuda

#endif // CXMOD_SPP_SIGMA_PRECOMP_H
