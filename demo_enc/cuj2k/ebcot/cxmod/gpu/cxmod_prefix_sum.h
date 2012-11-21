///
/// @file    cxmod_prefix_sum.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Parallel prefix sum implementation specific for cxmod needs.
///

#ifndef CXMOD_PREFIX_SUM_H
#define CXMOD_PREFIX_SUM_H


#include "cxmod_device_types.h"


namespace cxmod_cuda {
    
    
    
    /// At compile time, this computes number of reduction levels of prefix 
    /// sum of specified size and maximal size of last level.
    /// @tparam SIZE  number of items
    /// @tparam MAX   minimal number of items to split to next level
    template <int SIZE, int MAX>
    struct REDUCTIONS {
        private: enum { LAST = SIZE <= MAX || SIZE & 1 };
        public:  enum { NUM = LAST ? 0 : 1 + REDUCTIONS<SIZE / 2, MAX>::NUM };
    };

    
    
    /// Specialization to stop recursion of level computing.
    template <int MAX> struct REDUCTIONS<0, MAX> { enum { NUM = 0 }; };


    
    /// Computes prefix sum over items saved as triples. Theme must be at least
    /// one thread for each pair of triples for it to work correctly.
    /// @tparam ITEMS      number of prefix summed triples
    /// @param thread_idx  unique ID of this thread, IDs must be consecutive,
    ///                    starting with 0
    /// @param values      three packed values, w-component must be 0
    /// @param temp        buffer for performing prefix sum - aligned to 16B
    /// @param offset      number to be added to results of all threads
    /// @return 4 packed values:
    ///            X = sum of all previous Xs + offset
    ///            Y = sum of all previous Ys + offset + sum of all Xs
    ///            Z = sum of all previous Zs + offset + sum of all Xs and Ys
    ///            W = sum of all Xs, Ys and Zs (without offset)
    template <int ITEMS>
    __device__ static inline u16x4 packed_prefix_sum(const int thread_idx,
                                                     const u16x4 values,
                                                     u64x2 * const temp,
                                                     const int offset) {
        // compile time constants
        enum {
            // number of parallel reduction levels
            LEVELS = REDUCTIONS<ITEMS, 8>::NUM,
            
            // number of serially processed items
            NUM_SERIAL = ITEMS >> LEVELS,
            
            // offset of serially processed items
            OFFSET_SERIAL = 2 * (ITEMS - NUM_SERIAL)
        };
        
        // save thread's values at the right index
        ((u16x4*)temp)[thread_idx] = values;
        
        // reduce items in parallel
        #pragma unroll
        for(int l = 0; l < LEVELS; l++) {
            // wait for initialization or previous level to be safely written
            __syncthreads();
            
            // following should be resolved at compile time:
            const int offset = 2 * ITEMS - (1 << LEVELS + 1 - l) * NUM_SERIAL;
            const int in_count = ITEMS >> l;
            const int out_count = in_count >> 1;
            
            // possibly reduce to next level
            if(thread_idx < out_count) {
                const u64x2 val = temp[offset / 2 + thread_idx];
                ((u64*)temp)[offset + in_count + thread_idx] = val.x + val.y;
            }
        }
        
        // process top-level items serially after completing last level
        __syncthreads();
        if(0 == thread_idx) {
            // get sums of CX,Ds in three passes
            u64 sums = 0;
            #pragma unroll
            for(int i = 0; i < NUM_SERIAL; i++) {
                sums += ((u64*)temp)[OFFSET_SERIAL + i];
            }
            
            // compose initial value for offsets
            u64 offsets = 0x0001000100010000LL * sums
                        + 0x0000000100010001LL * offset;
            
            // add offsets to all top-level items
            #pragma unroll
            for(int i = 0; i < NUM_SERIAL; i++) {
                const u64 value = ((u64*)temp)[OFFSET_SERIAL + i];
                ((u64*)temp)[OFFSET_SERIAL + i] = offsets;
                offsets += value;
            }
        }
        
        // distribute offsets in parallel
        #pragma unroll
        for(int l = LEVELS - 1; l >= 0; l--) {
            // wait for previous level or serial reduction to be safely written
            __syncthreads();
            
            // following should be resolved at compile time:
            const int offset = 2 * ITEMS - (1 << LEVELS + 1 - l) * NUM_SERIAL;
            const int in_count = ITEMS >> l;
            const int out_count = in_count >> 1;
            
            // possibly distribute offsets to one pair from previous level
            if(thread_idx < out_count) {
                u64x2 reduced = {
                    ((u64*)temp)[offset + in_count + thread_idx],
                    ((u64*)temp)[offset + 2 * thread_idx]
                };
                reduced.y += reduced.x;
                temp[offset / 2 + thread_idx] = reduced;
            }
        }
        
        // wait for the distribution to be done and return the result
        __syncthreads();
        return ((u16x4*)temp)[thread_idx];
    }
    
    
    
}; // end of namespace cxmod_cuda

#endif // CXMOD_PREFIX_SUM_H
