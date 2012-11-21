/* 
 * Copyright (c) 2012, Martin Jirman (martin.jirman@cesnet.cz)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "rate_control.h"


// Quality exponent range and corresponding base.
#define MIN_QUALITY_EXPONENT -5000000
#define MAX_QUALITY_EXPONENT +5000000
__device__ const static float QUALITY_BASE = 1.000005f;

/// Exponent of maximal quality found that fits into the byte limit.
__device__ static int max_good_quality_exp;

/// Exponent of minimal quality that does not fit into the limit.
__device__ static int min_bad_quality_exp;

/// Currently examined quality.
__device__ static float quality;

/// True if sufficient quality found.
__device__ bool quality_found;

/// Byte count for last checked quality.
__device__ size_t quality_byte_count;


/// Kernel for intitialization of quality exponents.
/// (Only single thread is expected to be launched.)
__global__ static void quality_init_kernel() {
    max_good_quality_exp = MIN_QUALITY_EXPONENT;
    min_bad_quality_exp = MAX_QUALITY_EXPONENT;
    quality_found = false;
    quality_byte_count = 0;
    const int quality_exp = (min_bad_quality_exp + max_good_quality_exp) >> 1;
    quality = __powf(QUALITY_BASE, quality_exp);
}


/// Search for best truncation point in given list according to given quality.
/// @param quality  the quality value
/// @param in_rates  pointer to array of truncation point rates 
///                  (24 LSBs is byte size and 8 MSBs is pass count)
/// @param in_distortions  pointer to array of truncation point distortions
/// @param trunc_count  count of truncation points (must be positive)
/// @return best truncation point (24 LSBs = byte size, 8 MSBs = pass count)
__device__ static unsigned int trunc_search(
                const float quality,
                const unsigned int * const in_rates,
                const float * const in_distortions,
                const int trunc_count
) {
    // assumes that both distortions and rates buffers are aligned to 16 bytes
    const float4 * distortions4 = (const float4*)in_distortions;
    const uint4 * rates4 = (const uint4*)in_rates;
    
    // find truncation point best matching to current quality
    // (find i with minimal value of sizes[i] + quality * distortions[i])
    unsigned int best_trunc = 0;
    float best_score = __fdividef(1.0f, 0.0f);  // infinity
    
    // keep loading and testing 4 truncation points at once 
    // (up to 3 more points beyond count are guarranteed to be valid)
    for(int i = (trunc_count + 3) >> 2; i--;) {
        // load 4 truncation points
        const float4 distortions = *(distortions4++);
        const uint4 rates = *(rates4++);
        
        // evaluate their scores
        const float score_x = (rates.x & 0xffffff) + quality * distortions.x;
        const float score_y = (rates.y & 0xffffff) + quality * distortions.y;
        const float score_z = (rates.z & 0xffffff) + quality * distortions.z;
        const float score_w = (rates.w & 0xffffff) + quality * distortions.w;
        
        // remember only the point with best score
        if(score_x < best_score) {
            best_score = score_x;
            best_trunc = rates.x;
        }
        if(score_y < best_score) {
            best_score = score_y;
            best_trunc = rates.y;
        }
        if(score_z < best_score) {
            best_score = score_z;
            best_trunc = rates.z;
        }
        if(score_w < best_score) {
            best_score = score_w;
            best_trunc = rates.w;
        }
    }
        
    // return best truncation point info
    return best_trunc;
}


/// Checks currently examined quality exponent and sets next exponent 
/// for next search iteration. Expected to be called in single thread.
/// @param max_byte_count  byte count limit
/// @param byte_count_tolerance  tolerance for the byte count limit
__global__ static void quality_check_kernel(
                const size_t max_byte_count,
                const size_t byte_count_tolerance
) {
    // stop if already done
    if(quality_found) {
        return;
    }
    
    // get last checked exponent
    const int quality_exp = (min_bad_quality_exp + max_good_quality_exp) >> 1;
    
    // update either feasible maximum or unfeasable minimum
    if(quality_byte_count < max_byte_count) {
        if(quality_exp > max_good_quality_exp) {
            max_good_quality_exp = quality_exp;
        }
        if(max_byte_count - quality_byte_count < byte_count_tolerance) {
            quality_found = true;
        }
    } else {
        if(quality_exp < min_bad_quality_exp) {
            min_bad_quality_exp = quality_exp;
        }
    }
    
    // stop, if search range has been reduced enough
    if(min_bad_quality_exp - max_good_quality_exp <= 1) {
        quality_found = true;
    }
    
    // clear the byte count for next iteration
    quality_byte_count = 0;
    
    // set quality for next iteration
    const int new_exp = (min_bad_quality_exp + max_good_quality_exp) >> 1;
    quality = __powf(QUALITY_BASE, new_exp);
}


/// Kernel for search for maximal feasible quality within specified byte count.
/// @param thread_count_rcp  inverse of total count of threads - 1
/// @param trunc_byte_sizes  array of byte sizes for all truncation points
/// @param trunc_distortions  array of distortions for all truncation points
/// @param cblks  array of codeblock info structures
/// @param cblk_count  number of codeblocks to examine
/// @param byte_count_limit  max total byte count for all truncated codeblocks
/// @param byte_count_tolerance  maximal deviation from required byte count 
///                              to stop the search
__global__ static void quality_search_kernel(
                const unsigned int * const trunc_byte_sizes,
                const float * const trunc_distortions,
                const j2k_cblk * const cblks,
                const int cblk_count
) {
    // stop if best quality already found
    if(quality_found) {
        return;
    }
    
    // byte counts for all threads of the block 
    // (each thread finds best truncation point for one codeblock)
    extern __shared__ unsigned int byte_counts[];
    byte_counts[threadIdx.x] = 0;
    
    // select some codeblock (if not out of range)
    const int cblk_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(cblk_idx < cblk_count) {
        // pointer to thread's codeblock structure
        const j2k_cblk * const cblk = cblks + cblk_idx;
        
        // codeblock's truncation points offset in the buffer and their count
        const int trunc_offset = cblk->trunc_index;
        const int trunc_count = cblk->trunc_count;
        
        // find the best truncation point according to currently examined 
        // quality and save the correponding byte count (24 LSBs)
        byte_counts[threadIdx.x] = 0xFFFFFF & trunc_search(
                quality, 
                trunc_byte_sizes + trunc_offset, 
                trunc_distortions + trunc_offset,
                trunc_count
        );
    }
    
    // first warp reduces thread sums
    __syncthreads();
    if(threadIdx.x < 32) {
        unsigned int byte_count = 0;
        for(int i = threadIdx.x; i < blockDim.x; i += 32) {
            byte_count += byte_counts[i];
        }
        byte_counts[threadIdx.x] = byte_count;
        
        // thread #0 sums up all the byte counts and adds the sum to global sum
        if(threadIdx.x == 0) {
            unsigned int byte_count = 0;
            for(int i = 0; i < 32; i++) {
                byte_count += byte_counts[i];
            }
            atomicAdd((unsigned long long int*)&quality_byte_count, byte_count);
        }
    }
}


/// Truncate codestreams of all codeblocks according to maximal feasible 
/// quality found by previous kernel.
/// @param cblks  array of codeblock info structures
/// @param cblk_count  total count of codeblocks
/// @param trunc_byte_sizes  array of byte counts of all truncation points
/// @param trunc_distortions  array of distortions of all truncation points
__global__ static void truncation_kernel(
                j2k_cblk * const cblks,
//                 const int * const cblk_idx_list,
                const int cblk_count,
                const unsigned int * const trunc_byte_sizes,
                const float * const trunc_distortions
) {
//     // DEBUG: first thread prints exponent limits
//     if(blockIdx.x == 0 && threadIdx.x == 0) {
//         printf("Quality exponent min: %d, max: %d.\n",
//                min_bad_quality_exp, max_good_quality_exp);
//     }
    
    // each thread truncates single codeblock (get index of some codeblock)
    const int cblk_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(cblk_idx < cblk_count) {
        // get the best quality that can fit into the limit
        const float quality = __powf(QUALITY_BASE, max_good_quality_exp);
        
        // pointer to codeblock structure
        j2k_cblk * const cblk = cblks + /*cblk_idx_list[*/cblk_idx/*]*/;
        
        // codeblock's truncation points offset in the buffer and their count
        // (Truncation point's index corresponds to number of encoded passes.)
        const int trunc_offset = cblk->trunc_index;
        const int trunc_count = cblk->trunc_count;
        
        // pointer to first truncation's distortion and byte count
        const float * distortions = trunc_distortions + trunc_offset;
        const unsigned int * sizes = trunc_byte_sizes + trunc_offset;
        
        // find index of best truncation point according to selected quality
        // (The index represents number of included passes.)
        const unsigned int trunc = trunc_search(quality, sizes, distortions, trunc_count);
        
        // update the codeblock info accordingly
        cblk->pass_count = trunc >> 24;
        cblk->byte_count = trunc & 0xFFFFFF;
        
//         // DEBUG: show some info about the codeblock truncation
//         const int total_pass_count = max(0, cblk->bitplane_count * 3 - 2);
//         printf("CBLK #%05d: passes total: %d, included: %d, truncated: %d.\n",
//                cblk_idx, total_pass_count, pass_count, total_pass_count - pass_count
//         );
    }
}


/// Number of warps per threadblock in distortion estimation kernel.
const static int DISTORTION_WARPS_PER_TBLOCK = 8;

/// Computes distortion estimate for all passes.
/// threadIdx.x ranges from 0 to 31 inclusive
/// threadIdx.y is index of the warp within threadblock
/// Each thread computes distortion of one bitplane in one codeblock.
/// @param cblk_count  count of codeblocks
/// @param cblks  array of all codeblock info structures
/// @param bands  array of all band info structures
/// @param res  array of all resolution info structures
/// @param distortions  array for distortions of all truncation points
/// @param byte_sizes  array of byte sizes for truncation points
/// @param coefficents  array of input codefficients for quantization
__launch_bounds__(32 * DISTORTION_WARPS_PER_TBLOCK,
                  47 / DISTORTION_WARPS_PER_TBLOCK)
__global__ static void distortion_estimation_kernel(
    const int cblk_count,
    const j2k_cblk * const cblks,
    const j2k_band * const bands,
    const j2k_resolution * const res,
    float * const distortions,
    const unsigned int * const byte_sizes,
    const float * const coefficients
) {
    // index of this warp's codeblock
    const int cblk_idx = blockIdx.x * DISTORTION_WARPS_PER_TBLOCK + threadIdx.y;
    if(cblk_idx >= cblk_count) {
        // one of last warps in last threadblock = do nothing
        return;
    }
    
    // shared memory (for all warps) for raw magnitudes
    __shared__ float coefficients_all[DISTORTION_WARPS_PER_TBLOCK * 32];
    float * const coefficients_warp = coefficients_all + threadIdx.y * 32;
    
    // size of the codeblock
    const unsigned int cblk_size_x = cblks[cblk_idx].size.width;
    const unsigned int cblk_size_y = cblks[cblk_idx].size.height;
    
    // quantization coefficient for codeblock's band and its inverse
    const unsigned int cblk_band_idx = cblks[cblk_idx].band_index;
    const float stepsize = bands[cblk_band_idx].stepsize;
    
    // Each thread computes distortion caused by coding only bitplanes 
    // with indices greater than threadIdx.x:
    // Prepare divisor, which discards bits which are not coded
    const float bit_discard = 1 << threadIdx.x;
    const float reconstruction = stepsize * bit_discard;
    const float reconstruction_half = reconstruction * 0.5f;
    const float precision_loss = __fdividef(1.0f, reconstruction);
    
    // stride of codeblock pixels (band width) 
    // (rounded codeblock width is subtracted from the stride to move 
    // from the end of one row to begin of next row)
    const unsigned int cblk_stride_y = bands[cblk_band_idx].size.width
                                     - ((cblk_size_x + 31) & ~31);
    
    // pointer to first pixel to be loaded by this thread in first 
    // row of the codeblock (thread index is pre-added to the pointer)
    const float * next_coefficient = coefficients
                                   + cblks[cblk_idx].data_index
                                   + threadIdx.x;
    
    // acumulator for sum over squared differences between original 
    // magnitudes and corresponding magnitudes with some bits discarded
    float distortion = 0.0f;
    
    // read all rows of magnitudes from the codeblock
    for(int remaining_rows = cblk_size_y; remaining_rows--; ) {
        // load all coefficents of the row (up to 32 coefficients at once)
        for(int x = 0; x < cblk_size_x; x += 32) {
            // load magnitude if not out of bounds
            coefficients_warp[threadIdx.x] = threadIdx.x < cblk_size_x - x
                    ? fabsf(*next_coefficient) : 0.0f;
            
            // advance input pointer
            next_coefficient += 32;
            
            // all threads add distortions of newly loaded coefficients 
            // to their sums (each thread processes distortion of one bitplane)
            for(int c = 0; c < 32; c++) {
                // load next coefficient
                const float original = coefficients_warp[c];
                
                // compute low-precision encoded version of the coefficient 
                // after quantization and bitplane truncation
                // (each thread computes truncation of different bit count)
                const float encoded = floorf(original * precision_loss);
                
                // compute reconstructed coefficient
                const float reconstructed = 0.0f == encoded
                        ? 0.0f : encoded * reconstruction + reconstruction_half;
                
//                 if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
//                     printf("O=%f E=%f R=%f\n", original, encoded, reconstructed);
                        
                // add squared difference between original and reconstructed
                // coefficients to the total distortion
                const float diff = reconstructed - original;
                distortion += diff * diff;
            }
        }
        // advance pointer to next row
        next_coefficient += cblk_stride_y;
    }
    
    // add weight of band to the distortion
    distortion *= bands[cblk_band_idx].visual_weight;
    
    // offset of codeblock's byte sizes output distortions
    const unsigned int cblk_trunc_point_offset = cblks[cblk_idx].trunc_index;
    
    // number of bitplanes in the codeblock
    const unsigned int cblk_bpln_count = cblks[cblk_idx].bitplane_count;
    
    // all the work of this thread was useless if number of bitplanes is less
    // than threadIdx.x, because it computed distortion of skipping more 
    // bitplanes than there are :(
    if(threadIdx.x <= cblk_bpln_count) {
        // index of last of (up to) three truncation points, whose distortions 
        // were computed by this thread (for the total first distortion 
        // corresponds to "not coding at all", but the second one corresponds 
        // to coding first bitplane only - in cleanup pass)
        const int trunc_idx = cblk_trunc_point_offset
                            + 3 * (cblk_bpln_count - threadIdx.x);
        
        // pointer to output distortions for this thread
        float * const distortion_out = distortions + trunc_idx;
        
        // save distortions
        distortion_out[-2] = distortion;
        distortion_out[-1] = distortion;
        distortion_out[-0] = distortion;
    }
}


/// Left turn test (given 3 points in 2D).
/// @param pax  packed x coordinate (in 24 LSBs) of point A
/// @param pbx  packed x coordinate (in 24 LSBs) of point B
/// @param pcx  packed x coordinate (in 24 LSBs) of point C
/// @param ay   y coordinate of point A
/// @param by   y coordinate of point B
/// @param cy   y coordinate of point C
/// @return true if A->B->C is left turn (Imagine yourself standing in point A, 
///         looking towards B, and determining if the point C is to your left.)
__device__ static bool is_left_turn(
            const unsigned int pax,
            const unsigned int pbx,
            const unsigned int pcx,
            const float ay,
            const float by,
            const float cy
) {
    // unpack the actual x-coordinates (packed with index of the point)
    const float ax = pax & 0xFFFFFF;
    const float bx = pbx & 0xFFFFFF;
    const float cx = pcx & 0xFFFFFF;
    
    // use sign of determinant of two-vector matrix as the result
    return (bx - ax) * (cy - by) > (cx - bx) * (by - ay);
}


/// Given orderd list of 2D points (ordered by x coordinate, ascending), 
/// this removes those points, which are not part of lower part of convex 
/// hull of the points. Works in-place.
/// @param x  array of x coordinates
/// @param y  array of y coordinates
/// @param count  count of input points (assumes 
/// @param last_x_out  last convex hull point x-coordinate (or 0 no points)
/// @param last_y_out  last convex hull point y-coordinate (or 0 no points)
__device__ static int remove_nonconvex(
    unsigned int * const x,
    float * const y,
    const int count,
    unsigned int & last_x_out,
    float & last_y_out
) {
    // index of last output point
    int out_idx = 0;
    
    // keep last three points in registers
    unsigned int ax, bx = 0, cx = 0;
    float ay, by = 0, cy = 0;
    
    // keep adding points to partial convex hull (one point in each iteration)
    for(int in_idx = 0; in_idx < count; in_idx++) {
        // shift last three points, reading next one from array
        ax = bx;
        ay = by;
        bx = cx;
        by = cy;
        cx = x[in_idx] + (in_idx << 24);  // extra: pack the point with index
        cy = y[in_idx];
        
        // discard those mid-points, which are not convex
        if(out_idx >= 2) {
            // Non-convex point on lower part of convex hull can be determined
            // using its two neighbors: if the left-center-right points form
            // non-left turn, the center point is not a part of the convex hull.
            while(!is_left_turn(ax, bx, cx, ay, by, cy)) {
                // discard the center point (replace it with the first point)
                bx = ax;
                by = ay;
                out_idx--;
                if(out_idx >= 2) {
                    // ... and read new point from the array
                    ax = x[out_idx - 2];
                    ay = y[out_idx - 2];
                } else {
                    // ... or stop if there are only 2 points 
                    // in the partial convex hull
                    break;
                }
            }
        }
        
        // write new last point into the next output position in the buffer
        x[out_idx] = cx;
        y[out_idx] = cy;
        out_idx++;
    }
    
    // return last output points coordinates and count of convex hull points
    last_x_out = cx;
    last_y_out = cy;
    return out_idx;
}


/// Reduces lists of truncation points of each codeblock (keeps only those 
/// points, which are part of convex hull). Each thread reduces points 
/// of one codeblock.
/// @param cblk_count  count of codeblocks
/// @param cblks  array of all codeblock info structures
/// @param distortions  array for distortions of all truncation points
/// @param byte_sizes  array of byte sizes for truncation points
__global__ static void trunc_point_reduce(
    const int cblk_count,
    j2k_cblk * const cblks,
    float * const distortions,
    unsigned int * const byte_sizes
) {
    // get and check index of thread's codeblock
    const int cblk_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(cblk_idx >= cblk_count) {
        return;
    }
    
    // get info about codeblock;s truncation point list
    const int trunc_offset = cblks[cblk_idx].trunc_index;
    const int trunc_count = cblks[cblk_idx].trunc_count;
    float * const cblk_distortions = distortions + trunc_offset;
    unsigned int * const cblk_rates = byte_sizes + trunc_offset;
    
    // reduce list of truncation points of the codeblock
    unsigned int last_rate;
    float last_distortion;
    int out_count = remove_nonconvex(
                    cblk_rates,
                    cblk_distortions,
                    trunc_count,
                    last_rate,
                    last_distortion
    );
    
    // update the count of truncation points for the codeblock
    cblks[cblk_idx].trunc_count = out_count;
    
    // add few copies of last truncation point to the end for the search kernel
    // to be able to load groups of 4 truncation points without further checks
    while(out_count & 3) {
        cblk_distortions[out_count] = last_distortion;
        cblk_rates[out_count] = last_rate;
        out_count++;
    }
}


/** 
 * Initializes rate control stuff.
 * @return 0 for success, nonzero for failure
 */
int
j2k_rate_control_init() {
    // configure all kernels to use more shared memory.
    const cudaFuncCache s = cudaFuncCachePreferShared;
    if(cudaSuccess != cudaFuncSetCacheConfig(quality_init_kernel, s)) {
        return -1;
    }
    if(cudaSuccess != cudaFuncSetCacheConfig(quality_check_kernel, s)) {
        return -2;
    }
    if(cudaSuccess != cudaFuncSetCacheConfig(quality_search_kernel, s)) {
        return -3;
    }
    if(cudaSuccess != cudaFuncSetCacheConfig(truncation_kernel, s)) {
        return -4;
    }
    if(cudaSuccess != cudaFuncSetCacheConfig(distortion_estimation_kernel, s)) {
        return -5;
    }
    if(cudaSuccess != cudaFuncSetCacheConfig(trunc_point_reduce, s)) {
        return -6;
    }
    
    // indicate success
    return 0;
}


/// Reduces streams of codeblocks according to specified rate control rules.
/// @param enc  pointer to encoder instance
/// @param stream  CUDA stream to run in
/// @return 0 for success, nonzero for failure
int 
j2k_rate_control_reduce(struct j2k_encoder* enc, cudaStream_t stream) {
    // apply rate control if required
    if(enc->max_byte_count && CM_LOSSY_FLOAT == enc->params.compression) {
        // initialize limits of exponent of the quality 
        quality_init_kernel<<<1, 1, 0, stream>>>();
        
        // compute distortion estimates
        const int dist_tblks = (enc->cblk_count + DISTORTION_WARPS_PER_TBLOCK - 1)
                             / DISTORTION_WARPS_PER_TBLOCK;
        const dim3 dist_tblk_size(32, DISTORTION_WARPS_PER_TBLOCK);
        distortion_estimation_kernel<<<dist_tblks, dist_tblk_size, 0, stream>>>(
                enc->cblk_count,
                enc->d_cblk,
                enc->d_band,
                enc->d_resolution,
                enc->d_trunc_distortions,
                enc->d_trunc_sizes,
                (const float*)enc->d_data_dwt
        );
        
        // reduce truncation points to feasible ones in all lists
        const int trunc_block_threads = 128;
        const int trunc_blocks = (enc->cblk_count + trunc_block_threads - 1)
                               / trunc_block_threads;
        trunc_point_reduce<<<trunc_blocks, trunc_block_threads, 0, stream>>>(
            enc->cblk_count,
            enc->d_cblk,
            enc->d_trunc_distortions,
            enc->d_trunc_sizes
        );
        
        const int shmem_bytes = sizeof(unsigned int) * trunc_block_threads;
        
        // find best quality within limits (for all codeblocks)
        for(int r = MAX_QUALITY_EXPONENT - MIN_QUALITY_EXPONENT; r > 1; r /= 2) {
            quality_search_kernel<<<trunc_blocks, trunc_block_threads, shmem_bytes, stream>>>(
                    enc->d_trunc_sizes,
                    enc->d_trunc_distortions,
                    enc->d_cblk,
                    enc->cblk_count
            );
            quality_check_kernel<<<1, 1, 0, stream>>>(
                    enc->max_byte_count,
                    enc->max_byte_count / 1000 // size tolerance
            );
        }
        
        // reduce all codestreams according to the quality
        truncation_kernel<<<trunc_blocks, trunc_block_threads, 0, stream>>>(
                enc->d_cblk,
                enc->cblk_count,
                enc->d_trunc_sizes,
                enc->d_trunc_distortions
        );
    }
    
    return 0;
}


/// Initializes visual weights of bands.
/// @param enc  pointer to encoder instance
void
j2k_rate_control_init_weights(struct j2k_encoder* enc) {
    for(int band_idx = 0; band_idx < enc->band_count; band_idx++) {
        j2k_band * const band = enc->band + band_idx;
        const j2k_resolution * const res
                = enc->resolution + band->resolution_index;
        
        const double orientation_weight = band->type == LL
                ? 1.0 : (band->type == HH ? 4.0 : 2.0);
        const double resolution_weight = 1 << res->level;
        const static double mct_norms[3] = {1.732, 1.805, 1.573};
        const double comp_weight = enc->params.mct
                ? mct_norms[res->component_index] : 1.0;
        const double weight = orientation_weight * resolution_weight;
        
        band->visual_weight = comp_weight * comp_weight / (weight * weight);
    }
}

