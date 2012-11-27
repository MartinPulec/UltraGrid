///
/// @file    j2kd_dwt_float.cu (originally rdwt97.cu)
/// @brief   CUDA implementation of reverse 9/7 2D DWT.
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @date    2011-02-03 21:59
///
///
/// Copyright (c) 2011 Martin Jirman
/// All rights reserved.
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
///
///     * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
///     * Redistributions in binary form must reproduce the above copyright
///       notice, this list of conditions and the following disclaimer in the
///       documentation and/or other materials provided with the distribution.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
/// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
/// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
/// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
/// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
/// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
/// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.
///


#include "j2kd_dwt.h"
#include <stdio.h>



namespace cuj2kd {
    
    
    


/// Specification of 9/7 RDWT
struct RDWT97 {
    // type of transformed samples
    typedef float T;
    
    // compile time constants related to this type of DWT
    enum {
        LIFTING_STEP_COUNT = 2,
    };
    
    __device__ static inline T unupdate1(const T c, const T p, const T n) {
        return c + 0.05298011854f * (p + n);
    }
    
    __device__ static inline T unupdate2(const T c, const T p, const T n) {
        return c - 0.4435068522f * (p + n);
    }
    
    __device__ static inline T unpredict1(const T c, const T p, const T n) {
        return c + 1.586134342f * (p + n);
    }
    
    __device__ static inline T unpredict2(const T c, const T p, const T n) {
        return c - 0.8829110762f * (p + n);
    }
    
    // update undo
    __device__ static inline T unupdate(int step, T & lo, const T hi0, const T hi1) {
        return lo = step ? unupdate1(lo, hi0, hi1) : unupdate2(lo, hi0, hi1);
    }
    
    // predict undo
    __device__ static inline T unpredict(int step, T & hi, const T lo0, const T lo1) {
        return hi = step ? unpredict1(hi, lo0, lo1) : unpredict2(hi, lo0, lo1);
    }
    
    // lowpass sample scale undo
    __device__ static inline T loUnscale(const T lo) {
        return lo * 1.23017410491400f;
    }
    
    // highpass sample scale undo
    __device__ static inline T hiUnscale(const T hi) {
        return hi * 0.812893066115961f;
    }
    
    // saves a pair of output coefficients
    __device__ static inline void save(T * const out, const int stride, 
                                       const int y, const int x,
                                       const T & lVal, const T & hVal) {
        *(float2*)(out + y * stride + x) = make_float2(lVal, hVal);
    }
    
    // transforming state machine
    class StateMachine {
    private:
        // TODO: add attributes
        T lPrev, pPrev, uPrev, hPrev;
    public:
        __device__ inline void init(const T * const samples, const int stride) {
            const T h6 = hiUnscale(samples[6 * stride]);
            const T h4 = hiUnscale(samples[4 * stride]);
            const T h2 = hiUnscale(samples[2 * stride]);
            const T h0 = hiUnscale(samples[0 * stride]);
            const T u1 = unupdate2(loUnscale(samples[1 * stride]), h0, h2);
            const T u3 = unupdate2(loUnscale(samples[3 * stride]), h2, h4);
            const T u5 = unupdate2(loUnscale(samples[5 * stride]), h4, h6);
            const T p2 = unpredict2(h2, u1, u3);
            const T p4 = unpredict2(h4, u3, u5);
            const T l3 = unupdate1(u3, p2, p4);
            hPrev = h6;
            uPrev = u5;
            pPrev = p4;
            lPrev = l3;
        }
        __device__ inline void transform(const T lIn, const T hIn, T & lOut, T & hOut) {
            const T hTemp = hiUnscale(hIn);
            const T uTemp = unupdate2(loUnscale(lIn), hPrev, hTemp);
            const T pTemp = unpredict2(hPrev, uPrev, uTemp);
            const T lTemp = unupdate1(uPrev, pPrev, pTemp);
            hOut = unpredict1(pPrev, lPrev, lTemp);
            lOut = lPrev;
            hPrev = hTemp;
            uPrev = uTemp;
            pPrev = pTemp;
            lPrev = lTemp;
        }
    };
};





/// Specification of 5/3 RDWT
struct RDWT53 {
    // type of transformed samples
    typedef int T;
    
    // compile time constants related to this type of DWT
    enum { LIFTING_STEP_COUNT = 1 };
    
    // update undo
    __device__ static inline T unupdate(int step, T & lo, const T hi0, const T hi1) {
        return lo -= (hi0 + hi1 + 2) >> 2;  // F.3, page 118, ITU-T Rec. T.800 final draft
    }
    
    // predict undo
    __device__ static inline T unpredict(int step, T & hi, const T lo0, const T lo1) {
        return hi += (lo0 + lo1) >> 1;      // F.4, page 118, ITU-T Rec. T.800 final draft
    }
    
    // lowpass sample scale undo
    __device__ static inline T loUnscale(const T lo) { return lo; }
    
    // highpass sample scale undo
    __device__ static inline T hiUnscale(const T hi) { return hi; }
    
    // saves a pair of output coefficients
    __device__ static inline void save(T * const out, const int stride, 
                                       const int y, const int x,
                                       const T & lVal, const T & hVal) {
        *(int2*)(out + y * stride + x) = make_int2(lVal, hVal);
    }
    
    // transforming state machine
    class StateMachine {
    private:
        // TODO: add attributes
        T lPrev, hPrev;
    public:
        __device__ inline void init(const T * const samples, const int stride) {
            hPrev = samples[stride * 2];
            lPrev = samples[stride] - ((samples[0] + hPrev + 2) >> 2);
        }
        __device__ inline void transform(const T lIn, const T hIn, T & lOut, T & hOut) {
            lOut = lPrev;
            lPrev = lIn - ((hPrev + hIn + 2) >> 2);
            hOut = hPrev + ((lPrev + lOut) >> 1);
            hPrev = hIn;
        }
    };

};




/// Handles mirroring of image at edges in a DWT correct way.
/// @param d      a position in the image (will be replaced by mirrored d)
/// @param sizeD  size of the image along the dimension of 'd'
__device__ inline void mirrorImpl(int & d, const int & sizeD) {
    if(sizeD > 1) {
        const int sign = (d < 0) ? 1 : 0;
        if(sign || (d >= sizeD)) {
            const int distance = sign ? (-1 - d) : (d - sizeD);
            const int phase = (distance / (sizeD - 1)) & 1;
            const int remainder = distance % (sizeD - 1);
            d = (phase ^ sign) ? (remainder + 1) : (sizeD - 2 - remainder);
        }
    } else {
        d = 0;
    }
}


__device__ inline int mirror(const int d, const int size) {
    int result = d;
    mirrorImpl(result, size);
    return result / 2;
}
    

template<typename T>
__device__ inline T sampleLoad(const T * const * srcLines, int x, int y,
                               int yLine0, const T * src, int srcStride,
                               bool mirroring) {
    // either use vertically mirrored line pointers 
    // or direcly compute source index
    return mirroring
            ? ((volatile T * const *)srcLines)[y + yLine0][x]
            : ((volatile T *)src)[x + (y >> 1) * srcStride];
}


/// Horizontal RDWT transform.
/// @tparam RDWT  Reverse DWT specification
/// @tparam STRIDE  shared buffers stride (difference between indices 
///                 of two vertically neighboring samples)
/// @tparam SAMPLE_COUNT  number of sample pairs per thread (high- and lowpass)
/// @param lTemp shared buffer for lowpass samples exchange
/// @param hTemp shared buffer for highpass samples exchange
/// @param lSamples  array with thread's lowpass samples
/// @param hSamples  array with thread's highpass samples
template <typename RDWT, int STRIDE, int SAMPLE_COUNT>
__device__ inline void rdwtH(typename RDWT::T * const lTemp, 
                             typename RDWT::T * const hTemp, 
                             typename RDWT::T lSamples[SAMPLE_COUNT],
                             typename RDWT::T hSamples[SAMPLE_COUNT]) {
    // offset of this thread's samples in shared buffers
    const int x = threadIdx.x;
    
    // copy samples into the shared buffer
    #pragma unroll
    for(int y = 0; y < SAMPLE_COUNT; y++) {
        hTemp[x + y * STRIDE] = hSamples[y];
    }
    
    // transform samples
    #pragma unroll
    for(int stepIdx = 0; stepIdx < RDWT::LIFTING_STEP_COUNT; stepIdx++) {
        __syncthreads();
        #pragma unroll
        for(int y = 0; y < SAMPLE_COUNT; y++) {
            const int i = x + y * STRIDE;
            lTemp[i] = RDWT::unupdate(stepIdx, lSamples[y], hTemp[i - 1], hSamples[y]);
        }
        __syncthreads();
        #pragma unroll
        for(int y = 0; y < SAMPLE_COUNT; y++) {
            const int i = x + y * STRIDE;
            hTemp[i] = RDWT::unpredict(stepIdx, hSamples[y], lSamples[y], lTemp[i + 1]);
        }
    }
}





/// PAIRS_Y must be at least 4 for 9/7 and at least 2 for 5/3
template<int THREADS_X, int PAIRS_Y, typename RDWT, bool CHECK_X, bool CHECK_Y>
__device__ inline void rdwtTransform(
        const typename RDWT::T * const * const llLines,
        const typename RDWT::T * const * const hlLines,
        const typename RDWT::T * const * const lhLines,
        const typename RDWT::T * const * const hhLines,
        const typename RDWT::T * const llSrc,
        const typename RDWT::T * const hlSrc,
        const typename RDWT::T * const lhSrc,
        const typename RDWT::T * const hhSrc,
        const int lStride,
        const int hStride,
        const int2 * const mirrorX,
        const int outSizeX,
        const int outSizeY,
        const int outStrideY,
        typename RDWT::T * const out,
        typename RDWT::T * const lTemp,
        typename RDWT::T * const hTemp,
        const int stepsY
) {
    // type of samples
    typedef typename RDWT::T T;
    
    // compile time constants
    enum {
        INITIAL_LINE_COUNT = RDWT::LIFTING_STEP_COUNT * 4 - 1,
        BOUNDARY_PAIRS = RDWT::LIFTING_STEP_COUNT,
        TEMP_STRIDE = THREADS_X,
        MIRROR_IDX_OFFSET = RDWT::LIFTING_STEP_COUNT * 2
    };
    
    // number of central column pairss (non-boundary ones)
    const int COLUMN_PAIRS = THREADS_X - 2 * BOUNDARY_PAIRS;
    
    // index of thread's column in shared buffers
    const int colX = threadIdx.x;
    
    // index of this thread's source column relative to threadblock
    const int tidX = colX - BOUNDARY_PAIRS;
    
    // output sample coordinates
    const int pairX = blockIdx.x * COLUMN_PAIRS + tidX;
    const int outX = pairX * 2;
    int outY = (blockIdx.y * PAIRS_Y * stepsY) * 2;
    
    // true if this thread transforms a pair of boundary columns
    const bool isBoundary = tidX < 0 || tidX >= COLUMN_PAIRS;
    
    // true if this thread saves results of its transformation 
    // (or false if it is only a boundary thread)
    bool isSaving = outX < outSizeX && !isBoundary;
    
    // mirrored coordinates of lo and hi source columns
    const int2 mirroredSrcX = CHECK_X
            ? mirrorX[pairX + MIRROR_IDX_OFFSET] : make_int2(pairX, pairX);
    const int lSrcX = mirroredSrcX.x;
    const int hSrcX = mirroredSrcX.y;
    
    // vertical transform objects for thread's hi and lo columns
    typename RDWT::StateMachine loTransf;
    typename RDWT::StateMachine hiTransf;
    
    // initialization samples
    T lSamples[INITIAL_LINE_COUNT], hSamples[INITIAL_LINE_COUNT];
    
    // load vertically HI initial lines (with horizontal unscaling)
    #pragma unroll
    for(int y = 0; y < RDWT::LIFTING_STEP_COUNT * 2; y++) {
        // load samples
        const int srcY = outY + 2 * (y - RDWT::LIFTING_STEP_COUNT) + 1;
        const T lhSample = sampleLoad(lhLines, lSrcX, srcY, MIRROR_IDX_OFFSET, lhSrc, lStride, CHECK_Y);
        const T hhSample = sampleLoad(hhLines, hSrcX, srcY, MIRROR_IDX_OFFSET, hhSrc, hStride, CHECK_Y);
        
        // write them into the shared buffer
        lSamples[2 * y] = RDWT::loUnscale(lhSample);
        hSamples[2 * y] = RDWT::hiUnscale(hhSample);
    }
    
    // load vertically LO initial lines (with horizontal unscaling)
    #pragma unroll
    for(int y = 1; y < RDWT::LIFTING_STEP_COUNT * 2; y++) {
        // load samples
        const int srcY = outY + 2 * (y - RDWT::LIFTING_STEP_COUNT);
        const T llSample = sampleLoad(llLines, lSrcX, srcY, MIRROR_IDX_OFFSET, llSrc, lStride, CHECK_Y);
        const T hlSample = sampleLoad(hlLines, hSrcX, srcY, MIRROR_IDX_OFFSET, hlSrc, hStride, CHECK_Y);

        // write them into the shared buffer
        lSamples[2 * y - 1] = RDWT::loUnscale(llSample);
        hSamples[2 * y - 1] = RDWT::hiUnscale(hlSample);
    }
    
    // horizontally transform initial HI lines
    #pragma unroll
    for(int stepIdx = 0; stepIdx < RDWT::LIFTING_STEP_COUNT; stepIdx++) {
        __syncthreads();
        #pragma unroll
        for(int y = 0; y < INITIAL_LINE_COUNT; y++) {
            const int i = colX + TEMP_STRIDE * y;
            RDWT::unupdate(stepIdx, lTemp[i], hTemp[i - 1], hTemp[i]);
        }
        __syncthreads();
        #pragma unroll
        for(int y = 0; y < INITIAL_LINE_COUNT; y++) {
            const int i = colX + TEMP_STRIDE * y;
            RDWT::unpredict(stepIdx, hTemp[i], lTemp[i], lTemp[i + 1]);
        }
    }
    
    // horizontally transform initial lines (includes sync)
    rdwtH<RDWT, THREADS_X, INITIAL_LINE_COUNT>(lTemp, hTemp, lSamples, hSamples);
    
    // push HI and LO initial lines into transform objects
    __syncthreads();
    loTransf.init(lSamples, 1);
    hiTransf.init(hSamples, 1);
    
    // start transforming
    for(int remainingStepsY = stepsY; remainingStepsY--; ) {
        // wait for previous iteration or initialization to be done
        __syncthreads();
        
        // samples loaded and horizontally transformed in this iteration
        T lSamples[PAIRS_Y * 2], hSamples[PAIRS_Y * 2];
        
        // load more samples into the shared transform buffer
        #pragma unroll
        for(int y = 0; y < PAIRS_Y; y++) {
            // get pointers to 4 source lines
            const int srcY = outY + 2 * (y + RDWT::LIFTING_STEP_COUNT);
            const T llSample = sampleLoad(llLines, lSrcX, srcY + 0, MIRROR_IDX_OFFSET, llSrc, lStride, CHECK_Y);
            const T hlSample = sampleLoad(hlLines, hSrcX, srcY + 0, MIRROR_IDX_OFFSET, hlSrc, hStride, CHECK_Y);
            const T lhSample = sampleLoad(lhLines, lSrcX, srcY + 1, MIRROR_IDX_OFFSET, lhSrc, lStride, CHECK_Y);
            const T hhSample = sampleLoad(hhLines, hSrcX, srcY + 1, MIRROR_IDX_OFFSET, hhSrc, hStride, CHECK_Y);
            
            // load 4 source samples from 4 bands with horiozntal unscaling
            lSamples[y * 2 + 0] = RDWT::loUnscale(llSample);
            lSamples[y * 2 + 1] = RDWT::loUnscale(lhSample);
            hSamples[y * 2 + 0] = RDWT::hiUnscale(hlSample);
            hSamples[y * 2 + 1] = RDWT::hiUnscale(hhSample);
        }
        
        // horizontally transform newly loaded lines (includes syncthreads)
        rdwtH<RDWT, THREADS_X, PAIRS_Y * 2>(lTemp, hTemp, lSamples, hSamples);
        
        // start putting samples into vertical transformers and saving results
        #pragma unroll
        for(int pairY = 0; pairY < PAIRS_Y; pairY++) {
            // load samples
            const T llIn = lSamples[pairY * 2 + 0];
            const T lhIn = lSamples[pairY * 2 + 1];
            const T hlIn = hSamples[pairY * 2 + 0];
            const T hhIn = hSamples[pairY * 2 + 1];
            
            // transform samples
            T llOut, hlOut, lhOut, hhOut;
            loTransf.transform(llIn, lhIn, llOut, lhOut);
            hiTransf.transform(hlIn, hhIn, hlOut, hhOut);
            
            // save first two samples, update Y and check for output end
            if(isSaving) {
                RDWT::save(out, outStrideY, outY, outX, llOut, hlOut);
            }
            outY++;
            if(CHECK_Y && outY >= outSizeY) {
                isSaving = false;
            }
            
            // save other sample pair, update Y and check end again
            if(isSaving) {
                RDWT::save(out, outStrideY, outY, outX, lhOut, hhOut);
            }
            outY++;
            if(CHECK_Y && outY >= outSizeY) {
                isSaving = false;
            }
        }
        
        // stop if at the end
        if(CHECK_Y && outY >= outSizeY) {
            return;
        }
    }
}



template <typename T, int BOUNDARY_SIZE>
__global__ static void rdwtPrecomputeKernel(const int outSizeX,
                                      const int outSizeY,
                                      const int lStride,
                                      const int hStride,
                                      const T * const llSrc,
                                      const T * const hlSrc,
                                      const T * const lhSrc, 
                                      const T * const hhSrc, 
                                      int2 * const mirrorX,
                                      const T ** const llLines,
                                      const T ** const hlLines,
                                      const T ** const lhLines,
                                      const T ** const hhLines,
                                      const int lineCount,
                                      const int mirrorCount) {
    // get global id of this thread
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // shoud this thread compute mirroring of X coordinates?
    if(tid < mirrorCount) {
        const int pairX = 2 * (tid - BOUNDARY_SIZE);
        const int lSrcX = mirror(pairX + 0, outSizeX);
        const int hSrcX = mirror(pairX + 1, outSizeX);
        mirrorX[tid] = make_int2(lSrcX, hSrcX);
    }
    
    // should this thread compute line pointers?
    if(tid < lineCount) {
        const int y = mirror(tid - BOUNDARY_SIZE, outSizeY);
        llLines[tid] = llSrc + lStride * y;
        hlLines[tid] = hlSrc + hStride * y;
        lhLines[tid] = lhSrc + lStride * y;
        hhLines[tid] = hhSrc + hStride * y;
    }
}



/// PAIRS_Y must be at least 4 for 9/7 and at least 2 for 5/3
template <int THREADS_X, int PAIRS_Y, typename RDWT>
__launch_bounds__(THREADS_X, 640 / THREADS_X)
__global__ static void rdwtTransformKernel(
                            const typename RDWT::T * const * const llLines,
                            const typename RDWT::T * const * const hlLines,
                            const typename RDWT::T * const * const lhLines,
                            const typename RDWT::T * const * const hhLines,
                            const typename RDWT::T * const llSrc,
                            const typename RDWT::T * const hlSrc,
                            const typename RDWT::T * const lhSrc,
                            const typename RDWT::T * const hhSrc,
                            const int lStride,
                            const int hStride,
                            const int2 * const mirrorX,
                            const int outSizeX,
                            const int outSizeY,
                            const int outStride,
                            typename RDWT::T * const out,
                            const int stepsY) {
    // shared memory buffer for horizontal RDWT
    enum {TEMP_ITEM_COUNT = 2 * PAIRS_Y * THREADS_X};
    __shared__ typename RDWT::T temp[TEMP_ITEM_COUNT * 2];
    
    // limits of source sample coordinates read by this threadblock
    const int boundarySize = RDWT::LIFTING_STEP_COUNT * 2;
    const int workX = THREADS_X * 2;
    const int beginX = blockIdx.x * workX;
    const int minX = beginX - boundarySize;
    const int maxX = minX + workX + boundarySize;
    const int workY = 2 * PAIRS_Y * stepsY;
    const int beginY = blockIdx.y * workY;
    const int minY = beginY - boundarySize;
    const int maxY = beginY + workY + boundarySize;
    
    // does this threadblock process sample block near image boundary?
    const bool nearBoundaryX = minX < 0 || maxX >= outSizeX;
    const bool nearBoundaryY = minY < 0 || maxY >= outSizeY;
    
    // either use mirroring or directly use cooridnates if not near boundary
    if(nearBoundaryX) {
        if(nearBoundaryY) {
            rdwtTransform<THREADS_X, PAIRS_Y, RDWT, true, true>(
                llLines, hlLines, lhLines, hhLines, llSrc, hlSrc, lhSrc, hhSrc,
                lStride, hStride, mirrorX, outSizeX, outSizeY, outStride,
                out, temp + 0 * TEMP_ITEM_COUNT, temp + 1 * TEMP_ITEM_COUNT, 
                stepsY
            );
        } else {
            rdwtTransform<THREADS_X, PAIRS_Y, RDWT, true, false>(
                llLines, hlLines, lhLines, hhLines, llSrc, hlSrc, lhSrc, hhSrc,
                lStride, hStride, mirrorX, outSizeX, outSizeY, outStride,
                out, temp + 0 * TEMP_ITEM_COUNT, temp + 1 * TEMP_ITEM_COUNT, 
                stepsY
            );
        }
    } else {
        if(nearBoundaryY) {
            rdwtTransform<THREADS_X, PAIRS_Y, RDWT, false, true>(
                llLines, hlLines, lhLines, hhLines, llSrc, hlSrc, lhSrc, hhSrc,
                lStride, hStride, mirrorX, outSizeX, outSizeY, outStride,
                out, temp + 0 * TEMP_ITEM_COUNT, temp + 1 * TEMP_ITEM_COUNT, 
                stepsY
            );
        } else {
            rdwtTransform<THREADS_X, PAIRS_Y, RDWT, false, false>(
                llLines, hlLines, lhLines, hhLines, llSrc, hlSrc, lhSrc, hhSrc,
                lStride, hStride, mirrorX, outSizeX, outSizeY, outStride,
                out, temp + 0 * TEMP_ITEM_COUNT, temp + 1 * TEMP_ITEM_COUNT, 
                stepsY
            );
        }
    }
}



/// Division with rounding up.
/// @param n  numerator
/// @param d  denominator
/// @return  n/d rounded upwards
inline int div_round_up(const int n, const int d) {
    return (n / d) + ((n % d) ? 1 : 0);
}


template<int THREADS_X, typename RDWT>
static void rdwtLaunch(
        const Band & hl,
        const Band & lh,
        const Band & hh,
        const void * const hPtr,
        const void * const llPtr,
        const XY & llBegin,
        const XY & llEnd,
        const int llStride,
        void * outPtr,
        const int outStride,
        GPUBuffer & temp,
        cudaStream_t stream
) {
    // type of samples
    typedef typename RDWT::T T;
    
    // compile time constants
    enum { 
        PAIRS_Y = RDWT::LIFTING_STEP_COUNT * 2, // number of sample row pairs processed in each iteration
        BOUNDARY_SIZE = RDWT::LIFTING_STEP_COUNT * 2
    };
    
    // get sizes of bands
    const XY hlSize = hl.pixEnd - hl.pixBegin;
    const XY lhSize = lh.pixEnd - lh.pixBegin;
    const XY hhSize = hh.pixEnd - hh.pixBegin;
    const XY llSize = llEnd - llBegin;
    
    // check stuff
    if(llSize.x != lhSize.x) {
        throw Error(J2KD_ERROR_UNKNOWN, "LL width != LH width");
    }
    if(hlSize.x != hhSize.x) {
        throw Error(J2KD_ERROR_UNKNOWN, "HL width != HH width");
    }
    if(hlSize.y != llSize.y) {
        throw Error(J2KD_ERROR_UNKNOWN, "HL height != LL height");
    }
    if(hhSize.y != lhSize.y) {
        throw Error(J2KD_ERROR_UNKNOWN, "HH height != LH height");
    }
    if(hh.outPixStride != hl.outPixStride) {
        throw Error(J2KD_ERROR_UNKNOWN, "HH stride != HL stride");
    }
    if(hl.outPixStride != llStride) {
        throw Error(J2KD_ERROR_UNKNOWN, "LH stride != LL stride");
    }
    if(outStride & 1) {
        throw Error(J2KD_ERROR_UNKNOWN, "DWT output stride is odd");
    }
    
    // strides
    const int lStride = llStride;
    const int hStride = hh.outPixStride;
    
    // pointers to bands
    const T * const llSrc = (const T*)llPtr;
    const T * const hlSrc = (const T*)hPtr + hl.outPixOffset;
    const T * const lhSrc = (const T*)hPtr + lh.outPixOffset;
    const T * const hhSrc = (const T*)hPtr + hh.outPixOffset;
    T * const out = (T*)outPtr;
    
    // output size
    const XY outSize = llSize + hhSize;
    
    // number of iterations of each thread 
    // (each iteration consists of processing PAIRS_Y pairs of sample rows)
    const int totalStepsY = div_round_up(outSize.y, 2 * PAIRS_Y);
    const int stepsY = div_round_up(totalStepsY, 12);
    
    // number of pixels processed by each threadblock
    const int workX = 2 * (THREADS_X - 2 * BOUNDARY_SIZE);
    const int workY = 2 * PAIRS_Y * stepsY;
    
    // launch configuration
    const dim3 tTBlock(THREADS_X);
    const dim3 tGrid(div_round_up(outSize.x, workX), div_round_up(outSize.y, workY));
    
    // memory needed for precomputed line pointers and mirrored X coordinates
    const int mirrorCount = tGrid.x * THREADS_X + BOUNDARY_SIZE;
    const int lineCount = tGrid.y * stepsY * PAIRS_Y * 2 + 2 * BOUNDARY_SIZE;
    const int tempBytes = mirrorCount * sizeof(int2) + 4 * lineCount * sizeof(T*);
    
    // resize the temp buffer and prepare pointers
    const T ** const tempBuffer = (const T**)temp.resize(tempBytes);
    const T ** const llLines = tempBuffer + lineCount * 0;
    const T ** const hlLines = tempBuffer + lineCount * 1;
    const T ** const lhLines = tempBuffer + lineCount * 2;
    const T ** const hhLines = tempBuffer + lineCount * 3;
    int2 * const mirrorX = (int2*)(tempBuffer + lineCount * 4);
    
    // precompute line pointers and mirrored coordinates
    const dim3 pTBlock(64);
    const dim3 pGrid(div_round_up(max(mirrorCount, lineCount), pTBlock.x));
    rdwtPrecomputeKernel<T, BOUNDARY_SIZE><<<pGrid, pTBlock>>>(
            outSize.x, outSize.y, lStride, hStride, llSrc, hlSrc, lhSrc, hhSrc, 
            mirrorX, llLines, hlLines, lhLines, hhLines, lineCount, mirrorCount
    );
    
    // launch the RDWT
    rdwtTransformKernel<THREADS_X, PAIRS_Y, RDWT><<<tGrid, tTBlock>>>(
            llLines, hlLines, lhLines, hhLines, llSrc, hlSrc, lhSrc, hhSrc, 
            lStride, hStride, mirrorX, outSize.x, outSize.y, outStride, out, stepsY
    );
}





DWT::DWT() {
    // TODO: add cache setup
}



static void copyGpu(void * const dest,
                    const void * const src,
                    const size_t size,
                    cudaStream_t & stream) {
    // make sure that both pointers and size are aligned to 16B boundary
    if(15 & (size_t)dest) {
        throw Error(J2KD_ERROR_UNKNOWN, "DWT copy: dest unaligned.");
    }
    if(15 & (size_t)src) {
        throw Error(J2KD_ERROR_UNKNOWN, "DWT copy: src unaligned.");
    }
    if(15 & size) {
        throw Error(J2KD_ERROR_UNKNOWN, "DWT copy: (size & 15) != 0.");
    }
    
    // TODO: replace with custom kernel!!
    checkCudaCall(cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToDevice, stream), "DWT copy");
}   



void DWT::transform(Image& image,
                    IOBufferGPU<u8> & working,
                    GPUBuffer & tempBuffer,
                    cudaStream_t stream) {
    // swap buffers and possibly resize
    working.swap();
    working.outResize(image.bandsPixelCount * 4);  // 4 = size of sample
    
    // for each tile-component
    for(int tCompIdx = image.tComps.count(); tCompIdx--; ) {
        // reference to the tile, tile-component and its coding info
        const TComp & tComp = image.tComps[tCompIdx];
        const Tile & tile = image.tiles[tComp.tileIdx];
        const TCompCoding & cod = image.tCompCoding[tComp.codingIdx];
        
        // size of each component's samples
        const int height = tile.pixEnd.y - tile.pixBegin.y;
        const int outSize = height * tComp.outPixStride * 4;
        
        // transform or only copy?
        if(cod.dwtLevelCount) {
            // pointer to resized temp buffer and output pointer 
            // for this tile-component (and to all input samples)
            int * const temp = (int*)tempBuffer.resize(outSize);
            int * const out = (int*)working.outPtr() + tComp.outPixOffset;
            const int * const in = (int*)working.inPtr();
            
            // compose LL bands in all levels (last level is output)
            for(int resIdx = 0; resIdx < cod.dwtLevelCount; resIdx++) {
                // references to input and output resolutions
                Res & resOut = image.res[tComp.resIdx + resIdx + 1];
                Res & resIn = image.res[tComp.resIdx + resIdx];
                
                // pointers to input bands except the LL band
                const Band * const bands = &image.bands[resOut.bandOffset];
                
                // Is the number of remaining DWT levels to get the result
                // odd? (E.g. true, if this level composes the result.)
                const bool dwtCountOdd = (cod.dwtLevelCount - resIdx) & 1;
                
//                 printf("DWT direction: %s -> %s: ",
//                        resIdx ? (dwtCountOdd ? "temp" : "out") : "in",
//                        dwtCountOdd ? "out" : "temp");
                
                // select pointer to input LL band of this resolution 
                // (either pointer to first LL band or pointer to output 
                // of previous DWT level)
                const int * const llIn = resIdx
                        ? (dwtCountOdd ? temp : out)
                        : in + image.bands[resIn.bandOffset].outPixOffset;
                
                // pointer to lowpass coefficients from this resolution
                int * const llOut = dwtCountOdd ? out : temp;

                // launch the right kind of transform
                if(cod.reversible) {
                    rdwtLaunch<128, RDWT53>(bands[0], bands[1], bands[2],
                            (const void *)in, (const void *)llIn, resIn.begin,
                            resIn.end, resIn.outPixStride, (void*)llOut,
                            resOut.outPixStride, mirror, stream);
                } else {
                    rdwtLaunch<128, RDWT97>(bands[0], bands[1], bands[2],
                            (const void *)in, (const void *)llIn, resIn.begin,
                            resIn.end, resIn.outPixStride, (void*)llOut,
                            resOut.outPixStride, mirror, stream);
                }
            }
        } else {
            // no transform needed - only copy all tile-component samples
            // into correct position in the output buffer
            const int offset = tComp.outPixOffset;
            int * const out = (int*)working.outPtr() + offset;
            const int * const in = (const int*)working.inPtr() + offset;
            copyGpu(out, in, outSize, stream);
        }
    }
}




} // end of namespace cuj2kd





