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
    
    
    
// 9/7 forward DWT lifting schema coefficients
__device__ static const float f97Predict1 = -1.586134342;  ///< fwd 9/7 predict 1
__device__ static const float f97Update1 = -0.05298011854; ///< fwd 9/7 update 1
__device__ static const float f97Predict2 = 0.8829110762;  ///< fwd 9/7 predict 2
__device__ static const float f97Update2 = 0.4435068522;   ///< fwd 9/7 update 2


// 9/7 reverse DWT lifting schema coefficients
__device__ static const float r97update2 = -f97Update2;    ///< undo 9/7 update 2
__device__ static const float r97predict2 = -f97Predict2;  ///< undo 9/7 predict 2
__device__ static const float r97update1 = -f97Update1;    ///< undo 9/7 update 1
__device__ static const float r97Predict1 = -f97Predict1;  ///< undo 9/7 predict 1

// FDWT 9/7 scaling coefficients
__device__ static const float scale97Mul = 1.23017410491400f;
__device__ static const float scale97Div = 1.0 / scale97Mul;



/// Functor which adds scaled sum of neighbors to given central pixel.
struct AddScaledSum {
    const float scale;  // scale of neighbors
    __device__ AddScaledSum(const float scale) : scale(scale) {}
    __device__ void operator()(const float p, float & c, const float n) const {
        c += scale * (p + n);
    }
};



/// Returns index ranging from 0 to num threads, such that first half
/// of threads get even indices and others get odd indices. Each thread
/// gets different index.
/// Example: (for 8 threads)   threadIdx.x:   0  1  2  3  4  5  6  7
///                              parityIdx:   0  2  4  6  1  3  5  7
/// @tparam THREADS  total count of participating threads
/// @return parity-separated index of thread
template <int THREADS>
__device__ inline int parityIdx() {
    return (threadIdx.x * 2) - (THREADS - 1) * (threadIdx.x / (THREADS / 2));
}



// /// size of shared memory for different architectures
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
// const int SHM_SIZE = 48 * 1024;
// #else
// const int SHM_SIZE = 16 * 1024;
// #endif



/// Size of LL band after some count of DWT levels.
/// @param size    initial size
/// @param levels  number of levels
inline int llSize(const int size, const int levels) {
    return (size + (1 << levels) - 1) >> levels;
}
    
    
    
    


/// Buffer (in shared memory of GPU) where block of input image is stored,
/// but odd and even lines are separated. (Generates less bank conflicts when
/// using lifting schema.) All operations expect SIZE_X threads.
/// Also implements basic building blocks of lifting schema.
/// @tparam SIZE_X      width of the buffer excluding two boundaries (Also
///                     a number of threads participating on all operations.)
///                     Must be divisible by 4.
/// @tparam SIZE_Y      height of buffer (total number of lines)
/// @tparam BOUNDARY_X  number of extra pixels at the left and right side
///                     boundary is expected to be smaller than half SIZE_X
///                     Must be divisible by 2.
template <typename T, int SIZE_X, int SIZE_Y, int BOUNDARY_X>
class TransformBuffer {
public:
    enum {
        /// difference between pointers to two vertical neigbors
        VERTICAL_STRIDE = BOUNDARY_X + (SIZE_X / 2)
    };

private:
    enum {
        /// number of shared memory banks - needed for correct padding
#ifdef __CUDA_ARCH__
        SHM_BANKS = ((__CUDA_ARCH__ >= 200) ? 32 : 16),
#else
        SHM_BANKS = 16,  // for host code only - can be anything, won't be used
#endif

        /// size of one of two buffers (odd or even)
        BUFFER_SIZE = VERTICAL_STRIDE * SIZE_Y,

        /// unused space between two buffers
        PADDING = SHM_BANKS - ((BUFFER_SIZE + SHM_BANKS / 2) % SHM_BANKS),

        /// offset of the odd columns buffer from the beginning of data buffer
        ODD_OFFSET = BUFFER_SIZE + PADDING,
    };

    /// buffer for both even and odd columns
    T data[2 * BUFFER_SIZE + PADDING];



    /// Applies specified function to all central elements while also passing
    /// previous and next elements as parameters.
    /// @param count         count of central elements to apply function to
    /// @param prevOffset    offset of first central element
    /// @param midOffset     offset of first central element's predecessor
    /// @param nextOffset    offset of first central element's successor
    /// @param function      the function itself
    template <typename FUNC>
    __device__ void horizontalStep(const int count, const int prevOffset,
                                   const int midOffset, const int nextOffset,
                                   const FUNC & function) {
        // number of unchecked iterations
        const int STEPS = count / SIZE_X;

        // items remaining after last unchecked iteration
        const int finalCount = count % SIZE_X;

        // offset of items processed in last (checked) iteration
        const int finalOffset = count - finalCount;

        // all threads perform fixed number of iterations ...
        for(int i = 0; i < STEPS; i++) {
            const T previous = data[prevOffset + i * SIZE_X + threadIdx.x];
            const T next = data[nextOffset + i * SIZE_X + threadIdx.x];
            T & center = data[midOffset + i * SIZE_X + threadIdx.x];
            function(previous, center, next);
        }

        // ... but not all threads participate on final iteration
        if(threadIdx.x < finalCount) {
            const T previous = data[prevOffset + finalOffset + threadIdx.x];
            const T next = data[nextOffset + finalOffset + threadIdx.x];
            T & center = data[midOffset + finalOffset + threadIdx.x];
            function(previous, center, next);
        }
    }

public:

    /// Gets offset of the column with given index. Central columns have
    /// indices from 0 to NUM_LINES - 1, left boundary columns have negative
    /// indices and right boundary columns indices start with NUM_LINES.
    /// @param columnIndex  index of column to get pointer to
    /// @return  offset of the first item of column with specified index
    __device__ int getColumnOffset(int columnIndex) {
        columnIndex += BOUNDARY_X;             // skip boundary
        return columnIndex / 2                 // select right column
               + (columnIndex & 1) * ODD_OFFSET;  // select odd or even buffer
    }


    /// Provides access to data of the transform buffer.
    /// @param index  index of the item to work with
    /// @return reference to item at given index
    __device__ T & operator[] (const int index) {
        return data[index];
    }


    /// Applies specified function to all horizontally even elements in
    /// specified lines. (Including even elements in boundaries except
    /// first even element in first left boundary.) SIZE_X threads participate
    /// and synchronization is needed before result can be used.
    /// @param firstLine  index of first line
    /// @param numLines   count of lines
    /// @param func       function to be applied on all even elements
    ///                   parameters: previous (odd) element, the even
    ///                   element itself and finally next (odd) element
    template <typename FUNC>
    __device__ void forEachHorizontalEven(const int firstLine,
                                          const int numLines,
                                          const FUNC & func) {
        // number of even elemens to apply function to
        const int count = numLines * VERTICAL_STRIDE - 1;
        // offset of first even element
        const int centerOffset = firstLine * VERTICAL_STRIDE + 1;
        // offset of odd predecessor of first even element
        const int prevOffset = firstLine * VERTICAL_STRIDE + ODD_OFFSET;
        // offset of odd successor of first even element
        const int nextOffset = prevOffset + 1;

        // call generic horizontal step function
        horizontalStep(count, prevOffset, centerOffset, nextOffset, func);
    }


    /// Applies given function to all horizontally odd elements in specified
    /// lines. (Including odd elements in boundaries except last odd element
    /// in last right boundary.) SIZE_X threads participate and synchronization
    /// is needed before result can be used.
    /// @param firstLine  index of first line
    /// @param numLines   count of lines
    /// @param func       function to be applied on all odd elements
    ///                   parameters: previous (even) element, the odd
    ///                   element itself and finally next (even) element
    template <typename FUNC>
    __device__ void forEachHorizontalOdd(const int firstLine,
                                         const int numLines,
                                         const FUNC & func) {
        // numbet of odd elements to apply function to
        const int count = numLines * VERTICAL_STRIDE - 1;
        // offset of even predecessor of first odd element
        const int prevOffset = firstLine * VERTICAL_STRIDE;
        // offset of first odd element
        const int centerOffset = prevOffset + ODD_OFFSET;
        // offset of even successor of first odd element
        const int nextOffset = prevOffset + 1;

        // call generic horizontal step function
        horizontalStep(count, prevOffset, centerOffset, nextOffset, func);
    }


    /// Applies specified function to all even elements (except element #0)
    /// of given column. Each thread takes care of one column, so there's
    /// no need for synchronization.
    /// @param columnOffset  offset of thread's column
    /// @param f             function to be applied on all even elements
    ///                      parameters: previous (odd) element, the even
    ///                      element itself and finally next (odd) element
    template <typename F>
    __device__ void forEachVerticalEven(const int columnOffset, const F & f) {
        if(SIZE_Y > 3) { // makes no sense otherwise
            const int steps = SIZE_Y / 2 - 1;
            for(int i = 0; i < steps; i++) {
                const int row = 2 + i * 2;
                const T prev = data[columnOffset + (row - 1) * VERTICAL_STRIDE];
                const T next = data[columnOffset + (row + 1) * VERTICAL_STRIDE];
                f(prev, data[columnOffset + row * VERTICAL_STRIDE] , next);
            }
        }
    }


    /// Applies specified function to all odd elements of given column.
    /// Each thread takes care of one column, so there's no need for
    /// synchronization.
    /// @param columnOffset  offset of thread's column
    /// @param f             function to be applied on all odd elements
    ///                      parameters: previous (even) element, the odd
    ///                      element itself and finally next (even) element
    template <typename F>
    __device__ void forEachVerticalOdd(const int columnOffset, const F & f) {
        const int steps = (SIZE_Y - 1) / 2;
        for(int i = 0; i < steps; i++) {
            const int row = i * 2 + 1;
            const T prev = data[columnOffset + (row - 1) * VERTICAL_STRIDE];
            const T next = data[columnOffset + (row + 1) * VERTICAL_STRIDE];
            f(prev, data[columnOffset + row * VERTICAL_STRIDE], next);
        }
    }



    /// Scales elements at specified lines.
    /// @param evenScale  scaling factor for horizontally even elements
    /// @param oddScale   scaling factor for horizontally odd elements
    /// @param numLines   number of lines, whose elements should be scaled
    /// @param firstLine  index of first line to scale elements in
    __device__ void scaleHorizontal(const T evenScale, const T oddScale,
                                    const int firstLine, const int numLines) {
        const int offset = firstLine * VERTICAL_STRIDE;
        const int count = numLines * VERTICAL_STRIDE;
        const int steps = count / SIZE_X;
        const int finalCount = count % SIZE_X;
        const int finalOffset = count - finalCount;

        // run iterations, where all threads participate
        for(int i = 0; i < steps; i++) {
            data[threadIdx.x + i * SIZE_X + offset] *= evenScale;
            data[threadIdx.x + i * SIZE_X + offset + ODD_OFFSET] *= oddScale;
        }

        // some threads also finish remaining unscaled items
        if(threadIdx.x < finalCount) {
            data[threadIdx.x + finalOffset + offset] *= evenScale;
            data[threadIdx.x + finalOffset + offset + ODD_OFFSET] *= oddScale;
        }
    }


    /// Scales elements in specified column.
    /// @param evenScale     scaling factor for vertically even elements
    /// @param oddScale      scaling factor for vertically odd elements
    /// @param columnOffset  offset of the column to work with
    /// @param numLines      number of lines, whose elements should be scaled
    /// @param firstLine     index of first line to scale elements in
    __device__ void scaleVertical(const T evenScale, const T oddScale,
                                  const int columnOffset, const int numLines,
                                  const int firstLine) {
        for(int i = firstLine; i < (numLines + firstLine); i++) {
            if(i & 1) {
                data[columnOffset + i * VERTICAL_STRIDE] *= oddScale;
            } else {
                data[columnOffset + i * VERTICAL_STRIDE] *= evenScale;
            }
        }
    }

};  // end of class TransformBuffer






///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////                                                                       ////
////                       DWT Input/output indexing                       ////
////                                                                       ////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////



/// Base class for pixel loader and writer - manages computing start index,
/// stride and end of image for loading column of pixels.
/// @tparam T        type of image pixels
/// @tparam CHECKED  true = be prepared to image boundary, false = don't care
template <typename T, bool CHECKED>
class VerticalDWTPixelIO {
protected:
    int end;         ///< index of bottom neightbor of last pixel of column
    int stride;      ///< increment of pointer to get to next pixel

    /// Initializes pixel IO - sets end index and a position of first pixel.
    /// @param sizeX   width of the image
    /// @param sizeY   height of the image
    /// @param firstX  x-coordinate of first pixel to use
    /// @param firstY  y-coordinate of first pixel to use
    /// @return index of pixel at position [x, y] in the image
    __device__ int initialize(const int sizeX, const int sizeY,
                              int firstX, int firstY) {
        // initialize all pointers and stride
        end = CHECKED ? (sizeY * sizeX + firstX) : 0;
        stride = (sizeY > 1) ? sizeX : 0;
        return firstX + sizeX * firstY;
    }
};



/// Writes reverse transformed pixels directly into output image.
/// @tparam T        type of output pixels
/// @tparam CHECKED  true = be prepared to image boundary, false = don't care
template <typename T, bool CHECKED>
class VerticalDWTPixelWriter : VerticalDWTPixelIO<T, CHECKED> {
private:
    int next;   // index of the next pixel to be loaded

public:
    /// Initializes writer - sets output buffer and a position of first pixel.
    /// @param sizeX   width of the image
    /// @param sizeY   height of the image
    /// @param firstX  x-coordinate of first pixel to write into
    /// @param firstY  y-coordinate of first pixel to write into
    __device__ void init(const int sizeX, const int sizeY,
                         int firstX, int firstY) {
        if(firstX < sizeX) {
            next = this->initialize(sizeX, sizeY, firstX, firstY);
        } else {
            this->end = 0;
            this->stride = 0;
            next = 0;
        }
    }

    /// Writes given value at next position and advances internal pointer while
    /// correctly handling mirroring.
    /// @param output  output image to write pixel into
    /// @param value   value of the pixel to be written
    __device__ void writeInto(T * const output, const T & value) {
        if((!CHECKED) || (next != this->end)) {
            output[next] = value;
            next += this->stride;
        }
    }
};



/// Directly loads coefficients from four consecutively stored transformed
/// bands.
/// @tparam T        type of input band coefficients
/// @tparam CHECKED  true = be prepared to image boundary, false = don't care
template <typename T, bool CHECKED>
class VerticalDWTBandLoader {
private:
    int nextIdx;               ///< index of next sample to be loaded
    int sampleCountY;          ///< height of the output band - 1
    int remainingSampleCount;  ///< samples remaining before next mirroring
    
    /// increment of index to get from highpass band to the lowpass one
    int strideHighToLow;

    /// increment of index to get from the lowpass band to the highpass one
    int strideLowToHigh;
    
    /// Handles mirroring of image at edges in a DWT correct way.
    /// @param d      a position in the image (will be replaced by mirrored d)
    /// @param sizeD  size of the image along the dimension of 'd'
    __device__ static void mirror(int & d, const int & sizeD) {
        if(CHECKED) {
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
        } else {
            if(d >= sizeD) {
                d = 2 * sizeD - 2 - d;
            } else if(d < 0) {
                d = -d;
            }
        }
    }
    
    /// Checks internal index and possibly reverses direction of loader.
    /// (Handles mirroring at the bottom of the image.)
    /// @param input   input image to load next coefficient from
    /// @param stride  stride to use now (one of two loader's strides)
    /// @return loaded coefficient
    __device__ T updateAndLoad(const T * const input, const int stride) {
        // get result and update the index
        const T sampleValue = input[nextIdx];
        nextIdx += stride;
        
        // possibly mirror
        if(CHECKED && --remainingSampleCount == 0) {
            // repair next sample index
            nextIdx -= stride;
            
            // reset remaining sample count before next mirroring
            remainingSampleCount = sampleCountY;
            
            // swap and negate strides
            const int tempStride = strideLowToHigh;
            strideLowToHigh = -strideHighToLow;
            strideHighToLow = -tempStride;
        }
        
        // return the sample
        return sampleValue;
    }
public:

    /// Initializes loader - sets input size and a position of first pixel.
    /// @param imageSizeX   width of the image
    /// @param imageSizeY   height of the image
    /// @param firstX       x-coordinate of first pixel to load
    ///                     (Parity determines vertically low or high band.)
    /// @param firstY       y-coordinate of first pixel to load
    ///                     (Parity determines horizontally low or high band.)
    __device__ void init(const int imageSizeX, const int imageSizeY,
                         int firstX, const int firstY) {
        // mirror the x-coordinate to get the correct column
        mirror(firstX, imageSizeX);
        
        // remember number of samples before next mirroring and number 
        // of samples before two mirrorings
        sampleCountY = imageSizeY - 1;
        remainingSampleCount = imageSizeY - firstY;
        
        // initialize both strides
        strideHighToLow = (imageSizeX >> 1) + (firstX & imageSizeX & 1);
        strideLowToHigh = 0;
        
        // index of next sample to be loaded
        nextIdx = strideHighToLow * (firstY >> 1) + (firstX >> 1);
    }

    /// Sets all fields to zeros, for compiler not to complain about
    /// uninitialized stuff.
    __device__ void clear() {
        nextIdx = 0;
        sampleCountY = 0;
        remainingSampleCount = 0;
        strideHighToLow = 0;
        strideLowToHigh = 0;
    }

    /// Gets another coefficient from lowpass band and advances internal index.
    /// Call this method first if position of first pixel passed to init
    /// was in high band.
    /// @param input   input image to load next coefficient from
    /// @return next coefficient from the lowpass band of the given image
    __device__ T loadLowFrom(const T * const input) {
        return updateAndLoad(input, strideLowToHigh);
    }

    /// Gets another coefficient from the highpass band and advances index.
    /// Call this method first if position of first pixel passed to init
    /// was in high band.
    /// @param input   input image to load next coefficient from
    /// @return next coefficient from the highbass band of the given image
    __device__ T loadHighFrom(const T * const input) {
        return updateAndLoad(input, strideHighToLow);
    }

};





    
    
    
    
    


/// Wraps shared memory buffer and methods for computing 9/7 RDWT using
/// lifting schema and sliding window.
/// @tparam WIN_SIZE_X  width of the sliding window
/// @tparam WIN_SIZE_Y  height of the sliding window
template <int WIN_SIZE_X, int WIN_SIZE_Y>
class RDWT97 {
private:

    /// Info related to loading of one input column.
    /// @tparam CHECKED true if boundary chould be checked,
    ///                 false if there is no near boudnary
    template <bool CHECKED>
    struct RDWT97Column  {
        /// laoder of input pxels for given column.
        VerticalDWTBandLoader<float, CHECKED> loader;

        /// Offset of loaded column in shared memory buffer.
        int offset;

        /// Sets all fields to some values to avoid 'uninitialized' warnings.
        __device__ void clear() {
            loader.clear();
            offset = 0;
        }
    };


    /// Shared memory buffer used for 9/7 DWT transforms.
    typedef TransformBuffer<float, WIN_SIZE_X, WIN_SIZE_Y + 7, 4> RDWT97Buffer;

    /// Shared buffer used for reverse 9/7 DWT.
    RDWT97Buffer buffer;

    /// Difference between indices of two vertical neighbors in buffer.
    enum { STRIDE = RDWT97Buffer::VERTICAL_STRIDE };


    /// Horizontal 9/7 RDWT on specified lines of transform buffer.
    /// @param lines      number of lines to be transformed
    /// @param firstLine  index of the first line to be transformed
    __device__ void horizontalRDWT97(int lines, int firstLine) {
        __syncthreads();
        buffer.scaleHorizontal(scale97Mul, scale97Div, firstLine, lines);
        __syncthreads();
        buffer.forEachHorizontalEven(firstLine, lines, AddScaledSum(r97update2));
        __syncthreads();
        buffer.forEachHorizontalOdd(firstLine, lines, AddScaledSum(r97predict2));
        __syncthreads();
        buffer.forEachHorizontalEven(firstLine, lines, AddScaledSum(r97update1));
        __syncthreads();
        buffer.forEachHorizontalOdd(firstLine, lines, AddScaledSum(r97Predict1));
        __syncthreads();
    }


    /// Initializes one column of shared transform buffer with 7 input pixels.
    /// Those 7 pixels will not be transformed. Also initializes given loader.
    /// @tparam CHECKED  true if there are near image boundaries
    /// @param colIndex  index of column in shared transform buffer
    /// @param inputL    input buffer with low-pass band
    /// @param inputH    input buffer with high-pass band
    /// @param sizeX     width of the input image
    /// @param sizeY     height of the input image
    /// @param column    (uninitialized) info about loading one column
    /// @param firstY    index of first image row to be transformed
    template <bool CHECKED>
    __device__ void initColumn(const int colIndex,
                               const float * const inputL,
                               const float * const inputH,
                               const int sizeX, const int sizeY,
                               RDWT97Column<CHECKED> & column,
                               const int firstY) {
        // coordinates of the first coefficient to be loaded
        const int firstX = blockIdx.x * WIN_SIZE_X + colIndex;

        // offset of the column with index 'colIndex' in the transform buffer
        column.offset = buffer.getColumnOffset(colIndex);

        if(blockIdx.y == 0) {
            // topmost block - apply mirroring rules when loading first 7 rows
            column.loader.init(sizeX, sizeY, firstX, firstY);

            // load pixels in mirrored way
            buffer[column.offset + 3 * STRIDE] = column.loader.loadLowFrom(inputL);
            buffer[column.offset + 4 * STRIDE] =
                buffer[column.offset + 2 * STRIDE] = column.loader.loadHighFrom(inputH);
            buffer[column.offset + 5 * STRIDE] =
                buffer[column.offset + 1 * STRIDE] = column.loader.loadLowFrom(inputL);
            buffer[column.offset + 6 * STRIDE] =
                buffer[column.offset + 0 * STRIDE] = column.loader.loadHighFrom(inputH);
        } else {
            // non-topmost row - regular loading:
            column.loader.init(sizeX, sizeY, firstX, firstY - 3);
            buffer[column.offset + 0 * STRIDE] = column.loader.loadHighFrom(inputH);
            buffer[column.offset + 1 * STRIDE] = column.loader.loadLowFrom(inputL);
            buffer[column.offset + 2 * STRIDE] = column.loader.loadHighFrom(inputH);
            buffer[column.offset + 3 * STRIDE] = column.loader.loadLowFrom(inputL);
            buffer[column.offset + 4 * STRIDE] = column.loader.loadHighFrom(inputH);
            buffer[column.offset + 5 * STRIDE] = column.loader.loadLowFrom(inputL);
            buffer[column.offset + 6 * STRIDE] = column.loader.loadHighFrom(inputH);
        }
        // Now, the next coefficient, which will be loaded by loader, is #4.
    }


    /// Using given loader, it loads another WIN_SIZE_Y coefficients
    /// into specified column.
    /// @tparam CHECKED  true if there are near image boundaries
    /// @param col       info about loaded column
    /// @param inputL    buffer with input coefficients for low-pass band
    /// @param inputH    buffer with input coefficients for high-pass band
    template <bool CHECKED>
    inline __device__ void loadWindowIntoColumn(RDWT97Column<CHECKED> & col,
            const float * const inputL,
            const float * const inputH) {
        for(int i = 7; i < (7 + WIN_SIZE_Y); i += 2) {
            buffer[col.offset + i * STRIDE] = col.loader.loadLowFrom(inputL);
            buffer[col.offset + (i + 1) * STRIDE] = col.loader.loadHighFrom(inputH);
        }
    }


    /// Actual GPU 9/7 RDWT sliding window lifting schema implementation.
    /// @tparam CHECKED_LOADS   true if loader should check boundaries
    /// @tparam CHECKED_WRITES  true if boundaries should be taken into account
    ///                         when writing into output buffer
    /// @param inLL      input LL 9/7 coefficients
    /// @param inHL      input HL band
    /// @param inLH      input LH band
    /// @param inHH      input HH band
    /// @param out       output buffer (for reverse transformed image)
    /// @param sizeX     width of the output image
    /// @param sizeY     height of the output image
    /// @param winSteps  number of steps of sliding window
    template <bool CHECKED_LOADS, bool CHECKED_WRITES>
    __device__ void transform(const float * const inLL,
                              const float * const inHL,
                              const float * const inLH,
                              const float * const inHH,
                              float * const out,
                              const int sizeX, const int sizeY,
                              const int winSteps) {
        // info about one main column and one boundary column
        RDWT97Column<CHECKED_LOADS> column;
        RDWT97Column<CHECKED_LOADS> boundaryColumn;

        // index of first image row to be transformed
        const int firstY = blockIdx.y * WIN_SIZE_Y * winSteps;

        // input from vertically low- and high-pass band
        const int columnIdx = parityIdx<WIN_SIZE_X>();
        const float * const inL = columnIdx & 1 ? inHL : inLL;
        const float * const inH = columnIdx & 1 ? inHH : inLH;
//         const float * const boundaryInL = threadIdx.x & 1 ? inHL : inLL;

        // initialize boundary columns
        boundaryColumn.clear();
        if(columnIdx < 8) {
            // each thread among first 7 ones gets index of one of boundary columns
            const int colId = columnIdx + ((columnIdx < 4) ? WIN_SIZE_X : -8);

            // Thread initializes offset of the boundary column (in shared
            // buffer), first 7 pixels of the column and a loader for this column.
            initColumn(colId, inL, inH, sizeX, sizeY, boundaryColumn, firstY);
        }

        // All threads initialize central columns.
        initColumn(columnIdx, inL, inH, sizeX, sizeY, column, firstY);

        // horizontally transform first 7 rows
        horizontalRDWT97(7, 0);

        // writer of output pixels - initialize it
        const int outputX = blockIdx.x * WIN_SIZE_X + threadIdx.x;
        VerticalDWTPixelWriter<float, CHECKED_WRITES> writer;
        writer.init(sizeX, sizeY, outputX, firstY);

        // offset of column (in transform buffer) saved by this thread
        const int outColumnOffset = buffer.getColumnOffset(threadIdx.x);

        // (Each iteration assumes that first 7 rows of transform buffer are
        // already loaded with horizontally transformed pixels.)
        for(int w = 0; w < winSteps; w++) {
            // Load another WIN_SIZE_Y lines of this thread's column
            // into the transform buffer.
            loadWindowIntoColumn(column, inL, inH);

            // possibly load boundary columns
            if(columnIdx < 8) {
                loadWindowIntoColumn(boundaryColumn, inL, inH);
            }

            // horizontally transform all newly loaded lines
            horizontalRDWT97(WIN_SIZE_Y, 7);

            // Using 7 registers, remember current values of last 7 rows
            // of transform buffer. These rows are transformed horizontally
            // only and will be used in next iteration.
            float last7Lines[7];
            for(int i = 0; i < 7; i++) {
                last7Lines[i] = buffer[outColumnOffset + (WIN_SIZE_Y + i) * STRIDE];
            }

            // vertically transform all central columns
            buffer.scaleVertical(scale97Div, scale97Mul, outColumnOffset,
                                 WIN_SIZE_Y + 7, 0);
            buffer.forEachVerticalOdd(outColumnOffset, AddScaledSum(r97update2));
            buffer.forEachVerticalEven(outColumnOffset, AddScaledSum(r97predict2));
            buffer.forEachVerticalOdd(outColumnOffset, AddScaledSum(r97update1));
            buffer.forEachVerticalEven(outColumnOffset, AddScaledSum(r97Predict1));

            // Save all results of current window. Results are in transform buffer
            // at rows from #3 to #(3 + WIN_SIZE_Y). Other rows are invalid now.
            // (They only served as a boundary for vertical RDWT.)
            for(int i = 3; i < (3 + WIN_SIZE_Y); i++) {
                writer.writeInto(out, buffer[outColumnOffset + i * STRIDE]);
            }
            
            // Use last 7 remembered lines as first 7 lines for next iteration.
            // As expected, these lines are already horizontally transformed.
            for(int i = 0; i < 7; i++) {
                buffer[outColumnOffset + i * STRIDE] = last7Lines[i];
            }

            // Wait for all writing threads before proceeding to loading new
            // coeficients in next iteration. (Not to overwrite those which
            // are not written yet.)
            __syncthreads();
        }
    }


public:
    /// Main GPU 9/7 RDWT entry point.
    /// @param inLL     input LL 9/7 transformed coefficients
    /// @param inHL     input HL band
    /// @param inLH     input LH band
    /// @param inHH     input HH band
    /// @param out      output buffer (for reverse transformed image)
    /// @param sizeX    width of the output image
    /// @param sizeY    height of the output image
    __device__ static void run(const float * const inLL,
                               const float * const inHL,
                               const float * const inLH,
                               const float * const inHH,
                               float * const output,
                               const int sx, const int sy, const int steps) {
        // prepare instance with buffer in shared memory
        __shared__ RDWT97<WIN_SIZE_X, WIN_SIZE_Y> rdwt97;

        // Compute limits of this threadblock's block of pixels and use them to
        // determine, whether this threadblock will have to deal with boundary.
        // (3 in next expressions is for radius of impulse response of 9/7 RDWT.)
        const int maxX = (blockIdx.x + 1) * WIN_SIZE_X + 3;
        const int maxY = (blockIdx.y + 1) * WIN_SIZE_Y * steps + 3;
        const bool atRightBoudary = maxX >= sx;
        const bool atBottomBoudary = maxY >= sy;
        const bool smallSizeX = (WIN_SIZE_X + 4) / 2 >= sx;
        const bool smallSizeY = (WIN_SIZE_Y + 4) / 2 >= sy;

        // Select specialized version of code according to distance of this
        // threadblock's pixels from image boundary.
        if(atBottomBoudary || smallSizeY || smallSizeX) {
            // near bottom edge or small input => check both writing and reading
            rdwt97.transform<true, true>(inLL, inHL, inLH, inHH, output, sx, sy, steps);
        } else if(atRightBoudary) {
            // near right boundary only => check writing only
            rdwt97.transform<false, true>(inLL, inHL, inLH, inHH, output, sx, sy, steps);
        } else {
            // no nearby boundary => check nothing
            rdwt97.transform<false, false>(inLL, inHL, inLH, inHH, output, sx, sy, steps);
        }
    }

}; // end of class RDWT97



/// Main GPU 9/7 RDWT entry point.
/// @param inLL     input LL 9/7 transformed coefficients
/// @param inHL     input HL band
/// @param inLH     input LH band
/// @param inHH     input HH band
/// @param out      output buffer (for reverse transformed image)
/// @param sizeX    width of the output image
/// @param sizeY    height of the output image
template <int WIN_SX, int WIN_SY>
__global__ void rdwt97Kernel(const float * const inLL,
                             const float * const inHL,
                             const float * const inLH,
                             const float * const inHH,
                             float * const out,
                             const int sx, const int sy, const int steps) {
    RDWT97<WIN_SX, WIN_SY>::run(inLL, inHL, inLH, inHH, out, sx, sy, steps);
}



/// Only computes optimal number of sliding window steps,
/// number of threadblocks and then lanches the 9/7 RDWT kernel.
/// @tparam WIN_SX  width of sliding window
/// @tparam WIN_SY  height of sliding window
/// @param inLL     input 9/7 coefficionts
/// @param inOther  other input bands
/// @param out      output buffer
/// @param sx       width of the input image
/// @param sy       height of the input image
template <int WIN_SX, int WIN_SY>
void launchRDWT97Kernel (const float * inLL, const float * inOther,
                         float * out, int sx, int sy) {
    // compute optimal number of steps of each sliding window
    const int steps = divRndUp(sy, 15 * WIN_SY);

    // prepare grid size
    dim3 gSize(divRndUp(sx, WIN_SX), divRndUp(sy, WIN_SY * steps));

    // config kernel
    cudaFuncSetCacheConfig(rdwt97Kernel<WIN_SX, WIN_SY>, cudaFuncCachePreferShared);

    // finally launch kernel
    rdwt97Kernel<WIN_SX, WIN_SY><<<gSize, WIN_SX>>>(inLL, inOther, out, sx, sy, steps);
}



// /// Reverse 9/7 2D DWT. See common rules (dwt.h) for more details.
// /// @param in      Input DWT coefficients. Format described in common rules.
// ///                Will not be preserved (will be overwritten).
// /// @param out     output buffer on GPU - will contain original image
// ///                in normalized range [-0.5, 0.5].
// /// @param temp    temp buffer
// /// @param sizeX   width of input image (in pixels)
// /// @param sizeY   height of input image (in pixels)
// /// @param levels  number of recursive DWT levels
// void dwtReverseFloat(float *in, float *out, float * temp,
//                      int sizeX, int sizeY, int levels) {
//     // compose LL bands in all levels (level 0 = output)
//     for(int level = levels; level--;) {
//         // size of output resulting from this level
//         const int outSizeX = llSize(sizeX, level);
//         const int outSizeY = llSize(sizeY, level);
// 
//         // input and output buffers for temporary LL band composition in each level
//         const float * const llIn = (level == levels - 1) ? in : (level & 1 ? out : temp);
//         float * const levelOut = level & 1 ? temp : out;
// 
//         // select right width of kernel for the size of the image
//         if(outSizeX >= 960) {
//             launchRDWT97Kernel<160, 8>(llIn, in, levelOut, outSizeX, outSizeY);
//         } else if (outSizeX >= 480) {
//             launchRDWT97Kernel<128, 6>(llIn, in, levelOut, outSizeX, outSizeY);
//         } else {
//             launchRDWT97Kernel<64, 6>(llIn, in, levelOut, outSizeX, outSizeY);
//         }
//     }
// }




/// Wraps shared momory buffer and algorithms needed for computing 5/3 RDWT
/// using sliding window and lifting schema.
/// @tparam WIN_SIZE_X  width of sliding window
/// @tparam WIN_SIZE_Y  height of sliding window
template <int WIN_SIZE_X, int WIN_SIZE_Y>
class RDWT53 {
private:

    /// Shared memory buffer used for 5/3 DWT transforms.
    typedef TransformBuffer<int, WIN_SIZE_X, WIN_SIZE_Y + 3, 2> RDWT53Buffer;

    /// Shared buffer used for reverse 5/3 DWT.
    RDWT53Buffer buffer;

    /// Difference between indices of two vertically neighboring items in buffer.
    enum { STRIDE = RDWT53Buffer::VERTICAL_STRIDE };


    /// Info needed for loading of one input column from input image.
    /// @tparam CHECKED  true if loader should check boundaries
    template <bool CHECKED>
    struct RDWT53Column {
        /// loader of pixels from column in input image
        VerticalDWTBandLoader<int, CHECKED> loader;

        /// Offset of corresponding column in shared buffer.
        int offset;

        /// Sets all fields to some values to avoid 'uninitialized' warnings.
        __device__ void clear() {
            offset = 0;
            loader.clear();
        }
    };


    /// 5/3 DWT reverse update operation.
    struct Reverse53Update {
        __device__ void operator() (const int p, int & c, const int n) const {
            c -= (p + n + 2) >> 2;  // F.3, page 118, ITU-T Rec. T.800 final draft
        }
    };


    /// 5/3 DWT reverse predict operation.
    struct Reverse53Predict {
        __device__ void operator() (const int p, int & c, const int n) const {
            c += (p + n) >> 1;      // F.4, page 118, ITU-T Rec. T.800 final draft
        }
    };


    /// Horizontal 5/3 RDWT on specified lines of transform buffer.
    /// @param lines      number of lines to be transformed
    /// @param firstLine  index of the first line to be transformed
    __device__ void horizontalTransform(const int lines, const int firstLine) {
        __syncthreads();
        buffer.forEachHorizontalEven(firstLine, lines, Reverse53Update());
        __syncthreads();
        buffer.forEachHorizontalOdd(firstLine, lines, Reverse53Predict());
        __syncthreads();
    }


    /// Using given loader, it loads another WIN_SIZE_Y coefficients
    /// into specified column.
    /// @tparam CHECKED  true if loader should check image boundaries
    /// @param inputL    input coefficients to load from (low-pass)
    /// @param inputH    input coefficients to load from (high-pass)
    /// @param col       info about loaded column
    template <bool CHECKED>
    inline __device__ void loadWindowIntoColumn(const int * const inputL,
            const int * const inputH,
            RDWT53Column<CHECKED> & col) {
        for(int i = 3; i < (3 + WIN_SIZE_Y); i += 2) {
            buffer[col.offset + i * STRIDE] = col.loader.loadLowFrom(inputL);
            buffer[col.offset + (i + 1) * STRIDE] = col.loader.loadHighFrom(inputH);
        }
    }


    /// Initializes one column of shared transform buffer with 7 input pixels.
    /// Those 7 pixels will not be transformed. Also initializes given loader.
    /// @tparam CHECKED  true if loader should check image boundaries
    /// @param columnX   x coordinate of column in shared transform buffer
    /// @param inputL    input image buffer for vertically low-pass
    /// @param inputH    input image buffer for vertically high-pass
    /// @param sizeX     width of the input image
    /// @param sizeY     height of the input image
    /// @param loader    (uninitialized) info about loaded column
    template <bool CHECKED>
    __device__ void initColumn(const int columnX, const int * const inputL,
                               const int * const inputH, const int sizeX,
                               const int sizeY, RDWT53Column<CHECKED> & column,
                               const int firstY) {
        // coordinates of the first coefficient to be loaded
        const int firstX = blockIdx.x * WIN_SIZE_X + columnX;

        // offset of the column with index 'colIndex' in the transform buffer
        column.offset = buffer.getColumnOffset(columnX);

        if(blockIdx.y == 0) {
            // topmost block - apply mirroring rules when loading first 3 rows
            column.loader.init(sizeX, sizeY, firstX, firstY);

            // load pixels in mirrored way
            buffer[column.offset + 1 * STRIDE] = column.loader.loadLowFrom(inputL);
            buffer[column.offset + 0 * STRIDE] =
                buffer[column.offset + 2 * STRIDE] = column.loader.loadHighFrom(inputH);
        } else {
            // non-topmost row - regular loading:
            column.loader.init(sizeX, sizeY, firstX, firstY - 1);
            buffer[column.offset + 0 * STRIDE] = column.loader.loadHighFrom(inputH);
            buffer[column.offset + 1 * STRIDE] = column.loader.loadLowFrom(inputL);
            buffer[column.offset + 2 * STRIDE] = column.loader.loadHighFrom(inputH);
        }
        // Now, the next coefficient, which will be loaded by loader, is #2.
    }


    /// Actual GPU 5/3 RDWT implementation.
    /// @tparam CHECKED_LOADS   true if boundaries must be checked when reading
    /// @tparam CHECKED_WRITES  true if boundaries must be checked when writing
    /// @param inLL      input LL band (5/3 transformed coefficients)
    /// @param inHL      input HL band
    /// @param inLH      input LH band
    /// @param inHH      input HH band
    /// @param out       output buffer (for reverse transformed image)
    /// @param sizeX     width of the output image
    /// @param sizeY     height of the output image
    /// @param winSteps  number of sliding window steps
    template<bool CHECKED_LOADS, bool CHECKED_WRITES>
    __device__ void transform(const int * const inLL, const int * const inHL,
                              const int * const inLH, const int * const inHH,
                              int * const out, const int sizeX, const int sizeY,
                              const int winSteps) {
        // info about one main and one boundary column
        RDWT53Column<CHECKED_LOADS> column, boundaryColumn;

        // index of first row to be transformed
        const int firstY = blockIdx.y * WIN_SIZE_Y * winSteps;

        // input from vertically low- and high-pass band
        const int columnIdx = parityIdx<WIN_SIZE_X>();
        const int * const inL = columnIdx & 1 ? inHL : inLL;
        const int * const inH = columnIdx & 1 ? inHH : inLH;

        // some threads initialize boundary columns
        boundaryColumn.clear();
        if(columnIdx < 4) {
            // First 3 threads also handle boundary columns. Thread #0 gets right
            // column #0, thread #1 get right column #1 and thread #2 left column.
            const int colId = columnIdx + ((columnIdx < 2) ? WIN_SIZE_X : -4);

            // Thread initializes offset of the boundary column (in shared
            // buffer), first 3 pixels of the column and a loader for this column.
            initColumn(colId, inL, inH, sizeX, sizeY, boundaryColumn, firstY);
        }

        // All threads initialize central columns.
        initColumn(columnIdx, inL, inH, sizeX, sizeY, column, firstY);

        // horizontally transform first 3 rows
        horizontalTransform(3, 0);

        // writer of output pixels - initialize it
        const int outX = blockIdx.x * WIN_SIZE_X + threadIdx.x;
        VerticalDWTPixelWriter<int, CHECKED_WRITES> writer;
        writer.init(sizeX, sizeY, outX, firstY);

        // offset of column (in transform buffer) saved by this thread
        const int outputColumnOffset = buffer.getColumnOffset(threadIdx.x);

        // (Each iteration assumes that first 3 rows of transform buffer are
        // already loaded with horizontally transformed pixels.)
        for(int w = 0; w < winSteps; w++) {
            // Load another WIN_SIZE_Y lines of this thread's column
            // into the transform buffer.
            loadWindowIntoColumn(inL, inH, column);

            // possibly load boundary columns
            if(columnIdx < 4) {
                loadWindowIntoColumn(inL, inH, boundaryColumn);
            }

            // horizontally transform all newly loaded lines
            horizontalTransform(WIN_SIZE_Y, 3);

            // Using 3 registers, remember current values of last 3 rows
            // of transform buffer. These rows are transformed horizontally
            // only and will be used in next iteration.
            int last3Lines[3];
            last3Lines[0] = buffer[outputColumnOffset + (WIN_SIZE_Y + 0) * STRIDE];
            last3Lines[1] = buffer[outputColumnOffset + (WIN_SIZE_Y + 1) * STRIDE];
            last3Lines[2] = buffer[outputColumnOffset + (WIN_SIZE_Y + 2) * STRIDE];

            // vertically transform all central columns
            buffer.forEachVerticalOdd(outputColumnOffset, Reverse53Update());
            buffer.forEachVerticalEven(outputColumnOffset, Reverse53Predict());

            // Save all results of current window. Results are in transform buffer
            // at rows from #1 to #(1 + WIN_SIZE_Y). Other rows are invalid now.
            // (They only served as a boundary for vertical RDWT.)
            for(int i = 1; i < (1 + WIN_SIZE_Y); i++) {
                writer.writeInto(out, buffer[outputColumnOffset + i * STRIDE]);
            }

            // Use last 3 remembered lines as first 3 lines for next iteration.
            // As expected, these lines are already horizontally transformed.
            buffer[outputColumnOffset + 0 * STRIDE] = last3Lines[0];
            buffer[outputColumnOffset + 1 * STRIDE] = last3Lines[1];
            buffer[outputColumnOffset + 2 * STRIDE] = last3Lines[2];

            // Wait for all writing threads before proceeding to loading new
            // coeficients in next iteration. (Not to overwrite those which
            // are not written yet.)
            __syncthreads();
        }
    }


public:
    /// Main GPU 5/3 RDWT entry point.
    /// @param inLL      input LL band (5/3 transformed coefficients)
    /// @param inHL      input HL band
    /// @param inLH      input LH band
    /// @param inHH      input HH band
    /// @param output    output buffer (for reverse transformed image)
    /// @param sizeX     width of the output image
    /// @param sizeY     height of the output image
    /// @param winSteps  number of sliding window steps
    __device__ static void run(const int * const inLL,
                               const int * const inHL,
                               const int * const inLH,
                               const int * const inHH,
                               int * const output,
                               const int sx, const int sy, const int steps) {
        // prepare instance with buffer in shared memory
        __shared__ RDWT53<WIN_SIZE_X, WIN_SIZE_Y> rdwt53;

        // Compute limits of this threadblock's block of pixels and use them to
        // determine, whether this threadblock will have to deal with boundary.
        // (1 in next expressions is for radius of impulse response of 5/3 RDWT.)
        const int maxX = (blockIdx.x + 1) * WIN_SIZE_X + 1;
        const int maxY = (blockIdx.y + 1) * WIN_SIZE_Y * steps + 1;
        const bool atRightBoudary = maxX >= sx;
        const bool atBottomBoudary = maxY >= sy;
        const bool smallSizeX = (WIN_SIZE_X + 4) / 2 >= sx;
        const bool smallSizeY = (WIN_SIZE_Y + 4) / 2 >= sy;

        // Select specialized version of code according to distance of this
        // threadblock's pixels from image boundary.
        if(atBottomBoudary || smallSizeY || smallSizeX) {
            // near bottom edge or small input => check both writing and reading
            rdwt53.transform<true, true>(inLL, inHL, inLH, inHH, output, sx, sy, steps);
        } else if(atRightBoudary) {
            // near right boundary only => check writing only
            rdwt53.transform<false, true>(inLL, inHL, inLH, inHH, output, sx, sy, steps);
        } else {
            // no nearby boundary => check nothing
            rdwt53.transform<false, false>(inLL, inHL, inLH, inHH, output, sx, sy, steps);
        }
    }

}; // end of class RDWT53



/// Main GPU 5/3 RDWT entry point.
/// @param inLL      input LL band (5/3 transformed coefficients)
/// @param inHL      input HL band
/// @param inLH      input LH band
/// @param inHH      input HH band
/// @param out       output buffer (for reverse transformed image)
/// @param sizeX     width of the output image
/// @param sizeY     height of the output image
/// @param winSteps  number of sliding window steps
template <int WIN_SX, int WIN_SY>
__global__ void rdwt53Kernel(const int * const inLL,
                             const int * const inHL,
                             const int * const inLH,
                             const int * const inHH,
                             int * const out,
                             const int sx, const int sy, const int steps) {
    RDWT53<WIN_SX, WIN_SY>::run(inLL, inHL, inLH, inHH, out, sx, sy, steps);
}



/// Only computes optimal number of sliding window steps,
/// number of threadblocks and then lanches the 5/3 RDWT kernel.
/// @tparam WIN_SX  width of sliding window
/// @tparam WIN_SY  height of sliding window
/// @param 
/// TODO: decribe parameters!!!

template <int WIN_SX, int WIN_SY>
void launchRDWT53Kernel (const int * const llIn,
                         const int * const hlIn,
                         const int * const lhIn,
                         const int * const hhIn,
                         int * const out,
                         const XY & size,
                         const cudaStream_t & stream) {
    // compute optimal number of steps of each sliding window
    const int steps = divRndUp(size.y, 15 * WIN_SY);

    // prepare grid size
    dim3 gSize(divRndUp(size.x, WIN_SX), divRndUp(size.y, WIN_SY * steps));

    // finally transform this level
    rdwt53Kernel<WIN_SX, WIN_SY><<<gSize, WIN_SX, 0, stream>>>
                (llIn, hlIn, lhIn, hhIn, out, size.x, size.y, steps);
}





/// Only computes optimal number of sliding window steps,
/// number of threadblocks and then lanches the 9/7 RDWT kernel.
/// @tparam WIN_SX  width of sliding window
/// @tparam WIN_SY  height of sliding window
/// @param 
/// TODO: describe parameters
template <int WIN_SX, int WIN_SY>
void launchRDWT97Kernel (const int * const llIn,
                         const int * const hlIn,
                         const int * const lhIn,
                         const int * const hhIn,
                         int * const out,
                         const XY & size,
                         const cudaStream_t & stream) {
    // compute optimal number of steps of each sliding window
    const int steps = divRndUp(size.y, 15 * WIN_SY);

    // prepare grid size
    dim3 gSize(divRndUp(size.x, WIN_SX), divRndUp(size.y, WIN_SY * steps));

    // finally launch kernel
    rdwt97Kernel<WIN_SX, WIN_SY><<<gSize, WIN_SX, 0, stream>>>
                ((const float*)llIn, (const float*)hlIn, (const float*)lhIn,
                 (const float*)hhIn, (float*)out, size.x, size.y, steps);
}




// /// Reverse 5/3 2D DWT. See common rules (above) for more details.
// /// @param in      Input DWT coefficients. Format described in common rules.
// ///                Will not be preserved (will be overwritten).
// /// @param out     output buffer on GPU - will contain original image
// ///                in normalized range [-128, 127].
// /// @param temp    temp buffer
// /// @param sizeX   width of input image (in pixels)
// /// @param sizeY   height of input image (in pixels)
// /// @param levels  number of recursive DWT levels
// void dwtReverseInt(int * in, int * out, int * temp,
//                    int sizeX, int sizeY, int levels) {
//     // compose LL bands in all levels (level 0 = output)
//     for(int level = levels; level--;) {
//         // size of output resulting from this level
//         const int outSizeX = llSize(sizeX, level);
//         const int outSizeY = llSize(sizeY, level);
// 
//         // input and output buffers for temporary LL band composition in each level
//         const int * const llIn = (level == levels - 1) ? in : (level & 1 ? out : temp);
//         int * const levelOut = level & 1 ? temp : out;
// 
//         // select right width of kernel for the size of the image
//         if(outSizeX >= 960) {
//             launchRDWT53Kernel<192, 8>(llIn, in, levelOut, outSizeX, outSizeY);
//         } else if (outSizeX >= 480) {
//             launchRDWT53Kernel<128, 8>(llIn, in, levelOut, outSizeX, outSizeY);
//         } else {
//             launchRDWT53Kernel<64, 8>(llIn, in, levelOut, outSizeX, outSizeY);
//         }
//     }
// }



/// High level DWT implementation for JPEG 2000 decoder.
void DWT::launch(int * const out,
                   const int * const llIn,
                   const int * const hlIn,
                   const int * const lhIn,
                   const int * const hhIn,
                   const bool reversible,
                   const XY & begin,
                   const XY & end,
                   cudaStream_t & str) {
    // check output coordinates
    if((begin.x & 1) || (begin.y & 1)) {
        throw Error(J2KD_ERROR_UNSUPPORTED, "DWT on odd coordinates not supported.");
    }
    
    // size of output
    const XY size = end - begin;
    
//     printf("DWT (%d x %d, %s).\n", size.x, size.y, reversible ? "int" : "float");
    
    // select kernel size and data type
    if(reversible) {
        if(size.x >= 960) {
            launchRDWT53Kernel<192, 8>(llIn, hlIn, lhIn, hhIn, out, size, str);
        } else if (size.x >= 480) {
            launchRDWT53Kernel<128, 8>(llIn, hlIn, lhIn, hhIn, out, size, str);
        } else {
            launchRDWT53Kernel<64, 8>(llIn, hlIn, lhIn, hhIn, out, size, str);
        }
    } else {
        if(size.x >= 960) {
            launchRDWT97Kernel<192, 8>(llIn, hlIn, lhIn, hhIn, out, size, str);
        } else if (size.x >= 480) {
            launchRDWT97Kernel<128, 8>(llIn, hlIn, lhIn, hhIn, out, size, str);
        } else {
            launchRDWT97Kernel<64, 8>(llIn, hlIn, lhIn, hhIn, out, size, str);
        }
    }
}


void DWT::setup() {
    cudaFuncSetCacheConfig(rdwt53Kernel<192, 8>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(rdwt53Kernel<128, 8>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(rdwt53Kernel<164, 8>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(rdwt97Kernel<192, 8>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(rdwt97Kernel<128, 8>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(rdwt97Kernel<164, 8>, cudaFuncCachePreferShared);
}



void DWT::copyGpu(void * const dest,
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
        // reference ot the tile, tile-component and its coding info
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
                const int * const hlIn = in + bands[0].outPixOffset;
                const int * const lhIn = in + bands[1].outPixOffset;
                const int * const hhIn = in + bands[2].outPixOffset;
                
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
                launch(llOut, llIn, hlIn, lhIn, hhIn, cod.reversible,
                       resOut.begin, resOut.end, stream);
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





