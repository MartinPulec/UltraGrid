///
/// @file    j2kd_output_comp.cu
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Configurable component output formatter.
///

#include "j2kd_out_fmt.h"

namespace cuj2kd {



//
//  Integer kernel types:
//     - input type: int / float
//     - output type: 8 / 16 / 32
//     - signed / unsigned
//     - with / without final SHIFT
//     - bit depth extension: yes / no
//     - bit depth crop: yes / no
//     - OR-combine / replace
//  Total: 2 x 3 x 2 x 2 x 2 x 2 x 2 = 192 instances
//  
//  
//  Float kernels types:   (NOTE: commented out)
//     - input type: int / float
//     - output type: float / double
//     - with / without final MAD
//  Total: 2 x 2 x 2 = 8 instances
//  


    
/// Parameters for integer formatting kernel.    
struct FormatIntKernelParams {
    /// top-left source sample pointer
    const void * src;
    
    /// difference between indices of two vertically neighboring source samples
    int srcStrideY;
    
    /// top-left output sample pointer
    void * out;
    
    /// output strides (even negative strides allowed)
    XY outStride;
    
    /// size of formatted area
    XY imgSize;
    
    /// final shift to left
    int shiftLeft;
    
    /// minimum sample value (black clamp)
    s32 rangeMin;
    
    /// maximum sample value (white clamp)
    s32 rangeMax;
    
    /// DC level shift add
    int dcLevelShift;
    
    /// shift to get rid of extra bits
    int bitDepthShift;
    
    /// source samples bit depth
    int srcDepth;
    
    /// output samples bit depth
    int outDepth;
}; // end of struct FormatIntKernelParams



/// Integer type formatting kernel for single compnent in single tile.
/// @tparam TYPE  output sample data type
/// @tparam SOURCE_IS_FLOAT  true if source samples are float
/// @tparam OUTPUT_IS_UNSIGNED  true for DC level shift
/// @tparam DO_FINAL_SHIFT  true if final shift-left should be done
/// @tparam OUTPUT_OR_COMBINE  true if output should be ORed with value in 
///                            output memory, false if memory should 
///                            be overwritten
/// @tparam EXTEND_BIT_DEPTH  true if output bit depth is greater than source
/// @tparam DISCARD_BIT_DEPTH  true if some depth bits should be discarded
/// @param params  all parameters packed in one structure
template <
        typename TYPE,  // limited to int types (s8, s16, s32)
        bool SOURCE_IS_FLOAT,  // true if source samples are floats, false if integers
        bool OUTPUT_IS_UNSIGNED,
        bool DO_FINAL_SHIFT,
        bool OUTPUT_OR_COMBINE,
        bool EXTEND_BIT_DEPTH,
        bool DISCARD_BIT_DEPTH
>
__global__ static void fmtIntKernel(const FormatIntKernelParams p) {
    // global coordinates of the thread in the grid
    // (also coordinates of therad's pixel in currently processed area)
    const int pixX = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixY = threadIdx.y + blockIdx.y * blockDim.y;
    
    // check coordinates and quit if out of range
    if(pixX >= p.imgSize.x || pixY >= p.imgSize.y) {
        return;
    }
    
    // load the sample
    const int sampleIdx = pixX + p.srcStrideY * pixY;
    s32 sample = ((const s32*)p.src)[sampleIdx];
    
    // round the sample and covert it to int if it is float
    if(SOURCE_IS_FLOAT) {
        sample = (int)rintf(__int_as_float(sample));
    }
    
    // possibly duplicate the bit depth few times
    if(EXTEND_BIT_DEPTH) {
        // copy bits to upper positions to duplicate bit depth
        int depth = p.srcDepth;
        do {
            sample += sample << depth;
            depth += depth;
        } while(depth < p.outDepth);
    }
    
    // possibly crop the bit depth
    if(DISCARD_BIT_DEPTH) {
        sample >>= p.bitDepthShift;
    }
    
    // clamp to output range (with signed limits)
    sample = ::max(p.rangeMin, ::min(p.rangeMax, sample));
    
    // possibly do the DC level shift to get unsigned value range
    if(OUTPUT_IS_UNSIGNED) {
        sample += p.dcLevelShift;
    }
    
    // possibly shift the value to the left
    if(DO_FINAL_SHIFT) {
        sample <<= p.shiftLeft;
    }
    
    // output sample pointer
    TYPE * const outSamplePtr = (TYPE*)p.out
                              + p.outStride.x * pixX
                              + p.outStride.y * pixY;
    
    // either replace or combine the output value with the sample
    *outSamplePtr = OUTPUT_OR_COMBINE ? (sample | *outSamplePtr) : sample;
}



template<bool CONFIGURE_ONLY, typename TYPE, bool FLOAT, bool UNSIGNED, 
         bool SHIFT, bool OR, bool EXTEND, bool DISCARD>
static void fmtIntKernelSelect(const FormatIntKernelParams & params,
                               const cudaStream_t & stream) {
    if(CONFIGURE_ONLY) {
        // configure the kernel
        const cudaError_t result = cudaFuncSetCacheConfig(
            fmtIntKernel<TYPE, FLOAT, UNSIGNED, SHIFT, OR, EXTEND, DISCARD>,
            cudaFuncCachePreferShared
        );
        
        // check config status
        if(cudaSuccess != result) {
            throw Error(J2KD_ERROR_CUDA, "%s", cudaGetErrorString(result));
        }
    } else {
        // grid and threadblock size
        dim3 bSize(32, 8);
        dim3 gSize(divRndUp(params.imgSize.x, (int)bSize.x),
                   divRndUp(params.imgSize.y, (int)bSize.y));

        // launch the kernel
        fmtIntKernel<TYPE, FLOAT, UNSIGNED, SHIFT, OR, EXTEND, DISCARD>
                    <<<gSize, bSize, 0, stream>>>(params);
                    
//         const cudaError_t result = cudaDeviceSynchronize();
//         if(result != cudaSuccess) {
//             throw Error(J2KD_ERROR_CUDA, "%s", cudaGetErrorString(result));
//         }
    }
}



template<bool CONFIG, typename TYPE, bool FLOAT, bool UNSIGNED, 
         bool SHIFT, bool OR>
static void fmtIntKernelSelect(const FormatIntKernelParams & params,
                               const cudaStream_t & stream) {
    // decide whether to discard some bits and whether bid depth extension 
    // must be done
    if(params.outDepth > params.srcDepth) {
        if(params.bitDepthShift) {
            fmtIntKernelSelect<CONFIG, TYPE, FLOAT, UNSIGNED, SHIFT, OR,
                               true, true>(params, stream);
        } else {
            fmtIntKernelSelect<CONFIG, TYPE, FLOAT, UNSIGNED, SHIFT, OR,
                               true, false>(params, stream);
        }
    } else {
        if(params.bitDepthShift) {
            fmtIntKernelSelect<CONFIG, TYPE, FLOAT, UNSIGNED, SHIFT, OR,
                               false, true>(params, stream);
        } else {
            fmtIntKernelSelect<CONFIG, TYPE, FLOAT, UNSIGNED, SHIFT, OR,
                               false, false>(params, stream);
        }
    }
}



template<bool CONFIG, typename TYPE, bool FLOAT, bool OR>
static void fmtIntKernelSelect(const FormatIntKernelParams & params,
                               const cudaStream_t & stream) {
    // decide whether to discard some bits and whether bid depth extension 
    // must be done
    if(params.shiftLeft != 0) {
        if(params.dcLevelShift) {
            fmtIntKernelSelect<CONFIG, TYPE, FLOAT, true, true, OR>(params, stream);
        } else {
            fmtIntKernelSelect<CONFIG, TYPE, FLOAT, false, true, OR>(params, stream);
        }
    } else {
        if(params.dcLevelShift) {
            fmtIntKernelSelect<CONFIG, TYPE, FLOAT, true, false, OR>(params, stream);
        } else {
            fmtIntKernelSelect<CONFIG, TYPE, FLOAT, false, false, OR>(params, stream);
        }
    }
}



template <bool CONFIG, typename TYPE>
static void fmtIntKernelSelect(const bool srcFloat,
                               const bool useOR,
                               const FormatIntKernelParams & params,
                               const cudaStream_t & stream) {
    if(srcFloat) {
        if(useOR) {
            fmtIntKernelSelect<CONFIG, TYPE, true, true>(params, stream);
        } else {
            fmtIntKernelSelect<CONFIG, TYPE, true, false>(params, stream);
        }
    } else {
        if(useOR) {
            fmtIntKernelSelect<CONFIG, TYPE, false, true>(params, stream);
        } else {
            fmtIntKernelSelect<CONFIG, TYPE, false, false>(params, stream);
        }
    }
}



static void fmtIntKernelLaunch(const DataType outType, // only integer types are valid
                               const bool srcFloat,
                               const bool useOR,
                               const FormatIntKernelParams & params,
                               const cudaStream_t & stream) {
    // decide according to the data type
    switch (outType) {
        case J2KD_TYPE_INT8:
            fmtIntKernelSelect<false, u8>(srcFloat, useOR, params, stream);
            break;
        case J2KD_TYPE_INT16:
            fmtIntKernelSelect<false, u16>(srcFloat, useOR, params, stream);
            break;
        case J2KD_TYPE_INT32:
            fmtIntKernelSelect<false, u32>(srcFloat, useOR, params, stream);
            break;
        default:
            throw Error(J2KD_ERROR_UNSUPPORTED,
                        "Unsupported int type #%d.",
                        (int)outType);
    }
}



/// configures cache for all component formatting kernels
static void cacheConfig() {
    // integer kernels configuration parameters
    FormatIntKernelParams ip;
    ip.src = 0;
    ip.srcStrideY = 0;
    ip.out = 0;
    ip.outStride.x = 0;
    ip.outStride.y = 0;
    ip.imgSize.x = 1;
    ip.imgSize.y = 1;
    ip.shiftLeft = 0;
    ip.rangeMin = 0;
    ip.rangeMax = 1;
    ip.dcLevelShift = 0;
    ip.bitDepthShift = 0;
    ip.srcDepth = 0;
    ip.outDepth = 0;

    // output is/not unsigned
    for(ip.dcLevelShift = 2; ip.dcLevelShift--;) {
        // adjust is/not required
        for(ip.shiftLeft = 2; ip.shiftLeft--; ) {
            // combine/replace output
            for(int useOR = 2; useOR--; ) {
                // reduce bit depth or not
                for(ip.bitDepthShift = 2; ip.bitDepthShift--; ) {
                    // extend bit depth or not
                    for(ip.outDepth = 2; ip.outDepth--; ) {
                        // source is float
                        fmtIntKernelSelect<true, u8>(true, useOR, ip, 0);
                        fmtIntKernelSelect<true, u8>(true, useOR, ip, 0);
                        fmtIntKernelSelect<true, u8>(true, useOR, ip, 0);
                        
                        // source if not float
                        fmtIntKernelSelect<true, u8>(false, useOR, ip, 0);
                        fmtIntKernelSelect<true, u8>(false, useOR, ip, 0);
                        fmtIntKernelSelect<true, u8>(false, useOR, ip, 0);
                    }
                }
            }
        }
    }
    
    // TODO: configure float kernels as soon as they are implemented
}




    

    
    
// /// Kernel parameters including image info.    
// struct FmtParams {
//     XY imgBegin;    ///< begin coordinates of image
//     XY imgEnd;      ///< end coordinates of image
//     XY tileOrigin;
//     XY tileSize;
//     const void * input; ///< input component samples (either floats or ints)
//     void * output;
//     const Tile * tiles;
// //     const TileCoding * tCodings;
//     const TComp * tComps;
//     const TCompCoding * tCompCodings;
//     const Comp * comps;
//     int tileCountX;
//     int compFmtCount;                      ///< number of formatted components.
// };



   


// /// Arranges decoded image into required format.
// /// @param formatCount  number of components to be formatted
// /// @param formats      pointer to array of structures to be formatted
// __device__ inline void formatOutput(const CompFormat * const formats,
//                                     const FmtParams & params) {
//     // position of thread's pixel is specified both using thread's and 
//     // block's X and Y coordinates (position is relative to tile origin)
//     const int pixX = params.imgBegin.x + threadIdx.x + blockIdx.x * blockDim.x;
//     const int pixY = params.imgBegin.y + threadIdx.y + blockIdx.y * blockDim.y;
//     if(pixY >= params.imgEnd.y || pixX >= params.imgEnd.x) {
//         return;   // out of image bounds
//     }
//     
//     // position of pixel relative to tile origin
//     const int relPosX = pixX - params.tileOrigin.x;
//     const int relPosY = pixY - params.tileOrigin.y;
//     
//     // coordinates of the tile and coordinates of the pixel within that tile
//     const int tileX = relPosX / params.tileSize.x;
//     const int tileY = relPosY / params.tileSize.y;
//     const int tPixPosX = relPosX - (params.tileSize.x * tileX);
//     const int tilePixPosY = relPosY - (params.tileSize.y * tileY);
//     
//     // index of first tile-component of thread's tile
//     const int tCompOffset = params.tiles[tileX * tileY * params.tileCountX].tCompIdx;
//     
//     // format all component-samples of the pixel
//     for(int fmtIdx = 0; fmtIdx < params.compFmtCount; fmtIdx++) {
//         const CompFormat & fmt = formats[fmtIdx];
//         const int compIdx = fmt.component_idx;
//         
//         // pointer to related tile-component info
//         const TComp * const tComp = params.tComps + tCompOffset + compIdx;
//         
//         // input and output indices of the sample
//         const int outIdx = fmt.offset + pixX * fmt.stride_x + pixY * fmt.stride_y;
//         const int inIdx = tComp->outPixOffset + tPixPosX + tilePixPosY * tComp->outPixStride;
//         
//         // bit depth adjust
//         const int inDepth = params.comps[compIdx].bitDepth;
//         const int outDepth = fmt.bit_depth;
//         
//         // true if input is integer
//         const bool inputIsInt = params.tCompCodings[tComp->codingIdx].reversible;
//         
//         // is output integer or real?
//         if(fmt.type == J2KD_TYPE_FLOAT32 || fmt.type == J2KD_TYPE_FLOAT64) {
//             // either load as float or as int (if reversible coding used)
//             float sample = inputIsInt
//                          ? (float) (((const int *)params.input)[inIdx])
//                          : ((const float *)params.input)[inIdx];
//             
//             // eventually adjust output bit depth
//             if(inDepth != outDepth) {
//                 const float outMax = (1 << outDepth) - 1;
//                 const float inMax = (1 << inDepth) - 1;
//                 sample *= outMax / inMax;
//             }
//             
//             // save the sample
//             if(fmt.type == J2KD_TYPE_FLOAT32) {
//                 ((float*)params.output)[outIdx] = sample;
//             } else {
//                 ((double*)params.output)[outIdx] = sample;
//             }
//         } else {
//             // either load as float or as int (if reversible coding used)
//             int sample = inputIsInt
//                     ? ((const int *)params.input)[inIdx]
//                     : (int) (0.5f + ((const float *)params.input)[inIdx]);
//             
//             
//             // possibly adjust the output bit depth
//             if(inDepth != outDepth) {
//                 const s64 outMax = (1 << outDepth) - 1;
//                 const int inMax = (1 << inDepth) - 1;
//                 const int halfInMax = inMax >> 1;
//                 sample = (int)((halfInMax + outMax * sample) / inMax);
//             }
//             
//             // save the sample
//             switch(fmt.type) {
//                 case J2KD_TYPE_S8:
//                     ((s8*)params.output)[outIdx] = (s8)sample; break;
//                 case J2KD_TYPE_U8:
//                     ((u8*)params.output)[outIdx] = (u8)sample; break;
//                 case J2KD_TYPE_S16:
//                     ((s16*)params.output)[outIdx] = (s16)sample; break;
//                 case J2KD_TYPE_U16:
//                     ((u16*)params.output)[outIdx] = (u16)sample; break;
//                 case J2KD_TYPE_S32:
//                     ((s32*)params.output)[outIdx] = (s32)sample; break;
//                 case J2KD_TYPE_U32:
//                     ((u32*)params.output)[outIdx] = (u32)sample; break;
// //                 case J2KD_TYPE_S64:
// //                     ((s64*)params.output)[outIdx] = (s64)sample; break;
// //                 case J2KD_TYPE_U64:
// //                     ((u64*)params.output)[outIdx] = (u64)sample; break;
//             }
//         }
//     }
// }






/// Gets size of the output type in bytes.
static int getTypeSize(const DataType & t) {
    switch (t) {
        case J2KD_TYPE_INT8:
            return 1;
        case J2KD_TYPE_INT16:
            return 2;
        case J2KD_TYPE_INT32:
//         case J2KD_TYPE_FLOAT32:
            return 4;
//         case J2KD_TYPE_FLOAT64:
//             return 8;
        default:
            throw Error(J2KD_ERROR_UNKNOWN, "Unknown data type #%d.", (int)t);
    }
}



/// Computes index of pixel at given poistion and makes sure that it lies 
/// within the buffer, updating the required buffer size.
static void checkSize(size_t & maxSizeRequired,
                      const CompFormat & fmt,
                      const int fmtIdx,
                      const size_t bufferSize,
                      const int pixX,
                      const int pixY) {
    // compute offset of the pixel in the buffer
    const int typeSize = getTypeSize(fmt.type);
    if(typeSize == 0) {
        throw Error(J2KD_ERROR_ARGUMENT_OUT_OF_RANGE,
                    "Unknown output data type for component format #%d.",
                    fmtIdx);
    }
    const int64_t idx = fmt.offset + fmt.stride_x * pixX + fmt.stride_y * pixY;
    if(idx < 0) {
        throw Error(J2KD_ERROR_SMALL_BUFFER,
                    "Negative index required for sample of component #%d "
                    "(pixel's %d,%d).",
                    fmt.component_idx, pixX, pixY);
    }
    const size_t requiredBufferSize = (idx + 1) * typeSize;
    if(requiredBufferSize > bufferSize) {
        throw Error(J2KD_ERROR_SMALL_BUFFER,
                    "Sample of component #%d, pixel %d,%d should be saved "
                    "at offset %d, but buffer size is only %d.",
                    fmt.component_idx, pixX, pixY,
                    (int)(requiredBufferSize - typeSize), (int)bufferSize);
    }
    if(requiredBufferSize > maxSizeRequired) {
        maxSizeRequired = requiredBufferSize;
    }
}



/// Computes formatted output size, checking for indices out of bounds.
/// @param image  pointer to image structure instance
/// @param format  pointer to array of component format infos
///                (must be immutable at least to end of current decoding)
/// @param count   number of output components
/// @param capacity  capacity of the output buffer
/// @return size of the buffer needed for formatted data
size_t OutputFormatter::checkFmt(
    Image * const image,
    const CompFormat * const format,
    const int count,
    const size_t capacity
) {
    // references to image limits
    const XY & begin = image->imgBegin;
    const XY & end = image->imgEnd;
    
    // this will contain ouput size (maximal used output byte index + 1)
    size_t requiredSize = 0;
    
    // format count must be positive
    if(count < 1) {
        throw Error(J2KD_ERROR_ARGUMENT_OUT_OF_RANGE,
                    "Nothing to decode - use at least 1 component format.");
    }
    
    // format pointer must not be null
    if(0 == format) {
        throw Error(J2KD_ERROR_ARGUMENT_NULL,
                    "NULL pointer to component formats.");
    }
    
    // for each format ...
    for(int fmtIdx = count; fmtIdx--;) {
        const CompFormat & fmt = format[fmtIdx];
        
        // component index must be nonnegative
        if(fmt.component_idx < 0) {
            throw Error(J2KD_ERROR_ARGUMENT_OUT_OF_RANGE,
                        "Negative component index (%d) for format #%d.",
                        fmt.component_idx, fmtIdx);
        }
        
        // component index must be less than component count
        if(fmt.component_idx > (int)image->comps.count()) {
            throw Error(J2KD_ERROR_ARGUMENT_OUT_OF_RANGE,
                        "Component index (%d) for format #%d greater than "
                        "number of components (%d).",
                        fmt.component_idx, fmtIdx, (int)image->comps.count());
        }
        
        // bit depth must not be negative
        if(fmt.bit_depth < 0) {
            throw Error(J2KD_ERROR_ARGUMENT_OUT_OF_RANGE,
                        "Negative bit depth '%d' required for component %d",
                        fmt.bit_depth, fmt.component_idx);
        }
        
        // maximal bit depth is 32 bits per sample
        if(fmt.bit_depth > 32) {
            throw Error(J2KD_ERROR_ARGUMENT_OUT_OF_RANGE,
                        "Too big bit depth '%d' required for component %d",
                        fmt.bit_depth, fmt.component_idx);
        }
        
        // all 4 corner pixel must fit into the buffer (so that all pixels fit)
        checkSize(requiredSize, fmt, fmtIdx, capacity, begin.x, begin.y);
        checkSize(requiredSize, fmt, fmtIdx, capacity, begin.x, end.y - 1);
        checkSize(requiredSize, fmt, fmtIdx, capacity, end.x - 1, begin.y);
        checkSize(requiredSize, fmt, fmtIdx, capacity, end.x - 1, end.y - 1);
    }
    
    // return minimal buffer size needed to contain all formatted data
    return requiredSize;
}



/// This does the output formatting.
/// @param image   pointer to image structure instance
/// @param srcPtr  source buffer
/// @param outPtr  pointer to GPU output buffer
/// @param stream  CUDA stream to launch kernels in
/// @param format  pointer to array of component format infos
///                (must be immutable at least to end of current decoding)
/// @param count   number of output components
void OutputFormatter::run(
    Image * const image,
    const void * const srcPtr,
    void * const outPtr,
    const cudaStream_t & stream,
    const CompFormat * const format,
    const int count
) {
    // for each formatted component
    for(int fmtIdx = 0; fmtIdx < count; fmtIdx++) {
        // pointer to component formatting info
        const CompFormat & fmt = format[fmtIdx];
        
        // for each tile
        for(size_t tileIdx = 0; tileIdx < image->tiles.count(); tileIdx++) {
            // pointer to right tile, component, tile-component and its coding
            const Tile & tile = image->tiles[tileIdx];
            const Comp & comp = image->comps[fmt.component_idx];
            const TComp & tComp = image->tComps[tile.tCompIdx + fmt.component_idx];
            const TCompCoding & tCompCod = image->tCompCoding[tComp.codingIdx];
            
            // top-left source sample pointer and source stride
            const void * const src = (u8*)srcPtr + 4 * tComp.outPixOffset;
            const int srcStrideY = tComp.outPixStride;
            
            // output pointer
            const int outOffset = fmt.offset
                                + fmt.stride_x * tile.pixBegin.x
                                + fmt.stride_y * tile.pixBegin.y;
            void * const out = (u8*)outPtr + outOffset * getTypeSize(fmt.type);
            
            // compose formatting parameters
            const int rangeHalf = 1 << (fmt.bit_depth - 1);
            FormatIntKernelParams p;
            p.src = src;
            p.srcStrideY = srcStrideY;
            p.out = out;
            p.outStride = XY(fmt.stride_x, fmt.stride_y);
            p.imgSize = tile.pixEnd - tile.pixBegin;
            p.shiftLeft = fmt.final_shl;
            p.srcDepth = comp.bitDepth;
            p.outDepth = fmt.bit_depth;
            p.dcLevelShift = fmt.is_signed ? 0 : rangeHalf;
            p.rangeMin = - rangeHalf;
            p.rangeMax = (1 << p.outDepth) - 1 - rangeHalf;
            
            // prepare bit depth shift adjust
            int depth = p.srcDepth;
            while(depth < p.outDepth) {
                depth *= 2;
            }
            p.bitDepthShift = depth - p.outDepth;
            
            // launch the kernel
            fmtIntKernelLaunch(fmt.type, 0 == tCompCod.reversible,
                               fmt.combine_or, p, stream);
        }
    }
}



/// Constructor - Initializes static formatting stuff
OutputFormatter::OutputFormatter() {
    cacheConfig();
}



} // end of namespace cuj2kd
