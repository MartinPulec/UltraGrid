///
/// @file    j2kd_mct.cu
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Implementation of JPEG 2000 reverse MCT.
///

#include "j2kd_mct.h"

namespace cuj2kd {



/// Parameters of the JPEG 2000 decoder MCT.
struct MCTParams {
    void * data;
    int compCount;
    const Tile * tiles;
    const TileCoding * tCodings;
    const TCompCoding * tCompCodings;
    const TComp * tComps;
};



/// MCT kernel (can reverse MCT in up to 65536 tiles at once).
__global__ static void mctKernel(const int tileOffset, const MCTParams p) {
    // pointer to threadblock's tile info
    const Tile * const tile = p.tiles + tileOffset + blockIdx.z;
    
    // coordinates of pixel, whose samples will be processed by this thread
    const int pixX = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixY = threadIdx.y + blockIdx.y * blockDim.y;
    
    // stop if pixel is out of tile bounds
    if(pixY >= tile->pixEnd.y - tile->pixBegin.y) {
        return;
    }
    if(pixX >= tile->pixEnd.x - tile->pixBegin.x) {
        return;
    }
    
    // index of first tile-component of this tile
    const int tCompOffset = tile->tCompIdx;
        
    // chheck whether the MCT should be done in this tile
    if(p.compCount >= 3 && p.tCodings[tile->tileCodingIdx].useMct) {
        // pointer to array of 3 tile-components to be transformed transformed
        const TComp * const tComps = p.tComps + tCompOffset;
        
        // indices of all 3 samples
        const int rIdx = tComps[0].outPixOffset + pixX + pixY * tComps[0].outPixStride;
        const int gIdx = tComps[1].outPixOffset + pixX + pixY * tComps[1].outPixStride;
        const int bIdx = tComps[2].outPixOffset + pixX + pixY * tComps[2].outPixStride;
        
        // reversible or irreversible version? (All 3 first transformed 
        // components should share same settings.)
        if(p.tCompCodings[tComps[0].codingIdx].reversible) {
            // reversible => load 3 ints
            const int r = ((const int*)p.data)[rIdx];
            const int g = ((const int*)p.data)[gIdx];
            const int b = ((const int*)p.data)[bIdx];
            
            // save 3 transformed values
            const int gNew = r - ((g + b) >> 2);
            ((int*)p.data)[rIdx] = gNew + b;
            ((int*)p.data)[gIdx] = gNew;
            ((int*)p.data)[bIdx] = gNew + g;
            
        } else {
            // ireversible => load 3 floats
            const float r = ((const float*)p.data)[rIdx];
            const float g = ((const float*)p.data)[gIdx];
            const float b = ((const float*)p.data)[bIdx];
            
            // save 3 transformed floats
            ((float*)p.data)[rIdx] = r + 1.402f * b;
            ((float*)p.data)[gIdx] = r - 0.34413f * g - 0.71414f * b;
            ((float*)p.data)[bIdx] = r + 1.772f * g;
        }
    }
}



/// Gets true if inverse MCT needed.
static bool mctNeeded(const Image * const image) {
    for(int tileIdx = image->tiles.count(); tileIdx--; ) {
        if(image->tileCoding[image->tiles[tileIdx].tileCodingIdx].useMct) {
            return true;
        }
    }
    return false;
}



/// Performs reverse MCT.
/// @param image  pointer to image structure
/// @param data  pointer to buffer with data
/// @param cudaStream  cuda stream to be used for decoding kernels
/// @param logger  logger for tracing progress of the decoding
void mctUndo(Image * const image,
             void * const data,
             const cudaStream_t & cudaStream,
             Logger * const logger) {
    // run the kernel only if DC level shifting or MCT is needed
    if(mctNeeded(image)) {
        // common parameters of the kernel
        MCTParams params;
        params.data = data;
        params.compCount = image->comps.count();
        params.tiles = image->tiles.getPtrGPU();
        params.tCodings = image->tileCoding.getPtrGPU();
        params.tCompCodings = image->tCompCoding.getPtrGPU();
        params.tComps = image->tComps.getPtrGPU();
        
        // for each block of 65536 tiles :) ...
        for(u32 tOffset = 0; tOffset < image->tiles.count(); tOffset += 0xFFFF) {
            // launch configuration
            const dim3 bSize(64, 8);
            const dim3 gSize(divRndUp(image->tSize.x, (int)bSize.x),
                             divRndUp(image->tSize.y, (int)bSize.y),
                             min(0xFFFF, (int)(image->tiles.count() - tOffset)));

            // launch the kernel
            mctKernel<<<gSize, bSize, 0, cudaStream>>>(tOffset, params);
        }
    }
}



} // end of namespace cuj2kd
