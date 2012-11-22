///
/// @file    j2kd_image.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Image structure info.
///

#ifndef J2KD_IMAGE_H
#define J2KD_IMAGE_H

#include "j2kd_type.h"
#include "j2kd_buffer.h"

namespace cuj2kd {



/// Image structure info.
struct Image {
    XY imgBegin;                ///< origin of image
    XY imgEnd;                  ///< end of image
    XY tOrigin;                 ///< origin of tiles
    XY tSize;                   ///< base size of all tiles
    XY tCount;                  ///< count of tiles along both axes
    int capabilities;           ///< codestream capabilities identifier
    BufferPair<Tile> tiles;     ///< all tiles, in raster order
    BufferPair<TComp> tComps;   ///< all tile-components in all tiles
    BufferPair<Res> res;        ///< all resolutions of all tiles
    BufferPair<Band> bands;     ///< all bands of all resolutions
    BufferPair<Cblk> cblks;     ///< all codeblocks of all bands
    BufferPair<Comp> comps;     ///< buffers for info about components
    BufferPair<Seg> segs;       ///< codestream segments (codeblock data parts)
    BufferPair<TCompCoding> tCompCoding; ///< components' coding styles
    BufferPair<TileCoding> tileCoding;   ///< coding of tiles
    BufferPair<u32> cblkPerm;   /// codeblock permutation (coalescent decoding)
    
    /// pixel count across all bands and resolutions (including padding)
    size_t bandsPixelCount;  // TODO: rename ot bandsSampleCount
    
    /// size of temporary buffer needed for EBCOT
    size_t ebcotTempSize;
    
    /// Clears all buffers and usage counts.
    void clear() {
        bandsPixelCount = 0;
        tiles.clear();
        tComps.clear();
        res.clear();
        bands.clear();
        cblks.clear();
        comps.clear();
        segs.clear();
        tCompCoding.clear();
        tileCoding.clear();
        cblkPerm.clear();
    }
    
    /// Copies all buffers to GPU.
    void copyToGPU(const cudaStream_t cudaStream) {
        tiles.copyToGPUAsync(cudaStream);
        tComps.copyToGPUAsync(cudaStream);
        res.copyToGPUAsync(cudaStream);
        bands.copyToGPUAsync(cudaStream);
        cblks.copyToGPUAsync(cudaStream);
        comps.copyToGPUAsync(cudaStream);
        segs.copyToGPUAsync(cudaStream);
        tCompCoding.copyToGPUAsync(cudaStream);
        tileCoding.copyToGPUAsync(cudaStream);
        cblkPerm.copyToGPUAsync(cudaStream);
    }
};



} // end of namespace cuj2kd

#endif // J2KD_IMAGE_H
