///
/// @file    j2kd_type.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Definition of internal data types used in JPEG 2000 decoder.
///


// prevent from multiple includes into the same file
#ifndef J2KD_TYPE_H
#define J2KD_TYPE_H


// #include <cstdlib>
// #include <cstring>
#include <stddef.h>
#include <stdint.h>

// include interface
#include "j2kd_api.h"

namespace cuj2kd {


// import interface types into this namespace
typedef j2kd_image_info ImageInfo;
typedef j2kd_component_info ComponentInfo;
typedef j2kd_data_type DataType;
typedef j2kd_component_format CompFormat;
typedef j2kd_status_code StatusCode;
typedef j2kd_input_begin_callback InBeginCallback;
typedef j2kd_input_end_callback InEndCallback;
typedef j2kd_output_callback OutCallback;
typedef j2kd_postprocessing_callback PostprocCallback;
typedef j2kd_decoding_end_callback DecEndCallback;


/// Progression order type.
enum ProgOrder {
    PO_LRCP = 0,  ///< Layer-Resolution-Component-Position (default)
    PO_RLCP = 1,  ///< Resolution-Layer-Component-Position
    PO_RPCL = 2,  ///< Resolution-Position-Component-Layer
    PO_PCRL = 3,  ///< Position-Component-Resolution-Layer
    PO_CPRL = 4   ///< Component-Position-Resolution-Layer
}; 


/// 2D point or vector type.
struct XY {
    int x, y;
    XY(const int x, const int y) : x(x), y(y) {}
    XY() : x(0), y(0) {}
    XY operator-(const XY & o) const { return XY(x - o.x, y - o.y); }
    XY operator+(const XY & o) const { return XY(x + o.x, y + o.y); }
    XY operator/(const XY & o) const { return XY(x / o.x, y / o.y); }
    XY operator-(const int n) const { return XY(x - n, y - n); }
    XY operator&(const XY & mask) const { return XY(x & mask.x, y & mask.y); }
    XY operator~() const { return XY(~x, ~y); }
    bool operator<(const XY & o) const { return y == o.y ? x < o.x : y < o.y; }
    bool operator>(const XY & o) const { return o < *this; }
    bool operator==(const XY & o) const { return x == o.x && y == o.y; }
    bool operator!=(const XY & o) const { return !this->operator==(o); }
};


// Shorthands for basic type names.
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;


/// Orientation of the band.
enum Orientation {
    ORI_LL = 0,
    ORI_HL = 1,
    ORI_LH = 2,
    ORI_HH = 3
};


/// Quantization mode.
enum QuantMode {
    QM_NONE = 0,        ///< only dynamic range exponents signalized
    QM_IMPLICIT = 1,    ///< stepsize for LL band only
    QM_EXPLICIT = 2     ///< stepsizes for each band
};


// Forward declaration of packet iterator
class PacketIterator;


/// Represents one tile.
struct Tile {
    XY pixBegin;        ///< 0-based coordinates of tile's top-left pixel
    XY pixEnd;          ///< 0-based coords past tile's bottom-right pixel
    int tileIdx;        ///< 0-based index of the tile (in raster order)
    int tCompIdx;       ///< index of first tile-component of this tile
    int nextTPartIdx;   ///< expected index of next tile-part
    int tileCodingIdx;  ///< index of tile coding style for the tile
    PacketIterator * iter;  ///< CPU pointer to tile's packet iterator
}; // end of struct Tile


/// Tile-component.
struct TComp {
    int compIdx;        ///< index of the component
    int tileIdx;        ///< index of tile to which this component belongs
    int resIdx;         ///< index of first resolution (resolution 0)
    int resCount;       ///< count of tile's resolutions
    int quantIdx;       ///< index of component's quantization settings
    int codingIdx;      ///< index of coding style for this tile-component
    int outPixOffset;   ///< offset of first output pixel  // TODO: rename to outOffset
    int outPixStride;   ///< y-stride of output pixels     // TODO: rename to outStride
};


/// Resolution.
struct Res {
    int resolution;     ///< number of this resolution (e.g. 0 for LL only)
    int dwtCount;       ///< number of DWT levels to get this resolution
    int tCompIdx;       ///< index of resolution's parent tile-component
    int bandOffset;     ///< index of first band of this resolution
    int bandCount;      ///< number of bands in this resolution (1 or 3)
    XY precCount;       ///< numbers of precincts along both axes
    int outPixOffset;   ///< offset of first output pixel of whole resolution
    int outPixStride;   ///< Y-stride of output pixels
    XY begin;           ///< coordinates of top-left sample of the reolution
    XY end;             ///< coordinates past bottom-right sample
};


/// Quantization stepsize
struct Stepsize {
    int mantisa;        ///< quantization mantisa
    int exponent;       ///< quantization exponent
    Stepsize(const int m, const int e) : mantisa(m), exponent(e) {}
    Stepsize() {}
};


/// Band
struct Band {
    Orientation orient; ///< orientation of the band
    int resIdx;         ///< Index of band's parent resolution
    XY pixBegin;        ///< begin of the band (in pixels) scaled to resolution
    XY pixEnd;          ///< end of the band (in pixels) scaled to resolution
    int outPixOffset;   ///< index of band's first pixel in common output buffer
    int outPixStride;   ///< vertical difference between output indices
//     int cblkIdx;        ///< index (position) of band's first codeblock
//     int cblkCount;      ///< number of band's codeblocks
//     Stepsize stepsize;  ///< quantization stepsize
    float stepsize;    ///< stepsize as one number (equation E-3)
    int bitDepth;       ///< number of band's samples' bits (including guards)
    bool reversible;    ///< true if reversible coding used
};


/// Codeblock
struct Cblk {
    int bandIdx;        ///< index of parent band (in buffer for all bands)
    XY pos;             ///< codeblock's coordinates, relative to the band
    XY size;            ///< size of this codeblock in pixels
    XY stdSize;         ///< standard (uncropped) size of the codeblock
    int firstSegIdx;    ///< index of codeblock's first codestream chunk (or -1)
    int lastSegIdx;     ///< index of codeblock's last codestream chunk (or -1)
    int bplnCount;      ///< codeblock's encoded bitplanes count
    int passCount;      ///< total count of passes in all codeblock's segments
    int totalBytes;     ///< total codestream byte count (across all segments)
    u32 ebcotTemp;      ///< codeblock's temporary storage offset in EBCOT
        
    /// base bit count for codeblock's segment sizes signaled in packet headers
    /// (or 0 if not included in any packet yet)
    int segLenBits;
};


/// Represents input codestream segment contributing to some codeblock.
struct Seg {
    int cblkIdx;        ///< index of related codeblock
    int nextSegIdx;     ///< index of codeblock's next codestream segment or -1
    int codeByteOffset; ///< index of first codestream byte
    int codeByteCount;  ///< count of code bytes of this chunk
    u8 passCount;       ///< number of passes in this segment
    bool bypassAC;      ///< true if data in this segment bypass MQ coder
};


/// Represents component info.
struct Comp {
    int compIdx;        ///< color comp index represented by this structure
    int bitDepth;       ///< bit depth of the color component
    bool isSigned;      ///< true if color component is signed
    int defCStyleIdx;   ///< Index of default coding style for the component
    int defQuantIdx;    ///< Index of default quantization settings
};


/// T2 specific info about component's coding style.
struct TCompCoding  {
    int dwtLevelCount;  ///< number of DWT levels (0 - 32, both inclusive)
    XY cblkSize;        ///< codeblock size
    bool reversible;    ///< true iff reversible coding transforms used
    bool bypassAC;      ///< true if selective AC bypass mode is used
    bool termAll;       ///< true if AC terminates stream after each pass
    bool resetProb;     ///< true for context probabilities reset on each pass
    bool vericalCausal; ///< true for vertical causal context modeller mode
    bool predictTerm;   ///< true for AC's predictable termination mode
    bool segSymbols;    ///< true for using of segmentation symbols in AC
    int precSizesCount; ///< number of listed precinct sizes 
    const u8 * precSizesPtr; ///< packed precinct sizes in CPU codestream
    
//     /// Gets precinct size for given resolution.
//     j2kd_xy get_prec_size(const int res_idx) const {
//         // default precinct size
//         j2kd_xy prec_size = {1 << 15, 1 << 15};
//         
//         // have enough explicit precinct sizes?
//         if(res_idx < num_precinct_sizes) {
//             // for resolution 0, precincts are double-sized
//             const int base = res_idx == 0 ? 2 : 1;
//             
//             // unpack size
//             const u8 packed_size = packed_prec_sizes_ptr[res_idx];
//             prec_size.x = base << (packed_size & 0xF);
//             prec_size.y = base << (packed_size >> 4);
//         }
//         
//         // return the size
//         return prec_size;
//     }
};


/// T2 specific stuff related to tile coding.
struct TileCoding {
    int layerCount;         ///< layer count
    bool useMct;            ///< multi-component transform
    bool useSop;            ///< true if SOP markers start packet headers
    bool useEph;            ///< true if EPH markers terminate packet data
};


/// T2 specific info about component's quantization.
struct TCompQuant {
    QuantMode mode;         ///< quantization mode used in tile-component
    int guardBitCount;      ///< number of guard bits   TODO: is this needed?
    int stepsizeBytes;      ///< number of stepsize or dynamic-range bytes
    const u8 * stepsizePtr; ///< packed stepsizes or ranges in CPU codestream
};


} // end of namespace cuj2kd

#endif // J2KD_TYPE_H
