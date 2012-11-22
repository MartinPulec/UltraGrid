///
/// @file    j2kd_ebcot.cu
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Implementation of JPEG 2000 decoder EBCOT.
///


#include "j2kd_ebcot.h"
#include "j2kd_type.h"
#include "j2kd_timer.h"
#include <stdio.h>

namespace cuj2kd {


/// Contexts
enum CX {
    ZC0 = 0,
    ZC1 = 1,
    ZC2 = 2,
    ZC3 = 3,
    ZC4 = 4,
    ZC5 = 5,
    ZC6 = 6,
    ZC7 = 7,
    ZC8 = 8,

    UNI = 9,
    
    MRC0 = 11,
    MRC1 = 13,
    MRC2 = 15,
    
    // all SCs are even for the XOR bit to be able to be stored in LSB
    SC0 = 10,
    SC1 = 12,
    SC2 = 14,
    SC3 = 16,
    SC4 = 18,
    
    RLC = 17
};


/// packed probability estimation together with next states 
/// for both MPS and LPS exchanges in MQ decoder:
///        +---------------------------+---+---+------------+------------+
/// what:  |         Qe (16 b)         | 0 | 0 | NLPS (7 b) | NMPS (7 b) |
///        +---------------------------+---+---+------------+------------+
/// bit:    31                       16  15  14 13         7 6          0
__device__ static u32 nQe[47 * 2];
__shared__ u32 nQeShared[47 * 2];



/// packed significance flags of neighbors, 
/// 3 versions of the table (for different band orientations)
///        +---+---+---+---+---+---+---+---+---+
/// what:  | H | C | H | D | V | D | D | V | D | 
///        +---+---+---+---+---+---+---+---+---+
/// bit:     8   7   6   5   4   3   2   1   0
///        H = horizontal neighbor
///        V = vertical neighbor
///        D = diagonal neighbor
///        C = central pixel
__device__ static u8 llZcLut[512];  // for LL and LH bands
__device__ static u8 hlZcLut[512];  // for HL bands
__device__ static u8 hhZcLut[512];  // for HH bands



/// index consists of packed sign and significancy states of neighbors
///        +----+----+----+----+----+----+----+----+----+
/// what:  | Vs | V+ | Vs | V+ | Hs | H+ | ?? | Hs | H+ |
///        +----+----+----+----+----+----+----+----+----+
/// bit:      8    7    6    5    4    3    2    1    0
///         V = vertical neighbor
///         H = horizontal neighbor
///         + = sign bit (1 for -, 0 for +)
///         s = significancy flag (1 = significant, 0 = insignificant)
/// 
/// Contents of the LUT fields: 2 * (sign coding context) + (XOR bit)
__device__ __constant__ static u8 scLut[512];



/// Parameters for EBCOT kernel.
struct EbcotKernelParams {
    /// pointer to GPU buffer for temporary state of all cblocks
    void * output;
    
    /// GPU pointer to input codestream
    const u8 * cstream;
    
    /// pointer to GPU buffer with all codeblocks
    Cblk * cblks;
    
    /// total codeblock count (in all components/bands/resolutions/tiles)
    int cblkCount; 
    
    /// pointer to GPU buffer with codestream segments
    const Seg * segs; 
    
    /// pointer to GPU buffer with all band info structures
    const Band * bands;
    
    /// pointer to GPU buffer with all resolution structures
    const Res * res;
    
    /// pointer to GPU buffer with all component structures
    const Comp * comps;
    
    /// permutation indices for codeblocks
    const u32 * cblkPermuation;
}; // end of struct EbcotKernelParams



/// Dynamic (mutable) lookup tables for decoding of one codeblock.
struct CblkDecState {
    /// array of probability indices for all modeller contexts (values 0 to 46)
    u8 indices[19];
    
    /// padding to fit the structure into odd number of shared memory lanes
    u8 padding[1];
}; // end of struct CblkDecState



class BufferedReader {
private:
    // 32byte buffer aligned to 16byte boundary
    u8 * const buffer;  
    
    // index of next byte to be read from the buffer
    int bufferByteIdx;
    
    // count of remaining bytes
    int remainingBytes;
    
    // next source pointer (aligned to 16byte boundary)
    const u8 * srcNextPtr;

public:
    __device__ BufferedReader(u8 * const buffer) : buffer(buffer) {
        // Nothing to do here
    }
    
    __device__ void init(const u8 * const src, const int byteCount) {
        // index of first byte to be read from shared buffer
        bufferByteIdx = 15 & (size_t)src;
        
        // save begin pointer aligned to 16byte boundary
        srcNextPtr = src - bufferByteIdx;
        
        // save count of remaining bytes
        remainingBytes = byteCount;
        
        // read first 32 bytes of input
        ((int4*)buffer)[0] = ((int4*)srcNextPtr)[0];
        ((int4*)buffer)[1] = ((int4*)srcNextPtr)[1];
        srcNextPtr += 32;
    }
    
    __device__ u32 next2() {
        // read 2 bytes (in reversed order)
        u32 bytes = *(u16*)(buffer + bufferByteIdx);
        
        // set second byte to ones if reader at the end
        if(1 == remainingBytes) {
            bytes |= 0xFF00;
        }
        
        // advance byte counters
        bufferByteIdx += 2;
        remainingBytes -= 2;
        
        // return swapped bytes (in correct order)
        return __byte_perm(bytes, 0, 0x3201);
    }
    
    __device__ void load() {
        // possibly read more bytes into the buffer
        if(bufferByteIdx >= 16) {
            // replace contents of first half of the buffer with other half
            // and move new data to the other half
            ((int4*)buffer)[0] = ((int4*)buffer)[1];
            ((int4*)buffer)[1] = *(int4*)srcNextPtr;
            
            // advance source pointer and update index of next byte to be read
            srcNextPtr += 16;
            bufferByteIdx -= 16;
        }
    }
    
    __device__ bool end() const {
        return remainingBytes <= 0;
    }
}; // end of class BufferedReader



class MQDecoder {
private:
    /// reader for reading code bytes
    BufferedReader reader;
    
    /// boundary between MPS and LPS probability (upshifted in upper 16 bits)
    u32 A;
    
    /// code register (Upper 16 bits contain code for comparation with 
    /// probability register and lower 16 bits may contain additional 
    /// code bits, which are used, when whole code register is shifted up.)
    u32 C;
    
    /// extra code bit count in LSBs of code register (0 to 16, both inclusive)
    int extraCodeBitCount;
    
    /// set of states of this decoder (one index for each of 19 contexts)
    CblkDecState & state;
    
    /// 2 if bit unstuffing should be done on next byte, 1 if not
    int unstuff;
    
    /// Reads 2 more code bytes into 16 LSBs of code register. (Assumes that 
    /// there are no extra code bits left in LSBs of code register.)
    __device__ void read2CodeBytes() {
        // TODO: do unstuffing in separate kernel for each segment!!
        //       And add sufficient number of 0xFFs to the end to be able 
        //       to just simply skip final reads
        
        if(reader.end()) {
            // end of codestream => set all extra bits to 1
            extraCodeBitCount = 16;
            C |= 0xFFFF;
        } else {
            // initialize new count of extra bits
            extraCodeBitCount = 17 - unstuff;
            
            // load next two bytes
            u32 bits = reader.next2();
            
            // remember whether to unstuff first bit of next pair of bytes
            const int nextUnstuff = (bits & 0xFF) == 0xFF ? 2 : 1;
            
            // possibly unstuff second byte (if first byte is 0xFF)
            if(bits >= 0xFF00) {
                // computes: bits = hi_byte(bits) + (lo_byte(bits) << 1)
                bits += bits & 0xFF;
                
                // reflect unstuffed bit in number of extra bits
                extraCodeBitCount--;
            }
            
            // add newly loaded bits to code register and remember whether 
            // to unstuff first bit of next pair of bytes
            C += bits * unstuff;
            unstuff = nextUnstuff;
        }
    }
    
    /// Renormalizes probability register into expected range.
    __device__ void renormalize() {
        // Most significant zero bit count in A.
        // (Between 1 and 15, both inclusive.)
        int shift = __clz(A);
        
        // Do the biggest shift possible with remaining extra code bits.
        // (Between 0 and 15, both inclusive)
        const int firstShift = ::min(shift, extraCodeBitCount);
        A <<= firstShift;
        C <<= firstShift;
        extraCodeBitCount -= firstShift;
        shift -= firstShift;
        
        // Need to shift even more bits?
        if(shift) {
            // load 16 new code bits (or 15 bits if bit unstuffing appears) 
            read2CodeBytes();
            
            // finish the shift using newly loaded bits
            A <<= shift;
            C <<= shift;
            extraCodeBitCount -= shift;
        }
    }
    
public:
    /// Initializes new MQ decoder instance.
    /// @param state  reference to structure with initialized decoder state
    __device__ MQDecoder(CblkDecState & state, u8 * const buffer)
            : reader(buffer), state(state) { }
    
    /// Initializes decoder (populates code register with code bytes and aligns
    /// rest of codestream to two-byte boundary to be able to read two code 
    /// bytes at once.) Also initializes MPS/LPS probability.
    /// @param data  pointer to compressed data in GPU memory
    /// @param byteCount number of data bytes
    __device__ void reset(const u8 * const codeBegin, const int codeByteCount) {
        // all MPSs are initially set to 0 and almost all indices are 0 too...
        for(int i = 19; i--;) {
            state.indices[i] = 0;
        }
        
        // override special indices
        state.indices[ZC0] = 4 * 2;
        state.indices[RLC] = 3 * 2;
        state.indices[UNI] = 46 * 2;
        
        // count of bytes to be skipped for the pointer to be aligned 
        // to 2byte boundary
        const int alignBytes = 1 & (size_t)codeBegin;
        
        // reader reads 2 bytes at a time (so the pointer must be aligned)
        reader.init(codeBegin + alignBytes, codeByteCount - alignBytes);
        
        // possibly read the skipped byte (or initialize code to 0)
        if(alignBytes) {
            const u32 bits = *codeBegin;
            C = bits << 16;
            unstuff = bits == 0xFF ? 2 : 1; // 2 for next byte bit unstuffing
        } else {
            C = 0;
            unstuff = 1; // 1 == no unstuffing
        }
        
        // no extra code bits loaded yet
        extraCodeBitCount = 0;
        
        // load 2 more code bytes
        read2CodeBytes();
        
        // shift code bits to have exactly 15 loaded bits 
        // (and update extra code bit count accordingly)
        const int shift = 15 - 8 * alignBytes;  // either 7 or 15
        C <<= shift;
        extraCodeBitCount -= shift;

        // initialize interval register (in contrast with specification, 
        // this implementation puts 16 interval bits into upper half 
        // of 32bit register)
        A = 0x80000000;
    }
    
    /// Decodes one more decision in given context.
    __device__ u32 decode(const u32 context) {
        u32 idx = state.indices[context];
        
        const u32 packedUpdate = nQeShared[idx];
        const u32 qe = packedUpdate & 0xFFFF0000;
        
        A -= qe;
        
        const bool lps_exchange = C < qe;
        
        if(!lps_exchange) {
            C -= qe;
        }
        
        if(lps_exchange || A < 0x80000000) {
            // 1 if LPS, 0 if MPS
            u32 lps = A < qe ? 0x0 : 0x7;
            if(lps_exchange) {
                A = qe;
            } else {
                lps ^= 0x7;
            }
            state.indices[context] = 0x7F & (packedUpdate >> lps);
            
            idx ^= lps;
            
            renormalize();
        }
        
        return idx & 1;
    }
    
    /// Loads more code bytes into buffer in shared memory.
    /// At least 16 decisions can be decoded per one load.  TODO: or even more?
    __device__ void load() {
        reader.load();
    }
}; // end of class MQDecoder


template <typename T>
__device__ static T & getGrp(T * const states, const int stride, const int pixX, const int pixY) {
    return states[pixX * stride + (pixY >> 2)];
}


/// Returns "even context number 2 | xor bit"
template <int PIX>
__device__ inline u32 getSignCtx(const u32 lcSignif, const u32 rSignif) {
    // compile time constants
    enum {
        IDX_SIGN_SHIFT = 16 - PIX,
        IDX_SIGNIF_SHIFT = 3 - PIX,
        IDX_LC_BITS = 0x00A10A10 >> PIX,
        IDX_R_BITS = 0x00080080 >> PIX,
        IDX_MASK = 0x1FF,
    };
    
    // compose index to lookup table for sign coding
    const u32 scIdxBits = (lcSignif & IDX_LC_BITS) | (rSignif & IDX_R_BITS);
    const u32 scIdxSigns = scIdxBits >> IDX_SIGN_SHIFT;
    const u32 scIdxSigmas = scIdxBits >> IDX_SIGNIF_SHIFT;
    const u32 scIdx = (scIdxSigmas | scIdxSigns) & IDX_MASK;
    
    // get context and XOR bit from the lookup table
    return scLut[scIdx];
}


template <int PIX>
__device__ static void zc1(s32 & magn, u32 & lcSignif, const u32 rSignif,
                           const s32 bplnBit, MQDecoder & mqDecoder) {
    // compile time constants
    enum {
        PIX_SIGNIF_BIT = 1 << (10 - PIX),
        PIX_SIGN_BIT = 1 << (22 - PIX),
    };
    
    // set significance state
    lcSignif |= PIX_SIGNIF_BIT;
    magn |= bplnBit;
    
    // get sign context and decode the sign
    const int signCtxAndXor = getSignCtx<PIX>(lcSignif, rSignif);
    const u32 signDecision = mqDecoder.decode(signCtxAndXor & 0x1e);
    const int sign = 1 & (signDecision ^ signCtxAndXor);
    if(sign) {
        lcSignif |= PIX_SIGN_BIT;
    }
}


template <int PIX>
__device__ static void spp(s32 & magn, u32 & lcSignif, const u32 rSignif,
                           const u8 * const lutZC, const s32 bplnBit,
                           MQDecoder & mqDecoder) {
    // compile time constants
    enum {
        SHIFT = 3 - PIX,
        PIX_SIGNIF_BIT = 1 << (10 - PIX),
        PIX_SPP_BIT = 1 << (27 - PIX),
        LC_NEIGHB_BITS = 0xe38 >> PIX,  // neighborhood includes pixel itself
        R_NEIGHB_BITS = 0x1c0 >> PIX,
    };
    
    // compose neighborhood sigmas
    const u32 neighbSigmas = (lcSignif & LC_NEIGHB_BITS)
                           | (rSignif & R_NEIGHB_BITS);
                           
    // decode the pixel in this pass only if some neighboring pixel 
    // is significant and the pixel itself is not significant
    if(neighbSigmas && !(neighbSigmas & PIX_SIGNIF_BIT)) {
        // some significant neighbors => determine context to decode decision
        if(mqDecoder.decode(lutZC[neighbSigmas >> SHIFT])) {
            zc1<PIX>(magn, lcSignif, rSignif, bplnBit, mqDecoder);
        }
        
        // update the index of pixel's next bitplane 
        // (not to decode more pixel's bits in this bitplane)
        // (the bit is expected to be initialized to 0)
        lcSignif |= PIX_SPP_BIT;
    }
}


/// Composes packed flags of group and its vertical neighbors 
/// and returns it along with flags of the group.
/// @param statePtr   pointer to packed flags of the group:
///                       bit #:  15   12 11    8 7     4 3     0
///                              +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///                     content: | signs |refined|decoded| sigma |
///                              +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///                     pixel y:  0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
/// @tparam clearDecFlags true if all 4 'decoded' flags should be cleared
/// @return packed flags of all 4 pixels of the group and two vertical 
///         neighboring pixels
/// 
///  
///    bit #:  31 29 28   25 24   21 20       15 14        9 8         3 2   0
///           +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///  content: |  0  |refined|decoded|   signs   |     0     |   sigma   |  0  |   
///           +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///  pixel Y:        0 1 2 3 0 1 2 3-1 0 1 2 3 4            -1 0 1 2 3 4    
///  
__device__ static u32 flagsLoad(const u16 * const flagsPtr,
                                const bool clearDecFlags = false) {
    // load flags of the group and its upper and lower neighbors
    const u32 center = flagsPtr[0];
    const u32 up = flagsPtr[-1];
    const u32 down = flagsPtr[1];
    
    // mask which either cleans the 'decoded' flags or not
    const u32 state_mask = clearDecFlags ? 0x0f00 : 0x0ff0;
    
    // compose packed flags for 4 group pixels and their two vertical neighbors
    return ((0x1001 & up) << 8)
         + ((0xf00f & center) << 4)
         + ((state_mask & center) << 17)
         + (0x8008 & down);
}


/// Use previous center flags as new left flags and previous right flags 
/// as current flags.
/// @param lcFlagsOld  previous flags of central and left column
/// @param rFlagsOld   previous flags of right column (central one now)
/// @return new packed flags consiting of state of central column 
///         (previously right) and left column (previously central one)
__device__ static u32 updateFlags(const u32 lcFlagsOld, const u32 rFlagsOld) {
    return ((lcFlagsOld >> 6) & 0x0003f03f) + (rFlagsOld << 3);
}


/// Packs flags of central column back into format for flags of group.
/// @param lcFlags  packed flags in following format:
///    bit #:  31   28 27   24 23       18 17       12 11        6 5         0
///           +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///  content: |refined|decoded|cntrl signs|left signs |cntr sigmas|left sigmas|
///           +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///
///  pixel Y:  0 1 2 3 0 1 2 3-1 0 1 2 3 4-1 0 1 2 3 4-1 0 1 2 3 4-1 0 1 2 3 4
/// @return flags in format in which they are saved in group flags array:
///                                     bit #: 15   12 11    8 7     4 3     0
///                                           +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///                                  content: | signs |refined|decoded| sigmas|
///                                           +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///                                  pixel Y:  0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
/// 
__device__ static u32 lcFlagsToGroupFlags(const u32 lcFlags) {
    return ((lcFlags >> 20) & 0x0ff0) | ((lcFlags >> 7) & 0xf00f);
}


template <int PIX>
__device__ inline bool decodeZC(const u32 lcFlags,
                                const u32 rFlags,
                                MQDecoder & mqDecoder,
                                const u8 * const lutZC) {
    // compile time masks for getting index with neighborhood significancy
    const u32 IDX_SHIFT = 3 - PIX;
    const u32 LC_NEIGHB_BITS = 0xe38 >> PIX;  // neighbh. includes pixel itself
    const u32 R_NEIGHB_BITS = 0x1c0 >> PIX;
    
    // compose lookup table index from flags of neighboring pixels
    const int lutIdx = (lcFlags & LC_NEIGHB_BITS) | (rFlags & R_NEIGHB_BITS);
    
    // use lookup table to determine MQ decoder context
    const int context = lutZC[lutIdx >> IDX_SHIFT];
    
    // decode the decision bit in right context
    return mqDecoder.decode(context);
}



/// Gets true if pixel should be coded in cleanup pass AND becomes significant.
template <int PIX>
__device__ inline bool codeInCUP(const u32 lcFlags,
                                 const u32 rFlags,
                                 MQDecoder & mqDecoder,
                                 const u8 * const lutZC) {
    // compile time constants: SPP and significancy flag bits
    const u32 CODED_BEFORE_CUP_MASK = 0x08000000 >> PIX;
    
    // return false if coded in previous passes, decision otherwise
    return (CODED_BEFORE_CUP_MASK & lcFlags)
        ? false : decodeZC<PIX>(lcFlags, rFlags, mqDecoder, lutZC);
}


template <bool FIRST_PASS>
__device__ static void cleanupPass(int4 * magnitudesPtr,
                                   u16 * flagsPtr,
                                   const s32 bplnBit,
                                   const unsigned int strideX,
                                   const unsigned int cblkSizeX,
                                   MQDecoder & mqDecoder,
                                   const u8 * const lutZC,
                                   const bool last1,
                                   const bool last2,
                                   const bool last3,
                                   const int completeRowCount) {
    // compile time constants
    const u32 RLC_MASK = 0x00000FFF;
    const u32 ALL_DECODED_BITS = 0x0F000000;

    // for each row (in context modeller scan order)
    for(int remainingRowCount = completeRowCount; remainingRowCount--;) {
        // pointer to flags of the group
        u16 * groupFlagsPtr = flagsPtr;
        
        // load central and right column (will be transfomed to left 
        // and central columns)
        u32 lcFlags = flagsLoad(groupFlagsPtr - strideX, false) << 3;
        u32 rFlags = flagsLoad(groupFlagsPtr, false);
        
        // for each column in this row 
        for(int remainingColumnCount = cblkSizeX; remainingColumnCount--;) {
            // Load more MQ decoder input
            mqDecoder.load();
            
            // convert old right and central columns to central and left
            lcFlags = updateFlags(lcFlags, rFlags);
            
            // advance flags pointer to next column to load new right column flags
            u16 * const nextGroupFlagsPtr = groupFlagsPtr + strideX;
            rFlags = flagsLoad(nextGroupFlagsPtr, false);
            
            // is there any pixel to be decoded in CUP?
            if(FIRST_PASS || ALL_DECODED_BITS != (ALL_DECODED_BITS & lcFlags)) {
                // flags indicating that some decoding stages are skipped
                bool decodeSign = false;  // true to decode sign of next pixel
                bool decodePixel0 = true; // false to skip pixel #0
                bool decodePixel1 = true; // false to skip pixel #1
                bool decodePixel2 = true; // false to skip pixel #2
                
                // test whether run lenght coding should be used
                if(RLC_MASK & (lcFlags | rFlags)) {
                    // test, whether sign of first pixel should be decoded
                    decodeSign = codeInCUP<0>(lcFlags, rFlags, mqDecoder, lutZC);
                } else {
                    // test, whether any pixel is significant
                    decodeSign = mqDecoder.decode(RLC);
                    
                    // get index of first significant pixel to be decoded (defaults
                    // to "begin with 3rd pixel, which is however not significant, 
                    // so skip it as well"
                    int firstSignifPixIdx = 3; 
                    if(decodeSign) {
                        // get 2bit index of first significant pixel
                        const int uniMsb = mqDecoder.decode(UNI);
                        const int uniLsb = mqDecoder.decode(UNI);
                        firstSignifPixIdx = uniMsb * 2 + uniLsb;
                    }
                    
                    // disable encoding of right pixels
                    switch(firstSignifPixIdx) {
                        case 3: decodePixel2 = false; // falls through to other pixels
                        case 2: decodePixel1 = false;
                        case 1: decodePixel0 = false;
                    }
                }
                
                // load more MQ decoder input 
                mqDecoder.load();
                
                // load magnitudes of whole group
                int4 magn = FIRST_PASS ? make_int4(0, 0, 0, 0) : *magnitudesPtr;
                
                // decode first pixel sign (if significant and not skipped by RLC)
                if(decodePixel0) {
                    // decode the sign
                    if(decodeSign) {
                        zc1<0>(magn.x, lcFlags, rFlags, bplnBit, mqDecoder);
                    }
                    
                    // should the sign of next pixel be deocded?
                    decodeSign = codeInCUP<1>(lcFlags, rFlags, mqDecoder, lutZC);
                }
                
                // decode second pixel (if significant and not skipped by RLC)
                if(decodePixel1) {
                    // decode the sign
                    if(decodeSign) {
                        zc1<1>(magn.y, lcFlags, rFlags, bplnBit, mqDecoder);
                    }
                    
                    // should the sign of next pixel be deocded?
                    decodeSign = codeInCUP<2>(lcFlags, rFlags, mqDecoder, lutZC);
                }
                
                // decode next pixel sign (if significant and not skipped by RLC)
                if(decodePixel2) {
                    // decode the sign
                    if(decodeSign) {
                        zc1<2>(magn.z, lcFlags, rFlags, bplnBit, mqDecoder);
                    }
                    
                    // should the sign of next pixel be deocded?
                    decodeSign = codeInCUP<3>(lcFlags, rFlags, mqDecoder, lutZC);
                }
                
                // possibly decode the sign of last pixel
                if(decodeSign) {
                    zc1<3>(magn.w, lcFlags, rFlags, bplnBit, mqDecoder);
                }
                
                // write updated magnitudes and flags back
                *magnitudesPtr = magn;
                *groupFlagsPtr = lcFlagsToGroupFlags(lcFlags);
            }
            
            // advance magnitude and flags pointers to next group
            magnitudesPtr++;
            groupFlagsPtr = nextGroupFlagsPtr;
        }
                
        // advance row flags
        flagsPtr++;  // column based
    }
    
    // last incomplete row
    if(last1) {
        // load central and right column (will be transfomed to left 
        // and central columns)
        u32 lcFlags = flagsLoad(flagsPtr - strideX, false) << 3;
        u32 rFlags = flagsLoad(flagsPtr, false);
        
        // for each column in last row 
        for(int remainingColumnCount = cblkSizeX; remainingColumnCount--;) {
            // Load more MQ decoder input
            mqDecoder.load();
            
            // convert old right and central columns to central and left
            lcFlags = updateFlags(lcFlags, rFlags);
            
            // advance flags pointer to next column to load new right column flags
            u16 * const nextGroupFlagsPtr = flagsPtr + strideX;
            rFlags = flagsLoad(nextGroupFlagsPtr, false);
            
            // is there any pixel to be decoded in CUP in this group?
            if(FIRST_PASS || ALL_DECODED_BITS != (ALL_DECODED_BITS & lcFlags)) {
                // load magnitudes of all 4 pixels
                int4 magn = FIRST_PASS ? make_int4(0, 0, 0, 0) : *magnitudesPtr;
                
                // get significancy of first pixel and possibly decode its sign
                if(codeInCUP<0>(lcFlags, rFlags, mqDecoder, lutZC)) {
                    zc1<0>(magn.x, lcFlags, rFlags, bplnBit, mqDecoder);
                }
                
                // decode second and third pixel only if not out of bounds
                if(last2) {
                    // decode sign of second pixel if significant
                    if(codeInCUP<1>(lcFlags, rFlags, mqDecoder, lutZC)) {
                        zc1<1>(magn.y, lcFlags, rFlags, bplnBit, mqDecoder);
                    }
                    
                    // decode sign of third pixel if not out of bounds and if significant
                    if(last3 && codeInCUP<2>(lcFlags, rFlags, mqDecoder, lutZC)) {
                        zc1<2>(magn.z, lcFlags, rFlags, bplnBit, mqDecoder);
                    }
                }
                
                // write updated magnitudes and flags back
                *magnitudesPtr = magn;
                *flagsPtr = lcFlagsToGroupFlags(lcFlags);
            }
            
            // advance magnitude and flags pointers
            magnitudesPtr++;
            flagsPtr = nextGroupFlagsPtr;
        }
    }
}



template <int PIX>
__device__ static void magnitudeRefinement(u32 & lcFlags,
                                           const u32 rFlags, 
                                           MQDecoder & mqDecoder,
                                           s32 & magnitude,
                                           const u32 bplnBit) {
    // compile time constants
    const u32 CODE_IN_MRP_BITS = 0x08000400 >> PIX; // dec and signif flag bits
    const u32 CODE_IN_MRP_VALUE = 0x00000400 >> PIX;
    const u32 REFINED_BIT = 0x80000000 >> PIX;      // 1 if refined
    const u32 LC_NEIGHB_BITS = 0x00000A38 >> PIX;
    const u32 R_NEIGHB_BITS = 0x000001C0 >> PIX;
    const u32 DECODED_BIT = 0x08000000 >> PIX;
    
    // decode the pixel in this pass only if it has not been decoded
    // in this bitplane and is significant
    if((lcFlags & CODE_IN_MRP_BITS) == CODE_IN_MRP_VALUE) {
        // get context
        int context = MRC2; // defaults to "already refined"
        if((lcFlags & REFINED_BIT) == 0) {
            // not refined => decide using neighbor significancy
            context = (LC_NEIGHB_BITS & lcFlags) | (R_NEIGHB_BITS & rFlags) ? MRC1 : MRC0;
        }
        
        // decode the decision in the right context
        if(mqDecoder.decode(context)) {
            magnitude |= bplnBit;
        }
        
        // mark the pixel as refined and decoded in current bitplane
        lcFlags |= (REFINED_BIT | DECODED_BIT);
    }
}



__device__ static void magnitudeRefinementPass(const unsigned int cblkSizeX,
                                               const unsigned int strideX,
                                               const bool last1,
                                               const bool last2,
                                               const bool last3,
                                               const int completeRowCount,
                                               const u32 bplnBit,
                                               MQDecoder & mqDecoder,
                                               int4 * magnitudesPtr,
                                               u16 * flagsPtr) {
    // magnitude refinement pass:
    // for each row (in context modeller scan order)
    for(int remainingRowCount = completeRowCount; remainingRowCount--; ) {
        // pointer to flags of the group
        u16 * groupFlagsPtr = flagsPtr;
        
        // load central and right column (will be transfomed to left 
        // and central columns)
        u32 lcFlags = flagsLoad(groupFlagsPtr - strideX, false) << 3;
        u32 rFlags = flagsLoad(groupFlagsPtr, false);
        
        // for each column in this row 
        for(int remainingColumnCount = cblkSizeX; remainingColumnCount--; ) {
            // Load more MQ decoder input
            mqDecoder.load();
            
            // convert old right and central columns to central and left
            lcFlags = updateFlags(lcFlags, rFlags);
            
            // advance flags pointer to next column to load new right column flags
            u16 * const nextGroupFlagsPtr = groupFlagsPtr + strideX;
            rFlags = flagsLoad(nextGroupFlagsPtr, false);
            
            // load magnitudes of the group 
            int4 groupMagn = *magnitudesPtr;
                
            // run magnitude refinement for all 4 pixel sof the group
            magnitudeRefinement<0>(lcFlags, rFlags, mqDecoder, groupMagn.x, bplnBit);
            magnitudeRefinement<1>(lcFlags, rFlags, mqDecoder, groupMagn.y, bplnBit);
            magnitudeRefinement<2>(lcFlags, rFlags, mqDecoder, groupMagn.z, bplnBit);
            magnitudeRefinement<3>(lcFlags, rFlags, mqDecoder, groupMagn.w, bplnBit);
            
            // write updated state back and advance to next group
            *groupFlagsPtr = lcFlagsToGroupFlags(lcFlags);
            groupFlagsPtr = nextGroupFlagsPtr;
            
            // write magnitudes back, updating the pointer to next group
            *(magnitudesPtr++) = groupMagn;
        }
        
        // advance row flags
        flagsPtr++;  // column based
    }
    
    // last incomplete row
    if(last1) {
        // load central and right column (will be transfomed to left 
        // and central columns)
        u32 lcFlags = flagsLoad(flagsPtr - strideX, false) << 3;
        u32 rFlags = flagsLoad(flagsPtr, false);
        
        // for each column in this row 
        for(int remainingColumnCount = cblkSizeX; remainingColumnCount--; ) {
            // Load more MQ decoder input
            mqDecoder.load();
            
            // convert old right and central columns to central and left
            lcFlags = updateFlags(lcFlags, rFlags);
            
            // advance flags pointer to next column to load new right column flags
            u16 * const nextGroupFlagsPtr = flagsPtr + strideX;
            rFlags = flagsLoad(nextGroupFlagsPtr, false);
            
            // load magnitudes of the group 
            int4 groupMagn = *magnitudesPtr;
                
            // run magnitude refinement for all 4 pixel sof the group
            magnitudeRefinement<0>(lcFlags, rFlags, mqDecoder, groupMagn.x, bplnBit);
            if(last2) {
                magnitudeRefinement<1>(lcFlags, rFlags, mqDecoder, groupMagn.y, bplnBit);
                if(last3) {
                    magnitudeRefinement<2>(lcFlags, rFlags, mqDecoder, groupMagn.z, bplnBit);
                }
            }
            
            // write updated state back and advance to next group
            *flagsPtr = lcFlagsToGroupFlags(lcFlags);
            flagsPtr = nextGroupFlagsPtr;
            
            // write magnitudes back, updating magnitude pointer for next group
            *(magnitudesPtr++) = groupMagn;
        }
    }
}



__device__ static void significancePropagationPass(const unsigned int cblkSizeX,
                                                   const unsigned int strideX,
                                                   const bool last1,
                                                   const bool last2,
                                                   const bool last3,
                                                   const int completeRowCount,
                                                   const u32 bplnBit,
                                                   MQDecoder & mqDecoder,
                                                   int4 * magnitudesPtr,
                                                   u16 * flagsPtr,
                                                   const u8 * const lutZC) {
    // for each complete row
    for(int remainingRowCount = completeRowCount; remainingRowCount--; ) {
        // pointer to flags of the group
        u16 * groupFlagsPtr = flagsPtr;
        
        // load central and right column (will be transfomed to left 
        // and central columns)
        u32 lcFlags = flagsLoad(groupFlagsPtr - strideX, true) << 3;
        u32 rFlags = flagsLoad(groupFlagsPtr, true);
        
        // for each column in this row 
        for(int remainingColumnCount = cblkSizeX; remainingColumnCount--; ) {
            // Load more MQ decoder input
            mqDecoder.load();
            
            // convert old right and central columns to central and left
            lcFlags = updateFlags(lcFlags, rFlags);
            
            // advance flags pointer to next column to load new right column flags
            u16 * const nextGroupFlagsPtr = groupFlagsPtr + strideX;
            rFlags = flagsLoad(nextGroupFlagsPtr, true);
            
            // load magnitudes of all 4 pixels
            int4 magn = *magnitudesPtr;
            
            // for each pixel in the row
            spp<0>(magn.x, lcFlags, rFlags, lutZC, bplnBit, mqDecoder);
            spp<1>(magn.y, lcFlags, rFlags, lutZC, bplnBit, mqDecoder);
            spp<2>(magn.z, lcFlags, rFlags, lutZC, bplnBit, mqDecoder);
            spp<3>(magn.w, lcFlags, rFlags, lutZC, bplnBit, mqDecoder);
            
            // write magnitudes back and advance magnitude pointer
            *(magnitudesPtr++) = magn;
            
            // write updated state back and advance to next group
            *groupFlagsPtr = lcFlagsToGroupFlags(lcFlags);
            groupFlagsPtr = nextGroupFlagsPtr;
        }
        
        // advance pointer to next row of flags
        flagsPtr++;  // column based
    }
    
    // SPP finalization: for each column in this row 
    if(last1) {
        // load central and right column (will be transfomed to left 
        // and central columns)
        u32 lcFlags = flagsLoad(flagsPtr - strideX, true) << 3;
        u32 rFlags = flagsLoad(flagsPtr, true);
        
        for(int remainingColumnCount = cblkSizeX; remainingColumnCount--; ) {
            // Load more MQ decoder input
            mqDecoder.load();
            
            // convert old right and central columns to central and left
            lcFlags = updateFlags(lcFlags, rFlags);
            
            // advance flags pointer to next column to load new right column flags
            u16 * const nextGroupFlagsPtr = flagsPtr + strideX;
            rFlags = flagsLoad(nextGroupFlagsPtr, true);
            
            // load magnitudes of all 4 pixels
            int4 magn = *magnitudesPtr;
            
            // for each pixel in the row
            spp<0>(magn.x, lcFlags, rFlags, lutZC, bplnBit, mqDecoder);
            if(last2) {
                spp<1>(magn.y, lcFlags, rFlags, lutZC, bplnBit, mqDecoder);
                if(last3) {
                    spp<2>(magn.z, lcFlags, rFlags, lutZC, bplnBit, mqDecoder);
                }
            }
            
            // write magnitudes back and advance magnitude pointer
            *(magnitudesPtr++) = magn;
            
            // write updated state back and advance to next group
            *flagsPtr = lcFlagsToGroupFlags(lcFlags);
            flagsPtr = nextGroupFlagsPtr;
        }
    }
}


/// Checks number of remaining passes encoded in current codestream segment, 
/// possibly loads next segment into the MQ decoder and decreases 
/// the remaining segment count. 
/// @param segs  pointer to buffer with all segments
/// @param code  pointer to buffer with all codestreams
/// @param segPtr  reference to pointer to into about current codestream
/// @param remainingSegPassCount  ref to current segment's remaining pass count
/// @param mqDecoder  ref to MQ decoder instance for the codeblock
/// @return false if all passes from all segments were decoded, true otherwise
__device__ static bool mqReload(const Seg * const segs,
                                const u8 * const code,
                                const Seg * & segPtr,
                                int & remainingSegPassCount,
                                MQDecoder & mqDecoder) {
    // check remaining pass count 
    // (possibly load next segment with more encoded passes)
    if(remainingSegPassCount <= 0) {
        // update segment info pointer
        const int nextSegIdx = segPtr->nextSegIdx;
        if(-1 == nextSegIdx) {
            return false;
        }
        segPtr = segs + nextSegIdx;
        
        // reset MQ decoder
        mqDecoder.reset(code + segPtr->codeByteOffset, segPtr->codeByteCount);
        
        // set new remaining pass count
        remainingSegPassCount = segPtr->passCount;
    }
    
    // update remaining pass count (for next pass decoding)
    remainingSegPassCount--;
    
    // indicate that there are more passes to be decoded
    return true;
}



/// EBCOT kernel.
/// @tparam TCOUNT  number of threads per threadblock
/// @param params  all parameters in one structure
template <int TCOUNT>
__launch_bounds__(TCOUNT, 18 / (TCOUNT / 32))
__global__ static void ebcotDecodingKernel(const EbcotKernelParams params) {
    // all threads load Qe from global memory to shared memory
    for(int i = threadIdx.x; i < 47 * 2; i += TCOUNT) {
        nQeShared[i] = nQe[i];
    }
    __syncthreads();
    
    const int globalThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalThreadIdx >= params.cblkCount) {
        // this thread has no codeblock (is only a padding in last threadblock)
        return;
    }
    
    // get index of processed codeblock
    const int cblkIdx = params.cblkPermuation[globalThreadIdx];
//     const int cblkIdx = globalThreadIdx;
    
    // info about processed codeblock and the band
    const Cblk * const cblk = params.cblks + cblkIdx;
    const Band * const band = params.bands + cblk->bandIdx;
    const unsigned int cblkSizeX = cblk->size.x;
    const unsigned int cblkSizeY = cblk->size.y;
    const Orientation orientation = band->orient;
    const unsigned int cblkBplnCount = cblk->bplnCount;
    int nextSegIdx = cblk->firstSegIdx;
    
    // if codeblock has no segments, skip it
    if(nextSegIdx == -1) {
        return;
    }
    
    // buffers for MQ decoder input (for all threads)
    __shared__ int4 allSrcBuffers[TCOUNT * 2]; // 32 bytes per thread
    
    // this thread's MQ decoder source buffer
    u8 * const srcBuffer = ((u8*)allSrcBuffers) + 32 * threadIdx.x;
    
    // MQ decoder states (for all threads)
    __shared__ CblkDecState indices[TCOUNT];
    
    // initialize MQ decoder
    MQDecoder mqDecoder(indices[threadIdx.x], srcBuffer);
    
    // load first segment into MQ decoder and remember number of passes 
    // to be decoded from the segment (before loading next segment)
    const Seg * seg = params.segs + cblk->firstSegIdx;
    int passCount = seg->passCount;
    mqDecoder.reset(params.cstream + seg->codeByteOffset, seg->codeByteCount);
    
    // difference between indices of two horizontally neighboring group states
    const int strideX = (cblk->stdSize.y >> 2) + 1;
    
    // info about magnitudes of all pixel of thread's codeblock
    int4 magnitudes[1024];
    
    // maximal needed count of groups state structures in shared memory
    enum { MAX_GRP_FLAGS_8_COUNT = 2064 / 8 }; 
    
    // packed flags of all groups
    int4 allGroupFlags[MAX_GRP_FLAGS_8_COUNT];
    
    // initialize flags of all groups
    for(int flags8Idx = (1 + strideX * (cblkSizeX + 2) + 7) >> 3; flags8Idx--;) {
        allGroupFlags[flags8Idx] = make_int4(0, 0, 0, 0);
    }
    
    // TODO: try to change group flags layout to row based (now, it is column based)
    
    // select the right lookup table according to the band orientation
    const u8 * const lutZC = orientation == ORI_HH
            ? hhZcLut : (orientation == ORI_HL ? hlZcLut : llZcLut);
    
    // pointers to group flags with pre-added offset to first group (skips boundary)
    u16 * const flags = (u16*)allGroupFlags + 1 + strideX;
    
    // size and coordinates of last row
    const int lastRowHeight = cblkSizeY & 3;
    const bool last1 = lastRowHeight >= 1;
    const bool last2 = lastRowHeight >= 2;
    const bool last3 = lastRowHeight >= 3;
    
    // number of complete rows
    const int completeRowCount = cblkSizeY >> 2;
    
    // first cleanup pass (contains magnitude initialization)
    cleanupPass<true>(magnitudes, flags, 1 << (cblkBplnCount - 1),
                      strideX, cblkSizeX, mqDecoder, lutZC,
                      last1, last2, last3, completeRowCount);
    
    // one pass was decoded from current codestream segment:
    passCount--;
    
    // for all bitplanes (starting with the most significant one)
    for(int bplnIdx = cblkBplnCount - 2; bplnIdx >= 0; bplnIdx--) {
        // magnitude bit for current bitplane
        const s32 bplnBit = 1 << bplnIdx;
        
        // possibly reset MQ decoder (if no more passes are encodied in current
        // codestream segment and deocde SPP)
        if(!mqReload(params.segs, params.cstream, seg, passCount, mqDecoder)) {
            break; // stop if there are no more segments to be decoded
        }
        significancePropagationPass(cblkSizeX, strideX, last1, last2, last3, 
                completeRowCount, bplnBit, mqDecoder, magnitudes, flags, lutZC);
        
        // possibluy reload and decode MRP
        if(!mqReload(params.segs, params.cstream, seg, passCount, mqDecoder)) {
            break; // stop if there are no more segments to be decoded
        }
        magnitudeRefinementPass(cblkSizeX, strideX, last1, last2, last3,
                completeRowCount, bplnBit, mqDecoder, magnitudes, flags);
        
        // possibly reload and decode samples in CUP
        if(!mqReload(params.segs, params.cstream, seg, passCount, mqDecoder)) {
            break; // stop if there are no more segments to be decoded
        }
        cleanupPass<false>(magnitudes, flags, bplnBit, strideX, cblkSizeX,
                mqDecoder, lutZC, last1, last2, last3, completeRowCount);
    }
    
    // copy whole state into global memory
    int4 * dest = (int4*)((u8*)params.output + cblk->ebcotTemp);
    dest -= 15 & (size_t)dest;
    
    // copy group magnitudes first
    const int4 * magnSrc = magnitudes;
    for(int grpIdx = ((cblkSizeY + 3) >> 2) * cblkSizeX; grpIdx--;) {
        *(dest++) = *(magnSrc++);
    }
    
    // copying 8 group flags at once (16 bytes)
    const unsigned int grpFlagsCount = 1 + strideX * (cblkSizeX + 2);
    const unsigned int grpFlagsCopyCount = (grpFlagsCount + 7) >> 3;
    const int4 * flagsSrc = allGroupFlags;
    for(int copyIdx = grpFlagsCopyCount; copyIdx--; ) {
        *(dest++) = *(flagsSrc++);
    }
}



// (5+4)bit index, with 32x16 fields. Upper (5bit) part of the index contains 
// count of bitplanes which were not decoded at all (not counting partially 
// decoded bitplanes). Lower (4bit) part represents combination of pixels, 
// where some were decoded in the last bitplane and others were not decoded 
// in the bitplane. (Bit '1' reresents pixel decoded in the last bitplane and 
// bit '0' represents pixel NOT decoded in the bitplane).
// Each field of the array then contains four positive halfway values for 
// reconstruction of correpsonding combination of decoded/not-decoded pixels.
// E. g. index 50 (00011 0010 binary) represents reconstruction halfway values 
// for 4 pixels, where third pixel is missing 3 least significant bits and
// other three pixels are missing 4 least significant bits.
// There are two versions: for integers and for floats:
__constant__ __device__ static int4 reconstructionHalvesInt[32 * 16];
__constant__ __device__ static float4 reconstructionHalvesFloat[32 * 16];




template <typename T>
__device__ static T
reconstructPixel(int magnitude, const T half, const int idx, const u32 flags) {
    // add reconstruction halfway value if not zero
    T out = (T)magnitude;
    if(magnitude) {
        out += half;
//         if(half != 0) {
//             printf("Half: %f\n", (float)half);
//         }
    }
    
    // possibly apply sign and return
    if(flags & (0x8000 >> idx)) {
        out = -out;
    }
    return out;
}



/// Specialized for floats and for integers.
template <typename TYPE>
__device__ static void
reconstructGroup
(
        TYPE * &outPtr,
        const unsigned int outStride,
        const u32 flags,
        int4 magn,
        const unsigned int halfTableBaseOffset,
        const float stepsize,
        const bool save2,
        const bool save3,
        const bool save4
);



template <>
__device__ static void
reconstructGroup<float>
(
        float * &outPtr,
        const unsigned int outStride,
        const u32 flags,
        int4 magn,
        const unsigned int halfTableBaseOffset,
        const float stepsize,
        const bool save2,
        const bool save3,
        const bool save4
) {
    // load halfway values
    const unsigned int halfIdx = halfTableBaseOffset | (0xF & (flags >> 4));
    const float4 halves = reconstructionHalvesFloat[halfIdx];
    
    // reconstruct all 4 pixels
    const float outX = stepsize * reconstructPixel(magn.x, halves.x, 0, flags);
    const float outY = stepsize * reconstructPixel(magn.y, halves.y, 1, flags);
    const float outZ = stepsize * reconstructPixel(magn.z, halves.z, 2, flags);
    const float outW = stepsize * reconstructPixel(magn.w, halves.w, 3, flags);    
    // there is always at least 1 sample to save => save it and advance pointer
    *outPtr = outX;
    outPtr += outStride;
    
    // possibly reconstruct other pixels
    if(save2) {
        *outPtr = outY;
        outPtr += outStride;
        if(save3) {
            *outPtr = outZ;
            outPtr += outStride;
            if(save4) {
                *outPtr = outW;
                outPtr += outStride;
            }
        }
    }
}


template <>
__device__ static void
reconstructGroup<int>
(
        int * &outPtr,
        const unsigned int outStride,
        const u32 flags,
        int4 magn,
        const unsigned int halfTableBaseOffset,
        const float,
        const bool save2,
        const bool save3,
        const bool save4
) {
    // load integer halfway values
    const unsigned int halfIdx = halfTableBaseOffset | (0xF & (flags >> 4));
    const int4 halves = reconstructionHalvesInt[halfIdx];
    
    // reconstruct all 4 pixels
    const int outX = reconstructPixel(magn.x, halves.x, 0, flags);
    const int outY = reconstructPixel(magn.y, halves.y, 1, flags);
    const int outZ = reconstructPixel(magn.z, halves.z, 2, flags);
    const int outW = reconstructPixel(magn.w, halves.w, 3, flags);
    
    // there is always at least 1 sample to save => save it and advance pointer
    *outPtr = outX;
    outPtr += outStride;
    
    // possibly save other samples
    if(save2) {
        *outPtr = outY;
        outPtr += outStride;
        if(save3) {
            *outPtr = outZ;
            outPtr += outStride;
            if(save4) {
                *outPtr = outW;
                outPtr += outStride;
            }
        }
    }
}




template <typename TYPE>
__device__ static void reconstructAll
(
        const XY cblkSize,
        TYPE * outPtr,
        const int4 * magnitudesPtr,
        const u16 * const flagsBasePtr,
        const unsigned int flagsStrideX,
        const unsigned int outStride,
        const unsigned int halfTableBaseOffset,
        const float stepsize,
        const bool last1,
        const bool last2,
        const bool last3
) {
    // pre-add thread index to output pointer and to source magnitude pointer
    magnitudesPtr += threadIdx.x;
    outPtr += threadIdx.x;
    
    // each codeblock is processed by one warp - process all 32sample wide 
    // columns (each thread in warp taking care of single 1pixel column 
    // in each iteration):
    for(int x = threadIdx.x; x < cblkSize.x; x += blockDim.x) {
        // pointer to flags of first 4 thread's pixels
        const u16 * flagsPtr = &getGrp(flagsBasePtr, flagsStrideX, x, 0); 
        
        // run through all rows of the column (process 4row strips at once)
        for(int stripIdx = cblkSize.y >> 2; stripIdx--; ) {
            // load magnitudes and flags for 4 pixels and reconstruct 
            // the pixels, advancing the output pointer
            reconstructGroup(outPtr, outStride, *flagsPtr, *magnitudesPtr, 
                             halfTableBaseOffset, stepsize, true, true, true);
            
            // advance both source pointers (output pointer is being advanced 
            // in the reconstruction function)
            flagsPtr++; // flags buffer is transposed
            magnitudesPtr += cblkSize.x;
        }
        
        // last incomplete strip (less than 4 rows)
        if(last1) {
            // load magnitudes and flags and reconstruct
            reconstructGroup(outPtr, outStride, *flagsPtr, *magnitudesPtr, 
                             halfTableBaseOffset, stepsize, last2, last3, false);
        }
        
        // advance pointers to next column
        magnitudesPtr += blockDim.x;
        magnitudesPtr -= (cblkSize.y >> 2) * cblkSize.x;
        outPtr += blockDim.x;
        outPtr -= cblkSize.y * outStride;
    }
}


/// Codeblock zeroing (if there are no decoded coefficients)
template <typename T>
__device__ static void cblkClear(const XY & cblkSize,
                                 T * const out,
                                 const int outStride) {
    for(int y = cblkSize.y; y--; ) {
        for(int x = threadIdx.x; x < cblkSize.x; x += blockDim.x) {
            out[x + y * outStride] = (T)0;
        }
    }
}


// TODO: use template to specify actual thread count
__launch_bounds__(256, 4)
__global__ static void ebcotReconstructionKernel
(
        const Cblk * const cblks,
        const Band * const bands,
        const int cblkCount,
        const void * const src,
        void * out
) {
    // each warp reconstructs coefficients of one codeblock 
    // => get codeblock's index and check that it is not out of range
    const int cblkIdx = threadIdx.y + blockDim.y * blockIdx.x;
    if(cblkIdx >= cblkCount) {
        return;
    }
    
    // get all needed info about the codeblock
    const Cblk * const cblk = cblks + cblkIdx;
    const XY cblkPos = cblk->pos;
    const XY cblkSize = cblk->size;
    const XY cblkStdSize = cblk->stdSize;
    const u32 srcOffset = cblk->ebcotTemp;
    const int decodedPassCount = cblk->passCount;
    const int bplnCount = cblk->bplnCount;
    
    // get info about codeblock's band:
    const Band * const band = bands + cblk->bandIdx;
    const int outStride = band->outPixStride;
    const int bandOffset = band->outPixOffset;
    const bool reversible = band->reversible;
    const float stepsize = reversible ? 1.0f : band->stepsize;
    
    // top-left output pixel pointer (untyped - may be float or int)
    out = (void*)((u32*)out + bandOffset + cblkPos.x + cblkPos.y * outStride);
    
    // only clear the codeblock if there are no passes to be reconstructed
    if(0 == decodedPassCount) {
        if(reversible) {
            cblkClear(cblkSize, (int*)out, outStride);
        } else {
            cblkClear(cblkSize, (float*)out, outStride);
        }
        return;
    }
    
    // compose base offset for halfway reconstruction values table
    const unsigned int passCount = ::max(0, 3 * bplnCount - 2);
    const unsigned int missingPassCount = ::max(0, passCount - decodedPassCount);
    const unsigned int missingBplnCount = missingPassCount / 3;  // TODO: try mul_hi 0x55555556
    unsigned int halfTableBaseOffset = missingBplnCount * 16;
//     if(missingPassCount) {
//         printf("Missing pass count: %d\n", missingPassCount);
//     }
    if(missingPassCount - missingBplnCount * 3 == 0) { 
        // if last pass was CUP, treat all pixels as decoded in last bitplane,
        // even if their "decoded" bits are not set ("decoded" bits are not set 
        // in CUP)
        halfTableBaseOffset |= 0xF;
    }
    
    // counts and offsets of grouped stuff in the codeblock
    const unsigned int magnGroupCount = ((cblkSize.y + 3) >> 2) * cblkSize.x;
    const unsigned int flagsStrideX = (cblkStdSize.y >> 2) + 1;
    
    // pointers to source magnitudes and flags
    const int4 * const magnitudes = (const int4*)((const u8*)src + srcOffset);
    const u16 * const flags = ((const u16*)(magnitudes + magnGroupCount))
                            + flagsStrideX + 1; // skips dummy boundary flags
    
    // flags indicating whether there are at least N samples in last 
    // incomplete row
    const int lastStripSize = cblkSize.y & 3;
    const bool last1 = lastStripSize > 0;
    const bool last2 = lastStripSize > 1;
    const bool last3 = lastStripSize > 2;
    
    // use either float or integer version, depending on type of transforms
    if(reversible) {
        reconstructAll(cblkSize, (int*)out, magnitudes, flags, flagsStrideX,
                       outStride, halfTableBaseOffset, stepsize, last1, last2, 
                       last3);
    } else {
        reconstructAll(cblkSize, (float*)out, magnitudes, flags, flagsStrideX,
                       outStride, halfTableBaseOffset, stepsize, last1, last2, 
                       last3);
    }
}



/// Determines zero coding context according to sigmas of neighboring pixels.
/// (Neighbor sigmas are in clockwise order, starting with the top neighbor, 
/// with its sigma in the lsb of the parameter.)
static u8 zcContext(const int sigmas, const Orientation & ori) {
    if(sigmas == 0) {
        return ZC0;
    }
    
    int diagonalSignif = 0;
    int horizontalSignif = 0;
    int verticalSignif = 0;
    
    verticalSignif += (sigmas >> 8) & 1;
    verticalSignif += (sigmas >> 6) & 1;
    diagonalSignif += (sigmas >> 5) & 1;
    horizontalSignif += (sigmas >> 4) & 1;
    diagonalSignif += (sigmas >> 3) & 1;
    diagonalSignif += (sigmas >> 2) & 1;
    horizontalSignif += (sigmas >> 1) & 1;
    diagonalSignif += (sigmas >> 0) & 1;
    
    // separate case for HH
    if(ori == ORI_HH) {
        const int otherSignif = horizontalSignif + verticalSignif;
        if(diagonalSignif >= 3) return ZC8;
        if(diagonalSignif == 2) return otherSignif ? ZC7 : ZC6;
        if(otherSignif == 0) return ZC3;
        if(otherSignif == 1) return diagonalSignif ? ZC4 : ZC1;
        return diagonalSignif ? ZC5 : ZC2;
    } 
    
    // swap horizontal and vertical for HL, to use case for LH
    if (ori == ORI_HL) {
        const int temp = horizontalSignif;
        horizontalSignif = verticalSignif;
        verticalSignif = temp;
    }
    
    // case for LL or LH:
    if(horizontalSignif == 2) return ZC8;
    if(horizontalSignif == 1) {
        if(verticalSignif) return ZC7;
        return diagonalSignif ? ZC6 : ZC5;
    }
    if(verticalSignif == 2) return ZC4;
    if(verticalSignif == 1) return ZC3;
    if(diagonalSignif == 1) return ZC1;
    return ZC2;
}



static u8 scContext(const int index) {
    enum {
        TOP_SIGNIF_BIT = 0x100,
        BOTTOM_SIGNIF_BIT = 0x040,
        RIGHT_SIGNIF_BIT = 0x010,
        LEFT_SIGNIF_BIT = 0x002,
        TOP_SIGN_BIT = 0x080,
        BOTTOM_SIGN_BIT = 0x020,
        RIGHT_SIGN_BIT = 0x008,
        LEFT_SIGN_BIT = 0x001,
    };
    
    const int lContrib = index & LEFT_SIGNIF_BIT
                       ? (index & LEFT_SIGN_BIT ? -1 : 1) : 0;
    const int rContrib = index & RIGHT_SIGNIF_BIT
                       ? (index & RIGHT_SIGN_BIT ? -1 : 1) : 0;
    const int tContrib = index & TOP_SIGNIF_BIT
                       ? (index & TOP_SIGN_BIT ? -1 : 1) : 0;
    const int bContrib = index & BOTTOM_SIGNIF_BIT
                       ? (index & BOTTOM_SIGN_BIT ? -1 : 1) : 0;
    
    const int hContrib = ::max(::min(1, lContrib + rContrib), -1);
    const int vContrib = ::max(::min(1, tContrib + bContrib), -1);
    
    if(hContrib == 1) {
        if(vContrib == 1) return SC4 | 0;
        if(vContrib == 0) return SC3 | 0;
        return SC2 | 0;
    }
    if(hContrib == 0) {
        if(vContrib == 1) return SC1 | 0;
        if(vContrib == 0) return SC0 | 0;
        return SC1 | 1;
    }
    if(vContrib == 1) return SC2 | 1;
    if(vContrib == 0) return SC3 | 1;
    return SC4 | 1;
}


static void mqLutInit(const cudaStream_t & stream) {
    // MQ decoder: values from JPEG 2000 specification
    const u16 qe[] = {0x5601, 0x5601, 0x3401, 0x3401, 0x1801, 0x1801, 0x0ac1, 0x0ac1, 0x0521, 0x0521, 0x0221, 0x0221, 0x5601, 0x5601, 0x5401, 0x5401, 0x4801, 0x4801, 0x3801, 0x3801, 0x3001, 0x3001, 0x2401, 0x2401, 0x1c01, 0x1c01, 0x1601, 0x1601, 0x5601, 0x5601, 0x5401, 0x5401, 0x5101, 0x5101, 0x4801, 0x4801, 0x3801, 0x3801, 0x3401, 0x3401, 0x3001, 0x3001, 0x2801, 0x2801, 0x2401, 0x2401, 0x2201, 0x2201, 0x1c01, 0x1c01, 0x1801, 0x1801, 0x1601, 0x1601, 0x1401, 0x1401, 0x1201, 0x1201, 0x1101, 0x1101, 0x0ac1, 0x0ac1, 0x09c1, 0x09c1, 0x08a1, 0x08a1, 0x0521, 0x0521, 0x0441, 0x0441, 0x02a1, 0x02a1, 0x0221, 0x0221, 0x0141, 0x0141, 0x0111, 0x0111, 0x0085, 0x0085, 0x0049, 0x0049, 0x0025, 0x0025, 0x0015, 0x0015, 0x0009, 0x0009, 0x0005, 0x0005, 0x0001, 0x0001, 0x5601, 0x5601};
    const u8 nmps[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 76, 77, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 58, 59, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 90, 91, 92, 93};
    const u8 nlps[] = {3, 2, 12, 13, 18, 19, 24, 25, 58, 59, 66, 67, 13, 12, 28, 29, 28, 29, 28, 29, 34, 35, 36, 37, 40, 41, 42, 43, 29, 28, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 92, 93};
    
    // convert MQ decoder tables into packed format in host buffer
    u32 nQeHost[47 * 2];
    for(int i = 0; i < 47 * 2; i++) {
        nQeHost[i] = (((u32)qe[i]) << 16) | (((u16)nlps[i]) << 7) | nmps[i];
    }
    
    // copy packed MQ decoder tables into the constant memory buffer
    syncMemcpyToSymbol(nQeHost, nQe, sizeof(nQeHost), stream);
}



/// Makes sure that tables are initialized and possibly initializes them.
static void cxmodLutInit(const cudaStream_t & stream) {
    // prepare copy of zero decoding and sign decoding tables in host memory
    u8 hhZcLutHost[512], hlZcLutHost[512], llZcLutHost[512], scLutHost[512];
    for(int i = 512; i--;) {
        hhZcLutHost[i] = zcContext(i, ORI_HH);
        hlZcLutHost[i] = zcContext(i, ORI_HL);
        llZcLutHost[i] = zcContext(i, ORI_LL);
        scLutHost[i] = scContext(i);
    }
    
    // copy decoding tables to GPU memory
    syncMemcpyToSymbol(scLutHost, scLut, sizeof(scLutHost), stream);
    syncMemcpyToSymbol(hhZcLutHost, hhZcLut, sizeof(hhZcLutHost), stream);
    syncMemcpyToSymbol(hlZcLutHost, hlZcLut, sizeof(hlZcLutHost), stream);
    syncMemcpyToSymbol(llZcLutHost, llZcLut, sizeof(llZcLutHost), stream);
    
    // prepare and copy reconstruction tables for halfway values
    int4 halvesInt[32 * 16];
    float4 halvesFloat[32 * 16];
    for(int b = 0; b < 32; b++) {
        // halves for pixels decoded at the bitplane (bit 1) and
        // on next bitplane (bit 0)
        const int max1 = (1 << b) - 1;
        const int max0 = (2 << b) - 1;
        for(int c = 0; c < 16; c++) {
            const int i = b * 16 + c;
            halvesFloat[i].x = (i & 0x8 ? max1 : max0) * 0.5f;
            halvesFloat[i].y = (i & 0x4 ? max1 : max0) * 0.5f;
            halvesFloat[i].z = (i & 0x2 ? max1 : max0) * 0.5f;
            halvesFloat[i].w = (i & 0x1 ? max1 : max0) * 0.5f;
            halvesInt[i].x = (i & 0x8 ? max1 : max0) >> 1;
            halvesInt[i].y = (i & 0x4 ? max1 : max0) >> 1;
            halvesInt[i].z = (i & 0x2 ? max1 : max0) >> 1;
            halvesInt[i].w = (i & 0x1 ? max1 : max0) >> 1;
        }
    }
    syncMemcpyToSymbol(halvesInt, reconstructionHalvesInt,
                       sizeof(halvesInt), stream);
    syncMemcpyToSymbol(halvesFloat, reconstructionHalvesFloat,
                       sizeof(halvesFloat), stream);
}



/// Initializes the instance.
Ebcot::Ebcot() {
    cxmodLutInit(0);
    mqLutInit(0);
}



/// Releases all resources associated with the EBCOT instance.
Ebcot::~Ebcot() {
    // nothing to do
}



/// Performs EBCOT decoding on prepared decoder structure. Whole image
/// structure is expected to be copied in GPU memory.
/// @param cStream  pointer to codestream in GPU memory
/// @param image  pointer to image structure
/// @param working  working GPU double buffer
/// @param cudaStream  cuda stream to be used for decoding kernels
/// @param logger  logger for tracing procress of decoding
void Ebcot::decode(const u8 * const cStream,
                   Image * const image,
                   IOBufferGPU<u8> & working,
                   const cudaStream_t & cudaStream,
                   Logger * const logger) {
    // codeblock count
    const int cblkCount = image->cblks.count();
    
    // possibly resize (reallocate) the buffer for codeblocks
    working.outResize(image->ebcotTempSize);
    
    // kernel parameters
    EbcotKernelParams params;
    params.output = working.outPtr();
    params.cstream = (const u8*)cStream;
    params.cblks = image->cblks.getPtrGPU();
    params.cblkCount = cblkCount;
    params.segs = image->segs.getPtrGPU();
    params.bands = image->bands.getPtrGPU();
    params.res = image->res.getPtrGPU();
    params.comps = image->comps.getPtrGPU();
    params.cblkPermuation = image->cblkPerm.getPtrGPU();
    
    // number of threads per threadblock
    enum { TCOUNT = 128 };
    
    // number of threadblocks and threads in each of them
    const int blockSize = TCOUNT;
    const int gridSize = (image->cblks.count() + blockSize - 1) / blockSize;
    
    // launch the kernel
    ebcotDecodingKernel<TCOUNT><<<gridSize, blockSize, 0, cudaStream>>>(params);
    
    // possibly resize (reallocate) the buffer for codeblocks
    working.swap();
    working.outResize(image->bandsPixelCount * 4); // 4 for size of int or float
    
    // launch configuration for second kernel
    const dim3 rBSize(32, 8);
    const dim3 rGSize(divRndUp(cblkCount, (int)rBSize.y));
    
    // run the reconstruction kernel
    ebcotReconstructionKernel<<<rGSize, rBSize, 0, cudaStream>>>(
        image->cblks.getPtrGPU(),
        image->bands.getPtrGPU(),
        cblkCount,
        working.inPtr(),
        working.outPtr()
    );
}




/// Gets size of temporary memory needed for the codeblock.
/// @param size  real codeblock size (includes cropping)
/// @param stdSize  standard codeblock size (powers of two)
/// @return  number of bytes needed for temporary stuff of the codeblock
u32 Ebcot::cblkTempSize(const XY & size, const XY & stdSize) {
    // count of magnitude groups (without dummy boundary groups)
    const int magnGrpCount = size.x * (size.y + 3) / 4;
    
    // count of groups states (with dummy boundary groups states)
    const int strideX = (stdSize.y >> 2) + 1;
    const int grpStateCount = 1 + strideX * (size.x + 2);
    const int grpState8Count = (grpStateCount + 7) >> 3;
    
    // each group of magnitudes needs 16 bytes and each 8 group states 
    // need 16 bytes too
    return 16 * (grpState8Count + magnGrpCount);
}



} // end of namespace cuj2kd





