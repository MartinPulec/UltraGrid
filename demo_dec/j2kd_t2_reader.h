///
/// @file    j2kd_t2_reader.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Declaration of codestream reader for Tier-2 of JPEG 2000 decoder.
///



// prevent from multiple includes into the same file
#ifndef J2KD_T2_READER_H
#define J2KD_T2_READER_H

#include "j2kd_type.h"


namespace cuj2kd {


/// Allows one to read JPEG 2000 codestream at bit or byte levels, including
/// undoing effects of bit stuffing.
class T2Reader {
private:
    /// index of next codestream byte
    int next;
    
    /// pointer to begin of codestream partition
    const u8 * const begin;
    
    /// pointer to the end of the codestream (first byte after the end)
    const u8 * const end;
    
    /// index of next bit to be read
    int bitIdx;
    
public:
    /// Creates new instance of codestream reader, specifying start of 
    /// input codestream partition and its remaining size.
    T2Reader(const u8 * const begin, const size_t size)
        : next(0), begin(begin), end(begin + size), bitIdx(8) {}
    
    
    /// Gets true if there are at least specified number of bytes to be read.
    bool hasBytes(const int count) const { return (int)bytesRemaining() >= count; }
    
    
    /// Reads next 8bit number (1 byte), aligned to byte boundary.
    /// (Discards any remaining bits from previous byte.)
    u8 readU8() { return begin[next++]; }
    
    
    /// Reads 16bit number, aligned to byte boundary.
    /// (Discards any remaining bits from previous byte.)
    u16 readU16() {
        const u16 h = readU8();
        const u16 l = readU8();
        return (h << 8) | l;
    }
    
    
    /// Reads 32bit number, aligned to byte boundary.
    /// (Discards any remaining bits from previous byte.)
    u32 readU32() {
        const u32 h = readU16();
        const u32 l = readU16();
        return (h << 16) | l;
    }
    
    /// Reads 16bit number, aligned to byte boundary, but does not discard it.
    u16 getU16() {
        const u16 value = readU16();
        next -= 2;
        return value;
    }
    
    /// Skips specified number of bytes.
    void skip(const size_t bytes) {
        next += bytes;
    }
    
    
    /// Reads next bit, undoing effects of bit stuffing.
    /// @return either 1 or 0
    u8 readBit() {
        const u8 bit = 1 & (begin[next] >> --bitIdx);
        if(0 == bitIdx) {
            bitIdx = (begin[next++] == 0xff) ? 7 : 8;
        }
        return bit;
    }
    
    /// Gets more bits, applying bit unstuffing.
    u32 readBits(int count) {
        u32 result = 0;
        while(count--) {
            result = (result << 1) | readBit();
        }
        return result; 
    }
    
    /// Reads up to 'limit' 0 bits terminated by 1 bit.
    int readZeroBits(const int limit, bool & terminated) {
        for(int i = 0; i < limit; i++) {
            if(1 == readBit()) {
                terminated = true;
                return i;
            }
        }
        terminated = false;
        return limit;
    }
    
    /// Gets count of consecutive 1 bits terminated by 0 bit.
    int readOneBits() {
        int count = 0;
        while(readBit()) { count++; }
        return count;
    }
    
    /// Aligns the reader to next byte boundary, discarding up to 7 bits.
    void align() {
        if(bitIdx < 8) {
            bitIdx = 8;
            next += (begin[next] == 0xff) ? 2 : 1;
        }
    }
    
    
    /// Gets pointer to next byte to be read.
    const u8 * pos() const { return begin + next; }
    
    
    /// Gets count of remaining bytes.
    ptrdiff_t bytesRemaining() const { return (end - begin) - (ptrdiff_t)next; }
    
}; // end fo class T2Reader


} // end of namespace cuj2kd


#endif // J2KD_T2_READER_H
