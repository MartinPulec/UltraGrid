///
/// @file    t2_cpu_output.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Writer of output bytes and bits.
///


#ifndef T2_CPU_OUTPUT_H
#define T2_CPU_OUTPUT_H

#include <string.h>


/// JPEG2000 T2 output writer.
class t2_cpu_output_t {
private:
    /// pointer to next output byte
    unsigned char * out_ptr;
    
    /// pointer to end of output buffer
    const unsigned char * out_end;
    
    /// position of next bit in incomplete output byte
    int bit_pos;
    
    /// incomplete output byte
    unsigned char byte;
    
public:
    /// Sets begin and end of output buffer.
    void init(unsigned char * const out_begin,
              const unsigned char * const out_end)
    {
        this->out_ptr = out_begin;
        this->out_end = out_end;
        this->byte = 0;
        this->bit_pos = 8;
    }
    
    
    /// Flushes remaining bits which don't form complete byte.
    void flush_bits()
    {
        if(bit_pos != 8) {
            *(out_ptr++) = byte;
            bit_pos = 8;
            byte = 0;
        }
    }
    
    
    /// Puts multiple bits (up to 32), starting with msb.
    void put_bits(const unsigned int bits, int count)
    {
        while (count >= bit_pos) {
            // fill incomplete byte's lsbs with msbs of the bit string
            const unsigned int mask = (1 << bit_pos) - 1;
            count -= bit_pos;
            byte |= (bits >> count) & mask;
            
            // flush the byte
            *(out_ptr++) = byte;
            
            // reinitialize (with possible zero bit stuffing)
            bit_pos = (byte == 0xFF) ? 7 : 8;
            byte = 0;
        }
        
        // put remaining bits to the right place in the incomplete byte
        if(count) {
            const unsigned int mask = (1 << count) - 1;
            bit_pos -= count;
            byte |= (bits & mask) << bit_pos;
        }
    }
    
    
    /// Puts single 1 bit into the output.
    void put_one()
    {
        bit_pos--;
        byte |= 1 << bit_pos;
        if(bit_pos == 0)
        {
            *(out_ptr++) = byte;
            bit_pos = (byte == 0xFF) ? 7 : 8; // zero bit stuffing
            byte = 0;
        }
    }
    
    
    /// Puts single 0 bit into the output.
    void put_zero()
    {
        bit_pos--;
        if(bit_pos == 0)
        {
            *(out_ptr++) = byte;
            bit_pos = 8;
            byte = 0;
        }
    }
    
    
    /// Puts one byte into the output
    void put_byte(const unsigned char byte)
    {
        *(out_ptr++) = byte;
    }
    
    
    /// Puts two bytes into the output.
    void put_2bytes(const unsigned short bits)
    {
        put_byte((unsigned char)(bits >> 8));
        put_byte((unsigned char)(bits));
    }
    
    
    /// Puts 4 bytes into the output.
    void put_4bytes(const unsigned int bits)
    {
        put_byte((unsigned char)(bits >> 24));
        put_byte((unsigned char)(bits >> 16));
        put_byte((unsigned char)(bits >> 8));
        put_byte((unsigned char)(bits));
    }
    
    
    /// Gets current end of the output.
    unsigned char * get_end() const
    {
        return out_ptr;
    }
    
    
    /// Puts string of zero bits into the output.
    void put_zeros(int count)
    {
        // fill remaining bits with zeros and flush
        if(count >= bit_pos)
        {
            // get together complete byte using accumulated bits and write it
            count -= bit_pos;
            *(out_ptr++) = byte;
            
            // write full zero bytes
            while(count >= 8)
            {
                *(out_ptr++) = 0;
                count -= 8;
            }
            
            // initialize next byte
            bit_pos = 8 - count;
            byte = 0;
        }
        else
        {
            // just skip some count of zero bits
            bit_pos -= count;
        }
    }
    
 
    /// Puts string of one bits into the output.
    void put_ones(int count)
    {
        put_bits(~0, count);
    }

 
    /// Puts range of bytes into the output.
    void put_bytes(const unsigned char * const bytes_ptr, const int byte_count)
    {
        memcpy(out_ptr, bytes_ptr, byte_count);
        out_ptr += byte_count;
    }
    

    /// Is there enough space in the output?
    bool has_space(const int byte_count) const
    {
        return (out_end - out_ptr) >= byte_count;
    }
    
    
}; // end of struct t2_cpu_output_t



#endif // T2_CPU_OUTPUT_H


