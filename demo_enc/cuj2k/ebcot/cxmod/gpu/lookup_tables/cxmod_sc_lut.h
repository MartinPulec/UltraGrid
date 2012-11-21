///
/// @file    cxmod_sc_luts.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Variuos versions of lookup table for sign coding.
///

#ifndef CXMOD_SC_LUTS_H
#define CXMOD_SC_LUTS_H

namespace cxmod_cuda {
    
    namespace lookup_tables {

        // Sign coding lookup table.
        // (Each second bit is cleared - will be XORed with sign.)
        static const unsigned char sc_lut[25] = {
            0x35, // index 0,  V = -1, H = -1  =>  CX=13, X=1
            0x35, // index 1,  V = -1, H = -1  =>  CX=13, X=1
            0x31, // index 2,  V =  0, H = -1  =>  CX=12, X=1
            0x2D, // index 3,  V = +1, H = -1  =>  CX=11, X=1
            0x2D, // index 4,  V = +1, H = -1  =>  CX=11, X=1
            0x35, // index 5,  V = -1, H = -1  =>  CX=13, X=1
            0x35, // index 6,  V = -1, H = -1  =>  CX=13, X=1
            0x31, // index 7,  V =  0, H = -1  =>  CX=12, X=1
            0x2D, // index 8,  V = +1, H = -1  =>  CX=11, X=1
            0x2D, // index 9,  V = +1, H = -1  =>  CX=11, X=1
            0x29, // index 10, V = -1, H =  0  =>  CX=10, X=1
            0x29, // index 11, V = -1, H =  0  =>  CX=10, X=1
            0x24, // index 12, V =  0, H =  0  =>  CX=9,  X=0
            0x28, // index 13, V = +1, H =  0  =>  CX=10, X=0
            0x28, // index 14, V = +1, H =  0  =>  CX=10, X=0
            0x2C, // index 15, V = -1, H = +1  =>  CX=11, X=0
            0x2C, // index 16, V = -1, H = +1  =>  CX=11, X=0
            0x30, // index 17, V =  0, H = +1  =>  CX=12, X=0
            0x34, // index 18, V = +1, H = +1  =>  CX=13, X=0
            0x34, // index 19, V = +1, H = +1  =>  CX=13, X=0
            0x2C, // index 20, V = -1, H = +1  =>  CX=11, X=0
            0x2C, // index 21, V = -1, H = +1  =>  CX=11, X=0
            0x30, // index 22, V =  0, H = +1  =>  CX=12, X=0
            0x34, // index 23, V = +1, H = +1  =>  CX=13, X=0
            0x34  // index 24, V = +1, H = +1  =>  CX=13, X=0
        }; // end of sc_lut
        
    } // end of namespace lookup_tables
    
} // end of namespace cxmod_cuda

#endif // CXMOD_SC_LUTS_H

