///
/// @file    t2_cpu.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Interface of CPU (fallback) Tier 2 encoding for JPEG2000.
/// 

#ifndef T2_CPU_H
#define T2_CPU_H

#include "../../j2k.h"



/// Represents T2 CPU implementation's internal data.
struct t2_cpu_encoder;


/// Initializes T2 CPU encoder for given JPEG 2000 encoder structure.
/// @param enc  initialized structure of JPEG 2000 encoder
/// @return  either pointer to newly created T2 encoder 
///          or 0 if anything goes wrong
t2_cpu_encoder * t2_cpu_create(const struct j2k_encoder * const enc);


/// Encodes packets in given JPEG2000 encoder structure.
/// @param j2k_enc   pointer to J2K encoder with initialized T2 pointer
/// @param t2_enc    pointer to T2 CPU encoder for given J2K encoder
/// @param out_ptr   pointer to output buffer in main memory, where the output 
///                  should be written
/// @param out_size  size of output buffer
/// @param subsampled true if half sized image should be saved
/// @return  either size of output stream (in bytes) if encoded OK,
///          or negative error code if failed
int t2_cpu_encode(
    const struct j2k_encoder * const j2k_enc,
    t2_cpu_encoder * const t2_enc,
    unsigned char * const out,
    const int out_size,
    const bool subsampled
);


/// Releases resources associated to given T2 CPU encoder.
/// @param t2_enc  pointer to t2 encoder
/// @return 0 if succeded, negative if failed
int t2_cpu_destroy(t2_cpu_encoder * const t2_enc);



#endif // T2_CPU_H
