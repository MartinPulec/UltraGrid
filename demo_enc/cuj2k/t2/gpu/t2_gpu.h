///
/// @file    t2_gpu.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Interface of GPU (fast, but limited) Tier 2 encoding for JPEG2000.
/// 

#ifndef T2_GPU_H
#define T2_GPU_H

#include "../../j2k_encoder.h"



/// Represents T2 GPU implementation's internal data.
struct t2_gpu_encoder;


/// Initializes T2 GPU encoder for given JPEG 2000 encoder structure.
/// @param enc  initialized structure of JPEG 2000 encoder
/// @return  either pointer to newly created T2 encoder 
///          or 0 if anything goes wrong
t2_gpu_encoder * t2_gpu_create(const j2k_encoder * const enc);


/// Encodes packets in given JPEG2000 encoder structure.
/// @param j2k_enc   pointer to J2K encoder with initialized T2 pointer
/// @param t2_enc    pointer to T2 GPU encoder for given J2K encoder
/// @param out_ptr   pointer to output buffer in main memory, where the output 
///                  should be written
/// @param out_size  size of output buffer
/// @return  either size of output stream (in bytes) if encoded OK,
///          or negative error code if failed
int t2_gpu_encode(
    const struct j2k_encoder * const j2k_enc,
    t2_gpu_encoder * const t2_enc,
    unsigned char * const out,
    const int out_size
);


/// Releases resources associated to given T2 GPU encoder.
/// @param t2_enc  pointer to T2 GPU encoder
/// @return 0 if succeded, negative if failed
int t2_gpu_destroy(t2_gpu_encoder * const t2_enc);



#endif // T2_GPU_H
