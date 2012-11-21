///
/// @file    t2.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Interface of Tier 2 encoding for JPEG2000.
/// 

#ifndef T2_H
#define T2_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../j2k.h"
#include "t2_type.h"


/// Initializes T2 encoder for given JPEG 2000 encoder structure.
/// @param enc  initialized structure of JPEG 2000 encoder
/// @return  either pointer to newly created T2 encoder 
///          or 0 if anything goes wrong
struct j2k_t2_encoder * j2k_t2_create(const struct j2k_encoder * const enc);


/// Encodes packets in given JPEG2000 encoder structure.
/// @param j2k_enc   pointer to J2K encoder with initialized T2 pointer
/// @param t2_enc    pointer to T2 encoder for given J2K encoder
/// @param out_ptr   pointer to output buffer in main memory, where the output 
///                  should be written
/// @param out_size  size of output buffer
/// @return  either size of output stream (in bytes) if encoded OK,
///          or negative error code if failed
int j2k_t2_encode(
    const struct j2k_encoder * const j2k_enc,
    struct j2k_t2_encoder * const t2_enc,
    unsigned char * const out,
    const int out_size
);


/// Gets maximal size of headers for encoding images with given settings.
/// @param enc  pointer to initialized structure of encoder
/// @param rate  target byte count
/// @return maximal size (in bytes) of all headers needed for T2 encoding 
///         with given encoder structure
int j2k_t2_get_overhead(const struct j2k_encoder * const enc, const size_t rate);


/// Releases resources associated to given T2 encoder.
/// @param t2_enc  pointer to t2 encoder
/// @return 0 if succeded, negative if failed
int j2k_t2_destroy(struct j2k_t2_encoder * const t2_enc);



#ifdef __cplusplus
} // end of extern "C" {
#endif

#endif // T2_H

