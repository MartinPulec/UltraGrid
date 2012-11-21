///
/// @file t2_gpu.cpp
/// @author Martin Jirman (martin.jirman@cesnet.cz)
/// @brief Implementation of T2 GPU encoder for JPEG2000
///


#include "t2_gpu.h"



/// Represents T2 GPU implementation's internal data.
struct t2_gpu_encoder {
    // TODO: add parameters later
};



/// Initializes T2 GPU encoder for given JPEG 2000 encoder structure.
/// @param enc  initialized structure of JPEG 2000 encoder
/// @return  either pointer to newly created T2 encoder 
///          or 0 if anything goes wrong
t2_gpu_encoder * t2_gpu_create(const j2k_encoder * const /*enc*/) {
    return 0; // TODO: implement
}



/// Encodes packets in given JPEG2000 encoder structure.
/// @param j2k_enc   pointer to J2K encoder with initialized T2 pointer
/// @param t2_enc    pointer to T2 GPU encoder for given J2K encoder
/// @param out_ptr   pointer to output buffer in main memory, where the output 
///                  should be written
/// @param out_size  size of output buffer
/// @return  either size of output stream (in bytes) if encoded OK,
///          or negative error code if failed
int t2_gpu_encode(const struct j2k_encoder * const /*j2k_enc*/,
                  t2_gpu_encoder * const /*t2_enc*/,
                  unsigned char * const /*out*/,
                  const int /*out_size*/) {
    return -1; // TODO: implement
}



/// Releases resources associated to given T2 GPU encoder.
/// @param t2_enc  pointer to T2 GPU encoder
/// @return 0 if succeded, negative if failed
int t2_gpu_destroy(t2_gpu_encoder * const /*t2_enc*/) {
    return -1; // TODO: implement
}

