///
/// @file    t2.cpp
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Switch for encoding T2 either on GPU or on CPU.
///


#include <cstdlib>
#include "t2.h"
#include "cpu/t2_cpu.h"
#include "gpu/t2_gpu.h"



/// internal stuff of T2 encoder.
struct j2k_t2_encoder {
    /// Pointer to faster, but limited, GPU version of T2 encoder.
    t2_gpu_encoder * gpu_t2_enc_ptr;
    
    /// Pointer to slower, but universal, CPU version of T2 encoder.
    t2_cpu_encoder * cpu_t2_enc_ptr;
};



/// Initializes T2 encoder for given JPEG 2000 encoder structure.
/// @param enc  initialized structure of JPEG 2000 encoder
/// @return  either pointer to newly created T2 encoder 
///          or 0 if anything goes wrong
struct j2k_t2_encoder * j2k_t2_create(const struct j2k_encoder * const enc)  {
    // allocate the structure
    struct j2k_t2_encoder * t2_enc
            = (struct j2k_t2_encoder*)malloc(sizeof(struct j2k_t2_encoder));
    
    // clear both pointers
    t2_enc->cpu_t2_enc_ptr = 0;
    t2_enc->gpu_t2_enc_ptr = 0;
    
    // try to allocate GPU implementation first, then CPU one
    if((t2_enc->gpu_t2_enc_ptr = t2_gpu_create(enc))
        || (t2_enc->cpu_t2_enc_ptr = t2_cpu_create(enc))) {
        // either GPU or CPU implementation initialization succeded => return 
        return t2_enc;
    } else {
        // both implementations failed to initialize => free the structure ...
        free(t2_enc);
        
        // ... and indicate failure by returning 0
        return 0;
    }
}



/// Encodes packets in given JPEG2000 encoder structure.
/// @param j2k_enc   pointer to J2K encoder with initialized T2 pointer
/// @param t2_enc    pointer to T2 encoder for given J2K encoder
/// @param out_ptr   pointer to output buffer in main memory, where the output 
///                  should be written
/// @param out_size  size of output buffer
/// @return  either size of output stream (in bytes) if encoded OK,
///          or negative error code if failed
int j2k_t2_encode(const struct j2k_encoder * const j2k_enc,
              struct j2k_t2_encoder * const t2_enc,
              unsigned char * const out,
              const int out_size) {
    if(t2_enc->gpu_t2_enc_ptr) {
        return t2_gpu_encode(j2k_enc, t2_enc->gpu_t2_enc_ptr, out, out_size);
    } else if (t2_enc->cpu_t2_enc_ptr) {
        return t2_cpu_encode(j2k_enc, t2_enc->cpu_t2_enc_ptr, out, out_size);
    } else {
        return -1;
    }
}



/// Integer logarithm (base 2)
static int ilog2(size_t n) {
    int result = 0;
    while(n >>= 1) {
        result++;
    }
    return result;
}


/// Gets maximal size of headers for encoding images with given settings.
/// @param enc  pointer to initialized structure of encoder
/// @param rate  target byte count
/// @return maximal size (in bytes) of all headers needed for T2 encoding 
///         with given encoder structure
int j2k_t2_get_overhead(const struct j2k_encoder * const enc, const size_t rate) {
    // basic headers
    int overhead = 200 + enc->params.comp_count * 10;
    
    // per-precinct fixed overhead
    int prec_overhead = 2;
    if(enc->params.use_eph) {
        prec_overhead += 2;
    }
    if(enc->params.use_sop) {
        prec_overhead += 6;
    }
    overhead += prec_overhead * enc->precinct_count;
    
    // overal codeblock info overhead estimate
    overhead += (ilog2(enc->cblk_count) * enc->params.bit_depth) / 16;
    
    // target variable size overhead
    overhead += ilog2(rate) * 100 + rate / 100;
//     printf("Overhead: %d\n", overhead);
    return overhead;
}



/// Releases resources associated to given T2 encoder.
/// @param t2_enc  pointer to t2 encoder
/// @return 0 if succeded, negative if failed
int j2k_t2_destroy(struct j2k_t2_encoder * const t2_enc) {
    int status = 0;
    if(t2_enc) {
        if(t2_enc->gpu_t2_enc_ptr) {
            status |= t2_gpu_destroy(t2_enc->gpu_t2_enc_ptr);
        }
        if(t2_enc->cpu_t2_enc_ptr) {
            status |= t2_cpu_destroy(t2_enc->cpu_t2_enc_ptr);
        }
        free(t2_enc);
    }
    return status;
}






