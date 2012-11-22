///
/// @file    j2kd_ebcot.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Declaration of EBCOT interface.
///

#ifndef J2KD_EBCOT_H
#define J2KD_EBCOT_H

#include "j2kd_type.h"
#include "j2kd_image.h"
#include "j2kd_logger.h"

namespace cuj2kd {

    

/// EBCOT decoder for JPEG 2000.
class Ebcot {
private:
    // no attributes in current implementation
    
public:
    /// Initializes EBCOT decoder.
    Ebcot();
    
    /// Releases all resources associated with the EBCOT instance.
    ~Ebcot();
    
    /// Performs EBCOT decoding on prepared decoder structure. Whole image
    /// structure is expected to be copied in GPU memory.
    /// @param cStream  pointer to codestream in GPU memory
    /// @param image  pointer to image structure
    /// @param working  working GPU double buffer
    /// @param cudaStream  cuda stream to be used for decoding kernels
    /// @param logger  logger for tracing procress of decoding
    void decode(const u8 * const cStream,
                Image * const image,
                IOBufferGPU<u8> & working,
                const cudaStream_t & cudaStream,
                Logger * const logger);
    
    /// Gets size of temporary memory needed for the codeblock.
    /// @param size  real codeblock size (includes cropping)
    /// @param stdSize  standard codeblock size (powers of two)
    /// @return  number of bytes needed for temporary stuff of the codeblock
    static u32 cblkTempSize(const cuj2kd::XY& size, const cuj2kd::XY& stdSize);
    
}; // end of class Ebcot



} // end of namespace cuj2kd

#endif // J2KD_EBCOT_H




