///
/// @file    j2kd_out_fmt.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Output formatting stuff declarations.
///

#ifndef J2KD_OUT_FMT_H
#define J2KD_OUT_FMT_H

#include "j2kd_type.h"
#include "j2kd_image.h"

namespace cuj2kd {



/// Abstract output formatting class (the interface).
class OutputFormatter {
public:
    /// Constructor - Initializes static formatting stuff
    OutputFormatter();
    
    /// Computes formatted output size, checking for indices out of bounds.
    /// @param image  pointer to image structure instance
    /// @param format  pointer to array of component format infos
    ///                (must be immutable at least to end of current decoding)
    /// @param count   number of output components
    /// @param capacity  capacity of the output buffer
    /// @return size of the buffer needed for formatted data
    static size_t checkFmt(
        Image * const image,
        const CompFormat * const format,
        const int count,
        const size_t capacity
    );
    
    /// This does the output formatting.
    /// @param image  pointer to image structure instance
    /// @param src source buffer
    /// @param out  pointer to GPU output buffer
    /// @param stream  CUDA stream to launch kernels in
    /// @param format  pointer to array of component format infos
    ///                (must be immutable at least to end of current decoding)
    /// @param count   number of output components
    void run(
        Image * const image,
        const void * const src,
        void * const out,
        const cudaStream_t & stream,
        const CompFormat * const format,
        const int count
    );
};



} // end of namespace cuj2kd

#endif // J2KD_OUT_FMT_H

