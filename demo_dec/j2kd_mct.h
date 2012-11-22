///
/// @file    j2kd_mct.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Interface of JPEG 2000 reverse MCT.
///

#ifndef J2KD_MCT_H
#define J2KD_MCT_H

#include "j2kd_cuda.h"
#include "j2kd_image.h"
#include "j2kd_logger.h"

namespace cuj2kd {


/// Performs reverse MCT where needed.
/// @param image  pointer to image structure
/// @param data  pointer to buffer with data
/// @param cudaStream  cuda stream to be used for decoding kernels
/// @param logger  logger for tracing progress of the decoding
void mctUndo(Image * const image,
             void * const data,
             const cudaStream_t & cudaStream,
             Logger * const logger);


} // end of namespace cuj2kd

#endif // J2KD_MCT_H




