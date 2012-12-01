///
/// @file    j2kd_decoder.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Declaration of decoder structure.
///

#ifndef J2KD_DECODER_H
#define J2KD_DECODER_H


#include "j2kd_type.h"
#include "j2kd_t2.h"
#include "j2kd_image.h"
#include "j2kd_ebcot.h"
#include "j2kd_dwt.h"
#include "j2kd_out_fmt.h"


namespace cuj2kd {


    
    
    
    
    /// Declaration of decoder instance
class Decoder {
private:

    // TODO: add 3 work item structures here
    struct WorkItem {
        cudaStream_t stream;
        bool loaded;
        Error error;  // caught exception (initialized to OK for each image)
        const CompFormat * fmtPtr;
        int fmtCount;
        void * customImagePtr;
        
        WorkItem();
        ~WorkItem();
        void clear();
        bool isActive();
    };
    
    /// 3 Work items of the instance
    WorkItem workItems[3];
    
    // Buffers
    IOBufferGPU<u8> data;       ///< GPU working buffer pair (input and output)
    GPUBuffer temp[2];          ///< two temporary buffers
    Image images[2];            ///< two image structures
    GPUBuffer output;           ///< output buffer

    // Decoding stages:
    Tier2 t2;
    Ebcot ebcot;
    DWT dwt;
    OutputFormatter fmt;
    
    // Other stuff
    Logger logger;
    
    static bool invokeInputBeginCallback(WorkItem * info);
    
public:
    void decode(
        const u8 * const cStreamPtr,
        const size_t cStreamSize,
        void * outPtr,
        const size_t outCapacity,
        const int outOnGPU,
        const CompFormat * const compFormatPtr,
        const int compFormatCount
    );
    
    
    
    void run(
        InBeginCallback inBeginCallback,
        InEndCallback inEndCallback,
        OutCallback outCallback,
        PostprocCallback postprocCallback,
        DecEndCallback decEndCallback,
        void * const customCallbackPtr
    );
    
    
    
    
    
}; // end of struct Decoder












} // end of namespace cuj2kd

#endif // J2KD_DECODER_H
