///
/// @file    j2kd_decoding.cpp
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   High-level operations for JPEG 2000 decoding.
///


#include "j2kd_decoder.h"
#include "j2kd_timer.h"
#include "j2kd_mct.h"



namespace cuj2kd {


Decoder::WorkItem::WorkItem() : error(J2KD_OK, "OK") {
    // allocate cuda stream
    checkCudaCall(cudaStreamCreate(&stream), "cudaStreamCreate");
}


Decoder::WorkItem::~WorkItem() {
    // release CUDA stream
    checkCudaCall(cudaStreamDestroy(stream), "cudaStreamDestroy");
}


bool Decoder::WorkItem::isActive() {
    return loaded && error.getStatusCode() == J2KD_OK;
}


void Decoder::WorkItem::clear() {
    loaded = false;
    error = Error(J2KD_OK, "OK");
    fmtPtr = 0;
    fmtCount = 0;
    customImagePtr = 0;
}



void Decoder::run(
    InBeginCallback inBeginCallback,
    InEndCallback inEndCallback,
    OutCallback outCallback,
//     PostprocCallback postprocCallback,
    DecEndCallback decEndCallback,
    void * const customCallbackPtr
) {
    // initialize work items
    for(int i = 0; i < 3; i++) {
        workItems[i].clear();
    }
    
    // Pointers to various stages of encoding process.
    // (Rotated in each iteration.)
    WorkItem * loadingItem = workItems + 0;
    WorkItem * workingItem = workItems + 1;
    WorkItem * savingItem = workItems + 2;
    Image * loadingImage = images + 0;
    Image * workingImage = images + 1;
    GPUBuffer * loadingTemp = temp + 0;
    GPUBuffer * workingTemp = temp + 1;
    
    // true if more images should be asked for using callback
    bool loadInput = true;
    
    // output settings
    void * outBufferPtr = 0;
    size_t outBufferCapacity = 0;
    int outBufferInDeviceMem = 0;  // nonzero if in device memeory, 0 if NOT
    size_t outSize = 0;
    
    // run as long as there are partially decoded images 
    // or more images should be loaded
    do {
        // TODO: add try-catch blocks!!!!!!
        
        // start output memcpy if output work item is active
        if(savingItem->isActive()) {
            // start memcpy only if output buffer is not in device memory
            if(!outBufferInDeviceMem) {
                asyncMemcpy(
                    output.getPtr(),
                    outBufferPtr,
                    outSize,
                    false,
                    savingItem->stream
                );
            }
        }
        
        // start decoding if encoding work item is loaded (active)
        if(workingItem->isActive()) {
            // reorder buffers (so they use same resizing pattern each time)
            data.reset();
            
            // issue kernels for all decoding phases except of formatting
            const u8 * const cStreamPtr = (const u8*)workingTemp->getPtr();
            ebcot.decode(cStreamPtr, workingImage, data, workingItem->stream, &logger);
            dwt.transform(*workingImage, data, *workingTemp, workingItem->stream);
            mctUndo(workingImage, data.outPtr(), workingItem->stream, &logger);
        }
        
        // run input callback (if there may be more inputs)
        const void * cStreamPtr = 0;
        size_t cStreamSize = 0;
        if(loadInput) {
            loadingItem->clear();
            const CompFormat * compFormatArray = 0;
            int compFormatCount = 0;
            
            // should the callback block? (not if decoder has other work)
            const bool haveOtherWork = workingItem->isActive()
                    || savingItem->isActive();
            
            // get next input
            loadInput = inBeginCallback(
                customCallbackPtr,
                &loadingItem->customImagePtr,
                &cStreamPtr,
                &cStreamSize,
                &loadingItem->fmtPtr,
                &loadingItem->fmtCount,
                haveOtherWork ? 0 : 1
            );
            
            // mark working item as active if everything was set
            if(cStreamPtr) {
                loadingItem->loaded = true;
                
                // check other parameters ...
                if(0 == loadingItem->fmtPtr) {
                    loadingItem->error = Error(J2KD_ERROR_ARGUMENT_NULL,
                                               "Format pointer is NULL.");
                }
            }
        }
        
        // start loading if loading item is OK
        if(loadingItem->isActive()) {
            // start async memcpy of codestream into the GPU buffer
            loadingTemp->resize(cStreamSize + 256);
            asyncMemcpy(
                cStreamPtr,
                loadingTemp->getPtr(),
                cStreamSize,
                true,
                loadingItem->stream
            );
            
            // run T2 on CPU
            loadingImage->clear();
            t2.analyze(loadingImage, (const u8*)cStreamPtr, cStreamSize, &logger);
            
            // wait for codestream memcpy to be done
            streamSync(loadingItem->stream);
            
            // copy image structure into GPU buffers
            loadingImage->copyToGPU(loadingItem->stream);
        }
        
        // run input end callback if input item is loaded
        if(loadingItem->loaded) {
            // return input codestream buffer to caller
            inEndCallback(
                customCallbackPtr,
                loadingItem->customImagePtr,
                cStreamPtr
            );
        }
        
        // wait for output memcpy end if saving active
        if(savingItem->loaded) {
            // do not wait for memcpy if output is in GPU memory
            if(!outBufferInDeviceMem) {
                streamSync(savingItem->stream);
            }
            
            // invoke decoding-end callback
            decEndCallback (
                customCallbackPtr,
                savingItem->customImagePtr,  // TODO: add status message to API
                savingItem->error.getStatusCode(),
                savingItem->fmtPtr,
                outBufferPtr
            );
            
            // mark work item as unused
            savingItem->loaded = false;
        }
        
        // call output callback for working stage and issue formatting kernel
        if(workingItem->loaded) {
            // initialize output parameters
            outBufferPtr = 0;
            outBufferCapacity = 0;
            outBufferInDeviceMem = 0;
            
            // run the get-output-buffer callback
            outCallback(
                customCallbackPtr,
                workingItem->customImagePtr,
                &outBufferPtr,
                &outBufferCapacity,
                &outBufferInDeviceMem
            );
            
            // check output buffer pointer
            if(0 == outBufferPtr) {
                workingItem->error = Error(J2KD_ERROR_ARGUMENT_NULL,
                                           "Output buffer pointer is NULL");
            }
        }
        
        // run output formatting is working item is OK
        if(workingItem->isActive()) {
            // select output pointer for formatting
            data.swap();
            void * fmtDataOutPtr = outBufferPtr;
            if(!outBufferInDeviceMem) {
                // format into output GPU buffer and later memcpy to host
                fmtDataOutPtr = output.resize(outBufferCapacity);
            }
            
            // issue output formatting kernel
            outSize = fmt.run(
                workingImage,
                data.inPtr(),
                fmtDataOutPtr,
                outBufferCapacity,
                workingItem->stream,
                workingItem->fmtPtr,
                workingItem->fmtCount
            );
        }
        
        // rotate working items for next iteration
        WorkItem * const unusedItem = savingItem;
        savingItem = workingItem;  // save decoding output in next iteration
        workingItem = loadingItem; // decode loaded item in next iteration
        loadingItem = unusedItem;  // load into unused item in next iteration
        
        // swap image structures for next iteration
        Image * const unusedImage = workingImage;
        workingImage = loadingImage;  // decode using loaded image structure
        loadingImage = unusedImage;   // load next into unused image structure
        
        // swap temp buffers for next iteration
        GPUBuffer * const unusedTemp = workingTemp;
        workingTemp = loadingTemp;    // decode currently loaded codestream
        loadingTemp = unusedTemp;     // load into unused temp buffer
        
        // repeat if input callback returned nonzero
    } while(loadInput || workingItem->isActive() || savingItem->isActive());
}











struct SingleImageParams {
    StatusCode status;  // encoding result
    void * outBufferPtr;
    size_t outBufferSize;
    int outInDeviceMem;
    const void * cStreamPtr;
    size_t cStreamSize;
    const CompFormat * compFmtPtr;
    size_t compFmtCount;
};




int singleImageInBeginCallback(
    void * customCallbackPtr,
    void **,
    const void ** cStreamPtrOut,
    size_t * cStreamSizeOut,
    const CompFormat ** compFmtPtrOut,
    int * compFmtCountOut,
    int shouldBlock
) {
    // cast parameter to right type
    const SingleImageParams * params = (SingleImageParams*)customCallbackPtr;
    
    // fill all output parameters
    *cStreamPtrOut = params->cStreamPtr;
    *cStreamSizeOut = params->cStreamSize;
    *compFmtPtrOut = params->compFmtPtr;
    *compFmtCountOut = params->compFmtCount;
    
    // indicate that there won't be more images
    return 0;
}


void singleImageInEndCallback(void *, void *, const void *) {
    // does nothing
}


void singleImageOutCallback(
    void * customCallbackPtr,
    void *,
    void ** outPtrOut,
    size_t * outCapacityOut,
    int * outInDeviceMemOut
)  {
    // cast parameter to right type
    const SingleImageParams * params = (SingleImageParams*)customCallbackPtr;
    
    // fill all output parameters
    *outPtrOut = params->outBufferPtr;
    *outCapacityOut = params->outBufferSize;
    *outInDeviceMemOut = params->outInDeviceMem;
}


void singleImageDecEndCallback(
    void * customCallbackPtr,
    void *,
    StatusCode status,
    const CompFormat *,
    void *
) {
    // cast parameter to right type
    SingleImageParams * params = (SingleImageParams*)customCallbackPtr;
    
    // remember the status
    params->status = status;
}


void Decoder::decode(
    const u8 * const cStreamPtr,
    const size_t cStreamSize,
    void * outPtr,
    const size_t outCapacity,
    const int outOnGPU,
    const CompFormat * const compFormatPtr,
    const int compFormatCount
) {
    // prepare parameters
    SingleImageParams params;
    params.status = J2KD_ERROR_UNKNOWN;
    params.compFmtCount = compFormatCount;
    params.compFmtPtr = compFormatPtr;
    params.cStreamPtr = cStreamPtr;
    params.cStreamSize = cStreamSize;
    params.outBufferPtr = outPtr;
    params.outBufferSize = outCapacity;
    params.outInDeviceMem = outOnGPU;
    
    // call the decoder with set of special single-image-decoding callbacks
    run(
        singleImageInBeginCallback,
        singleImageInEndCallback,
        singleImageOutCallback,
        singleImageDecEndCallback,
        &params
    );
    
    // TODO: should throw error from output callback if not decoded correctly
}


} // end of namespace cuj2kd

