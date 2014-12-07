/**
 * @file   cmpto_j2k.h
 * @Author Comprimato Systems s.r.o. (support@comprimato.com)
 * @date   May, 2013
 * @brief  Comprimato JPEG2000@GPU API
 *
 * Comprimato JPEG2000@GPU API.
 */

#ifndef _CMPTO_J2K_H
#define _CMPTO_J2K_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>


/** Library API functions attributes definition */
#if defined(__linux__)
#define CMPTO_J2K_API_FUNCTION __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#ifdef CMPTO_API_EXPORTS
#define CMPTO_J2K_API_FUNCTION __declspec(dllexport)
#else
#define CMPTO_J2K_API_FUNCTION __declspec(dllimport)
#endif
#endif


/**
 * @brief J2K Encoder Context represents an instance of JPEG2000 encoder.
 *
 *
 * */
struct CMPTO_J2K_Enc_Context;

/**
 * @brief J2K Encoder Settings preserves JPEG2000 encoder parameters.
 *
 *
 * */
struct CMPTO_J2K_Enc_Settings;

/**
 * @brief J2K Encoder Image represents an image to be encoded.
 *
 *
 * */
struct CMPTO_J2K_Enc_Image;


/**
 * @brief J2K Decoder Context represents an instance of JPEG2000 decoder.
 *
 *
 * */
struct CMPTO_J2K_Dec_Context;

/**
 * @brief J2K Decoder Settings preserves JPEG2000 decoder parameters.
 *
 *
 * */
struct CMPTO_J2K_Dec_Settings;

/**
 * @brief J2K Decoder Image represents an image to be decoded.
 *
 *
 * */
struct CMPTO_J2K_Dec_Image;

/**
 * @brief Enumeration of possible error states.
 *
 * Almost every CMPTO_J2K API call returns an error states. Exceptions are
 * calls translating input to a simple message. Each error code can be translated using
 * CMPTO_J2K_Get_Error_Message() .
 *
 * */
enum CMPTO_J2K_Error { 
    CMPTO_J2K_OK = 0,
    CMPTO_J2K_Stopped = 1,
    CMPTO_J2K_Queue_Full = 2,
    CMPTO_J2K_CUDA_Error = 3,
    CMPTO_J2K_Invalid_Argument = 4,
    CMPTO_J2K_NULL_Pointer = 5,
    CMPTO_J2K_Alloc_Error = 6,
    CMPTO_J2K_System_Error = 7,
    CMPTO_J2K_Unknown_Error = 8,
    CMPTO_J2K_Unsupported = 9,
    CMPTO_J2K_Bad_Codestream = 10,
    CMPTO_J2K_Context_Running = 11
};

enum CMPTO_J2K_Data_Format {
    /** 8bit unsigned samples, 3 components, 4:4:4 sampling */
    CMPTO_J2K_444_u8_p012 = 0,
    
    /** 8bit unsigned samples, 3 components, 4:4:4 sampling,
     *  sample order: comp#2 comp#1 comp#0 */
    CMPTO_J2K_444_u8_p210 = 1,
    
    /** 12bit unsigned samples stored in LSBs of 16bit little endian 
     *  values, 3 components, 4:4:4 sampling */
    CMPTO_J2K_444_u12_lsb16le_p012 = 2,
    
    /** 12bit unsigned samples stored in MSBs of 16bit little endian 
     *  values, 3 components, 4:4:4 sampling */
    CMPTO_J2K_444_u12_msb16le_p012 = 3,
    
    /** 10bit unsigned samples stored in LSBs of 16bit little endian 
     *  values, 3 components, 4:4:4 sampling */
    CMPTO_J2K_444_u10_lsb16le_p012 = 4,
    
    /** 10bit unsigned samples stored in MSBs of 16bit little endian 
     *  values, 3 components, 4:4:4 sampling */
    CMPTO_J2K_444_u10_msb16le_p012 = 5,
    
    /** 10bit unsigned samples, 3 components, 4:4:4, one 32bit little 
     *  endian value per pixel, component #0 sample in bits 0-9, component
     *  #1 sample in bits 10-19 and component #2 sample in bits 20-29 */
    CMPTO_J2K_444_u10u10u10_lsb32le_p012 = 6,
    
    /** 10bit unsigned samples, 3 components, 4:4:4, one 32bit big endian 
     *  value per pixel, component #0 sample in bits 22-31, component
     *  #1 sample in bits 12-21 and component #2 sample in bits 2-11 */
    CMPTO_J2K_444_u10u10u10_msb32be_p210 = 7,
    
    /** 10bit unsigned samples, 3 components, 4:2:2 YUV, each 32bit little
     *  endian value contain 3 10bit values and 2 empty bits at the end;
     *  the pattern (U0+1, Y0, V0+1, Y1, U2+3, Y2, V2+3, Y3, U4+5, Y4, V4+5, Y5)
     *  repeats every 4 32bits. Each line padded to 128 bytes. */
    CMPTO_J2K_422_u10_v210 = 8,

    /** 8bit unsigned samples, 3 components, 4:2:2,
     *  order of samples: comp#1 comp#0 comp#2 comp#0 */
    CMPTO_J2K_422_u8_p1020 = 9,
    
    /** 10bit unsigned samples stored in MSBs of 16bit little endian 
     *  values, 3 components, 4:2:2, 
     *  order of samples: comp#1 comp#0 comp#2 comp#0 */
    CMPTO_J2K_422_u10_msb16le_p1020 = 10,
    
    /** 8bit unsigned samples, planar, 3 components, 4:2:2,
     *  planar */
    CMPTO_J2K_422_u8_p0p1p2 = 11,
    
    /** 8bit unsigned samples, planar, 3 components, 4:2:0,
     *  planar */
    CMPTO_J2K_420_u8_p0p1p2 = 12,
    
    /** 8bit unsigned samples, single color component */
    CMPTO_J2K_u8 = 13,
    
    /** 10bit unsigned samples, 3 components, 4:4:4, one 32bit little endian 
     *  value per pixel, component #0 sample in bits 2-11, component
     *  #1 sample in bits 12-21 and component #2 sample in bits 22-31 */
    CMPTO_J2K_444_u10u10u10_msb32le_p012 = 14
};


enum CMPTO_J2K_Tile_Mode {
    CMPTO_J2K_1_Tile = 0,
    CMPTO_J2K_4_Grid_Tiles = 1,
    CMPTO_J2K_4_Stripe_Tiles = 2
};

enum CMPTO_J2K_Feature {
    CMPTO_J2K_MCT = 0,
    CMPTO_J2K_SOP = 1,
    CMPTO_J2K_EPH = 2,
    CMPTO_J2K_TLM = 3,
    CMPTO_J2K_Lossless_Mode = 4,
    CMPTO_J2K_Rate_Control = 5,
    CMPTO_J2K_Component_Tileparts = 6
};

enum CMPTO_J2K_Progression {
    CMPTO_J2K_LRCP = 0,
    CMPTO_J2K_RLCP = 1,
    CMPTO_J2K_RPCL = 2,
    CMPTO_J2K_PCRL = 3,
    CMPTO_J2K_CPRL = 4,
    CMPTO_J2K_CPRL_DCI_4K = 5
};


/**
 * @brief Translates an error code to a message.
 *
 * Almost every CMPTO_J2K API call returns an error states. Exceptions are
 * calls translating input to a simple message. Each error code can be translated using
 * this call. 
 *
 * @return Message explaining the error code
 * @return @verbatim Invalid error code @endverbatim if the code is not specified in CMPTO_J2K_Error
 *
 * */
CMPTO_J2K_API_FUNCTION
const char * 
CMPTO_J2K_Get_Error_Message(
    enum CMPTO_J2K_Error s
);



/************************************************************************/
/************************************************************************/
/******************          ENCODER Settings           *****************/
/************************************************************************/
/************************************************************************/



CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Settings_Create(
    struct CMPTO_J2K_Enc_Context * ctx,
    struct CMPTO_J2K_Enc_Settings ** settings
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Settings_Destroy(
    struct CMPTO_J2K_Enc_Settings * settings
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_DWT_Count(
    struct CMPTO_J2K_Enc_Settings * settings,
    int dwt_count  /* 0 = single resolution, 1 = 2 resolutions, .... */
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Codeblock_Size(
    struct CMPTO_J2K_Enc_Settings * settings,
    int codeblock_width,  /* powers of 2 */
    int codeblock_height  /* powers of 2 */
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Precinct_Size(
    struct CMPTO_J2K_Enc_Settings * settings,
    int precinct_width,  /* powers of 2 */
    int precinct_height  /* powers of 2 */
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Quantization(
    struct CMPTO_J2K_Enc_Settings * settings,
    float quality /* 0.0 = poor quality, 1.0 = full quality */
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Rate_Limit(
    struct CMPTO_J2K_Enc_Settings * settings,
    size_t byte_count
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Enable(
    struct CMPTO_J2K_Enc_Settings * settings,
    enum CMPTO_J2K_Feature feature
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Disable(
    struct CMPTO_J2K_Enc_Settings * settings,
    enum CMPTO_J2K_Feature feature
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Signalize_Bit_Depth(
    struct CMPTO_J2K_Enc_Settings * settings,
    int signalized_bit_depth
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Capabilities(
    struct CMPTO_J2K_Enc_Settings * settings,
    int capabilities
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Progression(
    struct CMPTO_J2K_Enc_Settings * settings,
    enum CMPTO_J2K_Progression progression
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Tile_Mode(
    struct CMPTO_J2K_Enc_Settings * settings,
    enum CMPTO_J2K_Tile_Mode tile_mode
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Add_Text_Comment(
    struct CMPTO_J2K_Enc_Settings * settings,
    const char * comment
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Add_Binary_Comment(
    struct CMPTO_J2K_Enc_Settings * settings,
    const void * data,
    size_t size
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error 
CMPTO_J2K_Enc_Settings_Remove_Comments(
    struct CMPTO_J2K_Enc_Settings * settings
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Settings_Get_Status(
    struct CMPTO_J2K_Enc_Settings * settings,
    const char ** message  /* ignored if NULL */
);



/************************************************************************/
/************************************************************************/
/********************          ENCODER Image           ******************/
/************************************************************************/
/************************************************************************/



CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Image_Get_Source_Data_Ptr(
    struct CMPTO_J2K_Enc_Image * img,
    void ** ptr
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Image_Get_Codestream(
    struct CMPTO_J2K_Enc_Image * img,
    size_t * size,
    void ** ptr
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Image_Get_Status(
    struct CMPTO_J2K_Enc_Image * img,
    const char ** message  /* ignored if NULL */
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Image_Set_Custom_Data(
    struct CMPTO_J2K_Enc_Image * img,
    void * custom_data
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Image_Get_Custom_Data(
    struct CMPTO_J2K_Enc_Image * img,
    void ** ptr
);



/************************************************************************/
/************************************************************************/
/*******************          ENCODER Context           *****************/
/************************************************************************/
/************************************************************************/


/**
 * Checks device suitability for CUDA encoder context.
 * 
 * @param device_idx index of device to be checked
 * @return CMPTO_J2K_OK                if device can be used for encoding
 *         CMPTO_J2K_Unsupported       if not suitable for encoding
 *         CMPTO_J2K_Invalid_Argument  if device index is out of bounds
 *         CMPTO_J2K_CUDA_Error        in case of device querying error
 */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Check_CUDA_Device(
    int device_idx
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Context_Init_CUDA (
    const int * device_indices,
    int device_count,
    struct CMPTO_J2K_Enc_Context ** ctx
);


/* Unblocks all threads waiting in *_Put or *_Get */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Context_Stop (
    struct CMPTO_J2K_Enc_Context * ctx
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Context_Destroy(
    struct CMPTO_J2K_Enc_Context * ctx
);


/** @deprecated Context handles this automatically */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Context_Set_Image_Count(
    struct CMPTO_J2K_Enc_Context * ctx,
    int image_count
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Context_Get_Free_Image(
    struct CMPTO_J2K_Enc_Context * ctx,
    int width,
    int height,
    enum CMPTO_J2K_Data_Format format,
    struct CMPTO_J2K_Enc_Image ** img  /* Set to NULL if encoder stopped */
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Context_Encode_Image(
    struct CMPTO_J2K_Enc_Context * ctx,
    struct CMPTO_J2K_Enc_Image * img,
    struct CMPTO_J2K_Enc_Settings * settings
);


/**
 * Sets 'img' to pointer to encoded image.
 * This call blocks until a image is encoded.
 **/
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Context_Get_Encoded_Image(
    struct CMPTO_J2K_Enc_Context * ctx,
    struct CMPTO_J2K_Enc_Image ** img /* Set to NULL if encoder stopped */
);


/**
 * Sets 'img' to pointer to encoded image or to NULL if no image has been
 * encoded from last call. This call does not block.
 **/
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Context_Try_Get_Encoded_Image(
    struct CMPTO_J2K_Enc_Context * ctx,
    struct CMPTO_J2K_Enc_Image ** img
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Context_Return_Unused_Image(
    struct CMPTO_J2K_Enc_Context * ctx,
    struct CMPTO_J2K_Enc_Image * img
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Enc_Context_Get_Status(
    struct CMPTO_J2K_Enc_Context * ctx,
    const char ** message /* ignored if NULL */
);




/************************************************************************/
/************************************************************************/
/********************          DECODER Image           ******************/
/************************************************************************/
/************************************************************************/




/**
 * @brief Gets pointer to buffer, where codestream should be copied.
 * 
 * Either gets pointer to buffer which was previously set by a call to 
 * CMPTO_J2K_Dec_Image_Set_Codestream_Ptr, or gets pointer to buffer 
 * from decoder's internal buffer pool. Buffers from pool are managed 
 * by the context. Application-provided buffers are not managed by 
 * the context.
 *
 * @param img image instance pointer
 * @param ptr pointer to variable to be replaced with codestream buffer pointer
 * @return CMPTO_J2K_OK if OK, or some error code otherwise
 **/
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Get_Codestream_Ptr(
    struct CMPTO_J2K_Dec_Image * img,
    void ** ptr
);

/**
 * @brief Sets pointer to application-managed buffer which contains 
 * (or will contain) the codestream. 
 * 
 * The pointer should point to page-locked memory allocated using CUDA API
 * functions. Pointer is used only for next decoding and is forgotten 
 * when image is returned back to the deocder as an unused image 
 * or when other pointer is set to the image using this function.
 * Decoder-managed buffer is used for codestream if this function is not 
 * called (accessible using CMPTO_J2K_Dec_Image_Get_Codestream_Ptr).
 *
 * @param img image instance pointer
 * @param ptr pointer to application-managed codestream buffer
 * @return CMPTO_J2K_OK if OK, or some error code otherwise
 **/
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Set_Codestream_Ptr(
    struct CMPTO_J2K_Dec_Image * img,
    void * ptr
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Get_Info (
    struct CMPTO_J2K_Dec_Image * img,
    int * component_count,    /* ignored if null */
    int * width,              /* ignored if null */
    int * height,             /* ignored if null */
    int * capabilities        /* ignored if null */
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Get_Component_Info (
    struct CMPTO_J2K_Dec_Image * img,
    int component_index,      /* zero-based color component index */
    int * bit_depth,          /* ignored if null */
    int * sampling_factor_x,  /* ignored if null */
    int * sampling_factor_y,  /* ignored if null */
    int * is_signed           /* ignored if null */
);

/**
 * @brief Gets pointer to buffer with decoded data.
 * 
 * Either returns a pointer to the application-managed buffer set by 
 * a call to function CMPTO_J2K_Dec_Image_Set_Decoded_Data_Ptr or returns 
 * a pointer to buffer managed internally by the decoding context.
 *
 * @param img image instance pointer
 * @param data pointer to variable to be replaced with decoded data pointer
 * @return CMPTO_J2K_OK if OK, error code otherwise
 **/
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Get_Decoded_Data_Ptr(
    struct CMPTO_J2K_Dec_Image * img,
    void ** data
);

/**
 * @brief Sets pointer to buffer for decoded data.
 * 
 * Sets a pointer to the application-managed buffer for decoded samples.
 * Size of the buffer is checked during decoding, and an error is reported 
 * for the image if the size is insufficient. Provided pointer should 
 * point to page-locked memory allocated using CUDA API functions or to 
 * device memory. The device memory option is currently restricted to 
 * single GPU decoding.
 * Pointer is valid for single image decoding and is forgotten when image 
 * is returned back to decoder as an unused image or if replaced by 
 * other pointer by this function.
 * Decoder-managed host buffer is used for decoded samples if this 
 * function is not called before submitting the image for decoding.
 *
 * @param img image instance pointer
 * @param data pointer to buffer for decoded samples
 * @param capacity capacity of the buffer (checked during decoding)
 * @param is_in_device_mem nonzero if pointer points to device memory,
 *                         0 for host memory
 * @return CMPTO_J2K_OK if OK, error code otherwise
 **/
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Set_Decoded_Data_Ptr(
    struct CMPTO_J2K_Dec_Image * img,
    void * data,
    size_t capacity, 
    int is_in_device_mem
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Get_Status(
    struct CMPTO_J2K_Dec_Image * img,
    const char ** message /* ignored if NULL */
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Set_Custom_Data(
    struct CMPTO_J2K_Dec_Image * img,
    void * custom_data
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Get_Custom_Data(
    struct CMPTO_J2K_Dec_Image * img,
    void ** ptr
);

/**
 * Copies a block of custom data into image's internal buffer managed 
 * by decoder. Data remain valid unless replaced by another call 
 * to this function or unless unused image is returned to the decoder.
 * The internal buffer automatically expands according to required data 
 * block size.
 * 
 * @param img  the image instance to be anotated with custom data
 * @param src_ptr  pointer to data block
 * @param size  size of the block (in bytes)
 */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Set_Custom_Data_Block(
    struct CMPTO_J2K_Dec_Image * img,
    const void * src_ptr,
    size_t size
);

/**
 * Copies a block of previously set custom data from image's internal 
 * buffer.
 * 
 * @param img  the image instance with custom data in it
 * @param dest_ptr  pointer to destination data block
 * @param size  size of the block (invalid argument is indicated if 
 *              size is larger than data stored in the image)
 */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Get_Custom_Data_Block(
    struct CMPTO_J2K_Dec_Image * img,
    void * dest_ptr,
    size_t size
);

/**
 * Gets size of custom data block assigned to the image. Initially 
 * (after getting free image from the context), the size is 0.
 * 
 * @param img  the image instance
 * @param size_out  pointer to variable for custom data size
 */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Image_Get_Custom_Data_Block_Size(
    struct CMPTO_J2K_Dec_Image * img,
    size_t * size_out
);



/************************************************************************/
/************************************************************************/
/*******************         DECODER Settings           *****************/
/************************************************************************/
/************************************************************************/


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Settings_Create(
    struct CMPTO_J2K_Dec_Context * ctx,
    struct CMPTO_J2K_Dec_Settings ** settings
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Settings_Destroy(
    struct CMPTO_J2K_Dec_Settings * settings
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Settings_Data_Format(
    struct CMPTO_J2K_Dec_Settings * settings,
    enum CMPTO_J2K_Data_Format data_format
);

/**
 * @brief Sets planar output data format independent of sampling factors.
 * 
 * Each sample is stored either in 8bit unsigned integer if bit depth 
 * is not greater than 8 or in a 16bit unsigned little endian integer 
 * if bit depth is greater than 8. All samples of component #0 are 
 * stored first, in raster order, with no padding. Then, next components 
 * samples follow, if present.) This function is an alternative to 
 * CMPTO_J2K_Dec_Settings_Data_Format - can be used to specify different 
 * output data formats than the predefined ones.
 * @param bit_depth  required output bit depth of all decoded samples, 
 *                   independent of bit depth of components of the image
 *                   (min 1, max 16, both limits inclusive)
 * @param msb        nonzero if each sample value should be stored in most
 *                   significant bits of the type (8bit or 16bit integer),
 *                   or 0 for sample values to be stored in least 
 *                   significant bits (unused bits are set to 0 it present)
 * @return CMPTO_J2K_OK if OK, error code otherwise
 */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Settings_Data_Planar(
    struct CMPTO_J2K_Dec_Settings * settings,
    int bit_depth,
    int msb
);

CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Settings_Get_Status(
    struct CMPTO_J2K_Dec_Settings * settings,
    const char ** message  /* ignored if NULL */
);



/************************************************************************/
/************************************************************************/
/*******************          DECODER Context           *****************/
/************************************************************************/
/************************************************************************/



/**
 * Checks device suitability for CUDA decoder context.
 * 
 * @param device_idx index of device to be checked
 * @return CMPTO_J2K_OK                if device can be used for decoding
 *         CMPTO_J2K_Unsupported       if not suitable for decoding
 *         CMPTO_J2K_Invalid_Argument  if device index is out of bounds
 *         CMPTO_J2K_CUDA_Error        in case of device querying error
 */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Check_CUDA_Device(
    int device_idx
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Init_CUDA (
    const int * device_indices,
    int device_count,
    struct CMPTO_J2K_Dec_Context ** ctx
);


/**
 * @brief Creates a new decoder context using listed devices.
 * @param device_indices  array of indices of devices to be used for decoding
 * @param device_mem_limits  array of limits of GPU memory (in bytes) for each listed device(0 == no limit)
 * @param device_count  count of device indices (and memory limits)
 * @param ctx  pointer to variable, where new context pointer should be written
 * @return CMPTO_J2K_OK for success, error code otherwise
 */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Init_CUDA_Mem (
    const int * device_indices,
    const size_t * device_mem_limits,
    int device_count,
    struct CMPTO_J2K_Dec_Context ** ctx
); 


/* Unblocks all threads waiting in *_Put or *_Get */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Stop (
    struct CMPTO_J2K_Dec_Context * ctx
);


/** @deprecated Context handles this automatically */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Set_Image_Count(
    struct CMPTO_J2K_Dec_Context * ctx,
    int image_count
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Destroy(
    struct CMPTO_J2K_Dec_Context * ctx
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Get_Free_Image(
    struct CMPTO_J2K_Dec_Context * ctx,
    size_t codestream_size,
    struct CMPTO_J2K_Dec_Image ** img
);


/**
 * @brief Nonblocking version of CMPTO_J2K_Dec_Context_Get_Free_Image.
 * 
 * Variable pointed to by "img" pointer is overwritten with the image 
 * pointer if free image is available or with NULL if no free images 
 * are available now.
 *
 * @param ctx  decoder context instance pointer
 * @param codestream_size  size of decoded codestream
 * @param img  pointer to variable to hold the image pointer 
 *             or NULL if there are not any images yet
 **/
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Try_Get_Free_Image(
    struct CMPTO_J2K_Dec_Context * ctx,
    size_t codestream_size,
    struct CMPTO_J2K_Dec_Image ** img
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Decode_Image(
    struct CMPTO_J2K_Dec_Context * ctx,
    struct CMPTO_J2K_Dec_Image * img,
    struct CMPTO_J2K_Dec_Settings * settings
);

/* 
 * Blocking call sets 'img' to pointer to decoded image 
 **/
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Get_Decoded_Image(
    struct CMPTO_J2K_Dec_Context * ctx,
    struct CMPTO_J2K_Dec_Image ** img  /* Set to NULL if encoder stopped */
);


/* 
 * Non-blocking call sets 'img' to pointer to decoded image 
 * or NULL if none is decoded.
 **/
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Try_Get_Decoded_Image(
    struct CMPTO_J2K_Dec_Context * ctx,
    struct CMPTO_J2K_Dec_Image ** img
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Return_Unused_Image(
    struct CMPTO_J2K_Dec_Context * ctx,
    struct CMPTO_J2K_Dec_Image * img
);


CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Dec_Context_Get_Status(
    struct CMPTO_J2K_Dec_Context * ctx,
    const char ** message  /* ignored if NULL */
);



/************************************************************************/
/************************************************************************/
/********************          Version info           *******************/
/************************************************************************/
/************************************************************************/


/**
 * Gets info about the version of the JPEG 2000 codec.
 * @param id string with unique ID of the version
 * @param name name of codec version
 * @param year release year
 * @param month release month (1-12)
 * @param day release day (1-31)
 * @return string containing all the information above
 **/
CMPTO_J2K_API_FUNCTION
const char *
CMPTO_J2K_Get_Version(
    const char ** id,     /* ignored if NULL */
    const char ** name,   /* ignored if NULL */
    int * year,           /* ignored if NULL */
    int * month,          /* ignored if NULL */
    int * day             /* ignored if NULL */
);


/**
 * Gets extended info about the version of the JPEG 2000 codec.
 * @param major_num major version number
 * @param minor_num minor version number
 * @param maint_num maintenance version number
 * @return string containing all version numbers above together with other information 
 **/
CMPTO_J2K_API_FUNCTION
const char *
CMPTO_J2K_Get_Version_Numbers(
    int * major_num,          /* ignored if NULL */
    int * minor_num,          /* ignored if NULL */
    int * maint_num           /* ignored if NULL */
);


/**
 * Gets count of CUDA devices. (Just a wrapper for cudaGetDeviceCount)
 * Device indices range from 0 to count-1 (inclusive).
 * 
 * @param device_count  pointer to variable for CUDA device count
 *                      (set to 0 if not NULL in case of error)
 * @return CMPTO_J2K_OK            if device count queried correctly
 *         CMPTO_J2K_CUDA_Error    if device count cannot be queried
 *         CMPTO_J2K_NULL_Pointer  if null pointer given
 */
CMPTO_J2K_API_FUNCTION
enum CMPTO_J2K_Error
CMPTO_J2K_Get_CUDA_Device_Count(
    int * device_count
);



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
