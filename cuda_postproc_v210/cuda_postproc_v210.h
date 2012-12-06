/**
 * @file    cuda_postproc_v210.h
 * @author  Martin Jirman (jirman@cesnet.cz)
 * @brief   interface of basic image postprocessing CUDA implementation with
 *          v210 output format (http://wiki.multimedia.cx/index.php?title=V210)
 */

#ifndef CUDA_POSTPROC_V210_H
#define CUDA_POSTPROC_V210_H

#ifdef __cplusplus
extern "C" {
#endif


/** Output color component selection type */
enum cuda_postproc_v210_color 
{
    POSTPROC_RGB,  /* all color components included */
    POSTPROC_RG,   /* red and green components only */
    POSTPROC_RB,   /* red and blue components only */
    POSTPROC_GB,   /* green and blue components only */
    POSTPROC_R,    /* red component only */
    POSTPROC_G,    /* blue component only */
    POSTPROC_B,    /* blue component only */
    POSTPROC_GRAY, /* grayscale (luma only) */
    POSTPROC_BW    /* only black or white pixels with specified threshold */
};


/** Type of CUDA v210 postprocessor instance */
struct cuda_postproc_v210;


/**
 * Creates a new instance of CUDA v210 preprocessor for specified output size.
 * @param gpu_idx  index of CUDA device
 * @param out_size_x  output width in pixels
 * @param out_size_y  output height in pixels
 * @param max_in_size_x  maximal input width
 * @param max_in_size_y  maximal input height
 * @param verbose  nonzero for timing info and error descriptions 
 *                 to be printed to stdout, 0 = quiet
 * @return either new instance of the preprocessor or null if error occured
 */
struct cuda_postproc_v210 *
cuda_postproc_v210_create
(
    int gpu_idx,
    int out_size_x,
    int out_size_y,
    int max_in_size_x,
    int max_in_size_y,
    int verbose
);


/**
 * Runs postprocessing using specified instance. Returns after the result 
 * is written in the output buffer (a blocking call).
 * Must be called in the same CUDA context (e.g. same host thread), 
 * where correpsonding cuda_postproc_v21_create was called.
 * Both input and output buffers should be in page-locked memory allocated 
 * using cudaMalloc for best performance. Unfortunately, this call is NOT 
 * reentrant with respect to CUDA contexts (e.g. one cannot run two instances 
 * of postprocessor at the same time on the same GPU), because textures cannot
 * be allocated at run-time in CUDA.
 * @param postproc  pointer to postprocessor instance
 * @param src_size_x  input image width
 * @param src_size_y  input image height
 * @param begin_x  position of output's left boundary in src image (0.0 to 1.0)
 * @param end_x  position of output's right boundary in src image (0.0 to 1.0)
 * @param begin_y  position of output's top boundary in src image (0.0 to 1.0)
 * @param end_y  position of output's bottom boundary in src image (0.0 to 1.0)
 * @param color  type of color selection
 * @param threshold_bw  threshold for B&W color selection 
 *                       (0.0 black to 1.0 white, ignored if color != B&W)
 * @param src  source image pointer (in host memory)
 * @param out  output buffer pointer (in host memory)
 * @return 0 for success, nonzero for failure
 */
int
cuda_postproc_v210_run
(
    struct cuda_postproc_v210 * postproc,
    int src_size_x,
    int src_size_y,
    float begin_x,
    float end_x,
    float begin_y,
    float end_y,
    enum cuda_postproc_v210_color color,
    float threshold_bw,
    const void * src,
    void * out
);


/**
 * Releases all resources associated with given instance of postprocessor.
 * Must be called in the same CUDA context (e.g. same host thread), 
 * where correpsonding cuda_postproc_v210_create was called.
 * @param postproc  postprocessor instance to be destroyed
 */
void
cuda_postproc_v210_destroy
(
    struct cuda_postproc_v210 * postproc
);


/**
 * Computes byte size of v210 compressed image with specified width and height.
 * @param size_x  image width in pixels
 * @param size_y  image height in pixels
 * @return size (in bytes) of v210 compressed image with given width and height
 */
int
cuda_postproc_v210_out_size
(
    int size_x,
    int size_y
);


#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* CUDA_POSTPROC_V210_H */
