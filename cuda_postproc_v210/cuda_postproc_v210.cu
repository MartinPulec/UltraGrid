/// @file    cuda_postproc_v210.cu
/// @author  Martin Jirman (jirman@cesnet.cz)
/// @brief   basic image postprocessing CUDA implementation with v210 output 
///          format (http://wiki.multimedia.cx/index.php?title=V210)


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include "cuda_postproc_v210.h"


static texture<ushort4, cudaTextureType2D, cudaReadModeNormalizedFloat> tex;

/// Type of CUDA v210 postprocessor instance
struct cuda_postproc_v210 {
    bool verbose;
    void * tex_buffer;
    void * io_buffer;
    size_t out_size;
    size_t tex_pitch;
    int out_size_x;
    int out_size_y;
    dim3 t_postproc;
    dim3 g_postproc;
    int grp_count_x;
    cudaStream_t stream;
};



/// Divide and round up.
static int div_rnd_up(const int n, const int d) {
    return (n + d - 1) / d;
}



/// Creates a new instance of CUDA v210 preprocessor for specified output size.
/// @param gpu_idx  index of CUDA device
/// @param out_size_x  output width in pixels
/// @param out_size_y  output height in pixels
/// @param max_in_size_x  maximal input width
/// @param max_in_size_y  maximal input height
/// @param verbose  nonzero for timing info and error descriptions 
///                 to be printed to stdout, 0 = quiet
/// @return either new instance of the preprocessor or null if error occured
cuda_postproc_v210 * cuda_postproc_v210_create(int gpu_idx,
                                               int out_size_x,
                                               int out_size_y,
                                               int max_in_size_x,
                                               int max_in_size_y,
                                               int verbose) {
    // select GPU and get info about it
    if(cudaSuccess != cudaSetDevice(gpu_idx)) {
        printf("CUDA postproc ERROR: cannot set device #%d.\n", gpu_idx);
        return 0;
    }
    cudaDeviceProp info;
    if(cudaSuccess != cudaGetDeviceProperties(&info, gpu_idx)) {
        printf("CUDA postproc ERROR: cannot get device #%d info.\n", gpu_idx);
        return 0;
    }
    
    // check GPU capabilities
    if(info.major < 2) {
        printf("CUDA postproc ERROR: Needs CC >= 2.0 (device #%d)\n", gpu_idx);
        return 0;
    }
    
    // allocate main structure
    cuda_postproc_v210 * const p
            = (cuda_postproc_v210*)malloc(sizeof(cuda_postproc_v210));
    if(0 == p) {
        printf("CUDA postproc ERROR: malloc error\n");
        return 0;
    }
    
    // initialize main structure
    memset(p, 0, sizeof(cuda_postproc_v210));
    p->verbose = verbose;
    p->out_size = cuda_postproc_v210_out_size(out_size_x, out_size_y);
    p->out_size_x = out_size_x;
    p->out_size_y = out_size_y;
    p->grp_count_x = p->out_size / (out_size_y * 16);
    p->t_postproc = dim3(64, 8);
    p->g_postproc = dim3(div_rnd_up(p->grp_count_x, p->t_postproc.x),
                         div_rnd_up(p->out_size_y, p->t_postproc.y));
    
    // IO buffer size
    const size_t in_size = max_in_size_x * max_in_size_y * 4;
    const size_t io_size = in_size > p->out_size ? in_size : p->out_size;
    
    // allocate texture buffer
    if(cudaSuccess != cudaMallocPitch(
            &p->tex_buffer,
            &p->tex_pitch,
            max_in_size_x * 8, // 8 bytes per pixel
            max_in_size_y
    )) {
        if(verbose) {
            printf("CUDA postproc ERROR: cudaMallocPitch\n");
        }
        goto error;
    }
    
    // allocate i/o buffer
    if(cudaSuccess != cudaMalloc(&p->io_buffer, io_size)) {
        if(verbose) {
            printf("CUDA postproc ERROR: cudaMalloc %d bytes\n", (int)io_size);
        }
        goto error;
    }
    
    // create separate CUDA stream
    if(cudaSuccess != cudaStreamCreate(&p->stream)) {
        if(verbose) {
            printf("CUDA postproc ERROR: cudaStreamCreate\n");
        }
        goto error;
    }
    
    // success => return the instance
    return p;
    
error:
    // error => destroy partially initialized instance and return 0
    cuda_postproc_v210_destroy(p);
    return 0;
}



/// Converts 10bit sample to 16 bit sample.
__device__ static unsigned int upscale(const unsigned int n) {
    return (n << 6) + (n >> 4);
}



/// Unpacks 10bit RGB data to 16bit RGBA texture buffer.
template <int STEPS_Y>
__global__ static void unpack_kernel(const int src_size_x,
                                     const int src_size_y,
                                     const int src_pitch,
                                     const int out_pitch,
                                     const void * src_ptr,
                                     void * out_ptr) {
    // get coordinates of the first unpacked pixel
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = STEPS_Y * (threadIdx.y + blockIdx.y * blockDim.y);
    
    // unpack (if not out of image boundary)
    if(x < src_size_x && y < src_size_y) {
        // update src and out pointers to point to first input/output pixel
        const char * src = (const char *)src_ptr + y * src_pitch + x * 4;
        char * out = (char *)out_ptr + y * out_pitch + x * 8;
        
        for(int i = min(src_size_y - y, STEPS_Y); i--; ) {
            // load packed source RGB and advance src pointer
            const unsigned int src_pix = *(unsigned int*)src;
            src += src_pitch;
            
            // unpack to RGB
            const unsigned int b = upscale(0x3FF & src_pix);
            const unsigned int g = upscale(0x3FF & (src_pix >> 10));
            const unsigned int r = upscale(0x3FF & (src_pix >> 20));
            
            // save packed to 8 bytes and advance output pointer
            *(uint2*)out = make_uint2(r + (g << 16), b);
            out += out_pitch;
        }
    }
}



/// Packs 3 10bit samples into the 32 bit uint.
__device__ static unsigned int pack_10bit(float lo, float mid, float hi) {
    const unsigned int l = __saturatef(lo) * 1023.0f;
    const unsigned int m = __saturatef(mid) * 1023.0f;
    const unsigned int h = __saturatef(hi) * 1023.0f;
    return l + (m << 10) + (h << 20);
}



/// Samples texture, converts samples to YUV and saves them in v210 format.
/// @param tex_step_x  difference between texture x-coordinates of two 
///                    horizontally neighboring source pixels
/// @param tex_step_y  difference between texture y-coordinates of two 
///                    vertically neighboring source pixels
/// @param tex_base_x  offset of x texture coordinates
/// @param tex_base_y  offset of y texture coordinates
/// @param grp_count_x  count of 6pixel groups in each output line
/// @param grp_count_y  output line count
/// @param out  output buffer pointer
__global__ static void postproc_kernel(const float tex_step_x,
                                       const float tex_step_y,
                                       const float tex_base_x,
                                       const float tex_base_y,
                                       const int grp_count_x,
                                       const int grp_count_y,
                                       uint4 * const out,
                                       const float red_mul,
                                       const float green_mul,
                                       const float blue_mul,
                                       const float chroma_mul,
                                       const bool luma_threshold,
                                       const float inv_threshold_bw) {
    // get coordinates of this thread's group of 6 pixels
    const int grp_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int grp_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // possibly stop if out of bounds
    if(grp_y >= grp_count_y || grp_x >= grp_count_x) {
        return;
    }
    
    // output group index
    const int grp_out_idx = grp_y * grp_count_x + grp_x;
    
    // load and convert samples to YUV
    const float tex_y = tex_base_y + grp_y * tex_step_y;
    float y[6], u[6], v[6];
    #pragma unroll
    for(int x = 0; x < 6; x++) {
        // load from texture with linear interpolation
        const float tex_x = tex_base_x + (grp_x * 6 + x) * tex_step_x;
        const float4 src = tex2D(tex, tex_x, tex_y);
        
        // convert to YUV
        const float r = src.x * red_mul;
        const float g = src.y * green_mul;
        const float b = src.z * blue_mul;
        y[x] = 0.183f * r + 0.614f * g + 0.062f * b + 0.0627450980392f;
        u[x] = (-0.101f * r - 0.338f * g + 0.439f * b) * chroma_mul + 0.5f;
        v[x] = (0.439f * r - 0.399f * g - 0.040f * b) * chroma_mul + 0.5f;
        if(luma_threshold) {
            y[x] = floorf(y[x] * inv_threshold_bw);
        }
    }
    
    // pack YUV samples to output 16 bytes
    uint4 res;
    res.x = pack_10bit((u[0] + u[1]) * 0.5f, y[0], (v[0] + v[1]) * 0.5f);
    res.y = pack_10bit(y[1], (u[2] + u[3]) * 0.5f, y[2]);
    res.z = pack_10bit((v[2] + v[3]) * 0.5f, y[3], (u[4] + u[5]) * 0.5f);
    res.w = pack_10bit(y[4], (v[4] + v[5]) * 0.5f, y[5]);
    
    // save packed samples
    out[grp_out_idx] = res;
}



/// Gets current time in milliseconds.
static double get_time_ms() {
    timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec * 1000.0 + tv.tv_usec * 0.001;
}



/// Runs postprocessing using specified instance. Returns after the result 
/// is written in the output buffer (a blocking call).
/// Must be called in the same CUDA context (e.g. same host thread), 
/// where correpsonding cuda_postproc_v21_create was called.
/// Both input and output buffers should be in page-locked memory allocated 
/// using cudaMalloc for best performance. Unfortunately, this call is NOT 
/// reentrant with respect to CUDA contexts (e.g. one cannot run two instances 
/// of postprocessor at the same time on the same GPU), because textures cannot
/// be allocated at run-time in CUDA.
/// @param postproc  pointer to postprocessor instance
/// @param src_size_x  input image width
/// @param src_size_y  input image height
/// @param begin_x  position of output's left boundary in src image (0.0 to 1.0)
/// @param end_x  position of output's right boundary in src image (0.0 to 1.0)
/// @param begin_y  position of output's top boundary in src image (0.0 to 1.0)
/// @param end_y  position of output's bottom boundary in src image (0.0 to 1.0)
/// @param color  type of color selection
/// @param threshold_bw  threshold for B&W color selection 
///                       (0.0 black to 1.0 white, ignored if color != B&W)
/// @param src  source image pointer (in host memory)
/// @param out  output buffer pointer (in host memory)
/// @return 0 for success, nonzero for failure
int cuda_postproc_v210_run(struct cuda_postproc_v210 * postproc,
                           int src_size_x,
                           int src_size_y,
                           float begin_x,
                           float end_x,
                           float begin_y,
                           float end_y,
                           enum cuda_postproc_v210_color color,
                           float threshold_bw,
                           const void * src,
                           void * out) {
    // check arguments
    if(0 == postproc) {
        printf("CUDA postproc ERROR: postprocessor pointer == NULL.\n");
        return -1;
    }
    if(0 == src || 0 == out || src_size_x < 1 || src_size_y < 1) {
        if(postproc->verbose) {
            printf("CUDA postproc ERROR: null buffer or non-positive size\n");
        }
        return -2;
    }
    
    // possibly start time measurement
    const double begin_time_ms = postproc->verbose ? get_time_ms() : 0.0;
    
    // copy the input to GPU
    if(cudaSuccess != cudaMemcpyAsync(
            postproc->io_buffer,
            src,
            4 * src_size_x * src_size_y,
            cudaMemcpyHostToDevice,
            postproc->stream
    )) {
        if(postproc->verbose) {
            printf("CUDA postproc ERROR: memcpy H->D\n");
        }
        return -3;
    }
    
    // issue first kernel (unpacks source image to buffer bound to the texture)
    enum { STEPS_Y = 32 };
    const dim3 t_unpack(128, 4);
    const dim3 g_unpack(div_rnd_up(src_size_x, t_unpack.x),
                        div_rnd_up(src_size_y, t_unpack.y * STEPS_Y));
    unpack_kernel<STEPS_Y><<<g_unpack, t_unpack, 0, postproc->stream>>>(
            src_size_x,
            src_size_y,
            4 * src_size_x,
            postproc->tex_pitch,
            postproc->io_buffer,
            postproc->tex_buffer
    );
    
    // setup texture reference and bind texture buffer to texture reference
    tex.normalized = 1;
    tex.filterMode = cudaFilterModeLinear;
    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.addressMode[2] = cudaAddressModeBorder;
    const cudaError_t status = cudaBindTexture2D(
            0,
            tex,
            postproc->tex_buffer,
            src_size_x, 
            src_size_y,
            postproc->tex_pitch
    );
    if(cudaSuccess != status) {
        if(postproc->verbose) {
            printf("CUDA postproc ERROR: texture bind (%s)\n",
                   cudaGetErrorString(status));
        }
        return -5;
    }
    
    // select color components to be kept
    float red_mul = 1.0f;
    float green_mul = 1.0f;
    float blue_mul = 1.0f;
    float chroma_mul = 1.0f;
    bool luma_threshold = false;
    switch(color) {
        case POSTPROC_RGB:
            // nothing to be done
            break;
        case POSTPROC_RG:
            blue_mul = 0.0f;
            break;
        case POSTPROC_RB:
            green_mul = 0.0f;
            break;
        case POSTPROC_GB:
            red_mul = 0.0f;
            break;
        case POSTPROC_R:
            blue_mul = 0.0f;
            green_mul = 0.0f;
            break;
        case POSTPROC_G:
            blue_mul = 0.0f;
            red_mul = 0.0f;
            break;
        case POSTPROC_B:
            red_mul = 0.0f;
            green_mul = 0.0f;
            break;
        case POSTPROC_GRAY:
            chroma_mul = 0.0f;
            break;
        case POSTPROC_BW:
            chroma_mul = 0.0f;
            luma_threshold = true;
            break;
        default:
            if(postproc->verbose) {
                printf("CUDA postproc ERROR: unknown mode %d\n", (int)color);
            }
            return -4;
    }
    
    // issue second kernel, which scales and transforms the image, 
    // packing result into v210 format
    postproc_kernel
        <<<postproc->g_postproc, postproc->t_postproc, 0, postproc->stream>>>(
            (end_x - begin_x) / postproc->out_size_x,
            (end_y - begin_y) / postproc->out_size_y,
            begin_x + 0.5 / postproc->out_size_x,
            begin_y + 0.5 / postproc->out_size_y,
            postproc->grp_count_x,
            postproc->out_size_y,
            (uint4*)postproc->io_buffer,
            red_mul,
            green_mul,
            blue_mul,
            chroma_mul,
            luma_threshold,
            1.0 / threshold_bw
    );
    
    // copy result back
    if(cudaSuccess != cudaMemcpyAsync(
            out,
            postproc->io_buffer,
            postproc->out_size,
            cudaMemcpyDeviceToHost,
            postproc->stream
    )) {
        if(postproc->verbose) {
            printf("CUDA postproc ERROR: memcpy D->H\n");
        }
        return -6;
    }
    
    // Check CUDA status after all the calls and wait for memcpy
    if(cudaSuccess != cudaStreamSynchronize(postproc->stream)) {
        if(postproc->verbose) {
            printf("CUDA postproc ERROR: stream sync (or kernel calls)\n");
        }
        return -7;
    }
    
    // possibly show current time
    if(postproc->verbose) {
        const double end_time_ms = get_time_ms();
        printf("Postprocessing time %.2f ms (input size %dx%d).\n",
               end_time_ms - begin_time_ms, src_size_x, src_size_y);
    }
    
    // indicate success
    return 0;
}



/// Releases all resources associated with given instance of postprocessor.
/// Must be called in the same CUDA context (e.g. same host thread), 
/// where correpsonding cuda_postproc_v210_create was called.
/// @param postproc  postprocessor instance to be destroyed
void cuda_postproc_v210_destroy(struct cuda_postproc_v210 * postproc) {
    if(postproc) {
        if(postproc->stream) {
            cudaStreamDestroy(postproc->stream);
        }
        if(postproc->io_buffer) {
            cudaFree(postproc->io_buffer);
        }
        if(postproc->tex_buffer) {
            cudaFree(postproc->tex_buffer);
        }
        free(postproc);
    }
}



/// Computes byte size of v210 compressed image with specified width and height.
/// @param size_x  image width in pixels
/// @param size_y  image height in pixels
/// @return size (in bytes) of v210 compressed image with given width and height
int cuda_postproc_v210_out_size(int size_x, int size_y) {
    // 6pixel group count per each line
    const int line_grps = (size_x + 5) / 6;
    
    // byte count per line aligned to multiples of 128
    const int line_bytes = (line_grps * 16 + 127) & ~127;
    
    // return total byte count in all lines
    return line_bytes * size_y;
}

