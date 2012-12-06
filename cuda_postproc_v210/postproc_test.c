/**
 * @file    postproc_test.c
 * @author  Martin Jirman (jirman@cesnet.cz)
 * @brief   Test app for CUDA postprocessor with v210 output.
 */


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "cuda_postproc_v210.h"


#define MAX_PATH_LEN (16 * 1024)


/* global parameters */
float begin_x, end_x, begin_y, end_y;
void *in_buffer, *out_buffer;
int out_size_x, out_size_y, in_size_x, in_size_y;
size_t out_size;
struct cuda_postproc_v210 * postproc = 0;


/** Shows correct usage and exits. */
static void usage(const char * app, const char * msg) {
    printf("ERROR: %s\n", msg);
    printf("Usage: %s in_width in_height out_width out_height "
           "center_x center_y scale_in_percents input.10b\n", app);
    printf("Input format: 4 bytes per pixel, little endian:\n");
    printf("   bits 0 - 9 blue\n");
    printf("   bits 10 - 19 green\n");
    printf("   bits 20 - 29 red\n");
    printf("   bits 30 and 31 unused\n");
    exit(-1);
}


/** Runs postprocessing and saves the result. */
static void process(enum cuda_postproc_v210_color color, float threshold,
                    const char * path_base, const char * path_suffix) {
    char out_path[MAX_PATH_LEN + 1];
    FILE * out_file = 0;
    
    /* run postprocessing */
    if(cuda_postproc_v210_run(
            postproc,
            in_size_x,
            in_size_y,
            begin_x,
            end_x, 
            begin_y, 
            end_y, 
            color, 
            threshold, 
            in_buffer, 
            out_buffer
    ))
    {
        printf("ERROR: postprocessing error (%s).\n", path_suffix);
        return;
    }
    
    /* compose output filename and save */
    snprintf(out_path, MAX_PATH_LEN, "%s.%s.v210", path_base, path_suffix);
    if(out_file = fopen(out_path, "w"))
    {
        if(1 != fwrite(out_buffer, out_size, 1, out_file))
            printf("ERROR: cannot write into output file %s.\n", out_path);
    } 
    else 
        printf("ERROR: cannot open output file %s.\n", out_path);
    fclose(out_file);
}


/** Checks arguments, loads input and saves few postprocessed outputs. */
int main(int argn, char ** argv) {
    int center_x, center_y, scale_percents;
    float scale;
    const char * in_path;
    size_t in_size;
    FILE * in_file = 0;
    
    /* extract and check arguments */
    if(argn != 9)
        usage(argv[0], "Wrong argument count.");
    in_size_x = atoi(argv[1]);
    in_size_y = atoi(argv[2]);
    out_size_x = atoi(argv[3]);
    out_size_y = atoi(argv[4]);
    center_x = atoi(argv[5]);
    center_y = atoi(argv[6]);
    scale_percents = atoi(argv[7]);
    in_path = argv[8];
    if(in_size_x <= 0 || in_size_y <= 0 || out_size_x <= 0 || out_size_y <= 0)
        usage(argv[0], "Width and height must be positive integers.");
    
    /* compute viewport parameters */
    scale = 100.0f / scale_percents;
    begin_x = center_x / (float)in_size_x - 0.5f * scale;
    begin_y = center_y / (float)in_size_y - 0.5f * scale;
    end_x = begin_x + scale;
    end_y = begin_y + scale;
    
    /* allocate buffers */
    in_size = in_size_x * in_size_y * 4;
    out_size = cuda_postproc_v210_out_size(out_size_x, out_size_y);
    if(cudaSuccess != cudaMallocHost(&in_buffer, in_size))
        usage(argv[0], "Cannot allocate input buffer.");
    if(cudaSuccess != cudaMallocHost(&out_buffer, out_size))
        usage(argv[0], "Cannot allocate output buffer.");
    
    /* prepare postprocessor instance */
    postproc = cuda_postproc_v210_create(0, out_size_x, out_size_y,
                                         in_size_x, in_size_y, 1);
    if(0 == postproc)
        usage(argv[0], "Postprocessor allocation error.");
    
    /* load input image */
    in_file = fopen(in_path, "r");
    if(0 == in_file)
        usage(argv[0], "Cannot open input file.");
    if(0 != fseek(in_file, -in_size, SEEK_END))
        usage(argv[0], "Cannot seek input file.");
    if(1 != fread(in_buffer, in_size, 1, in_file))
        usage(argv[0], "Cannot read enough bytes from input file.");
    fclose(in_file);
    
    /* save few versions of postprocessed input */
    process(POSTPROC_RGB, 0.0f, in_path, "rgb");
    process(POSTPROC_RG, 0.0f, in_path, "rg");
    process(POSTPROC_RB, 0.0f, in_path, "rb");
    process(POSTPROC_GB, 0.0f, in_path, "gb");
    process(POSTPROC_R, 0.0f, in_path, "r");
    process(POSTPROC_G, 0.0f, in_path, "g");
    process(POSTPROC_B, 0.0f, in_path, "b");
    process(POSTPROC_GRAY, 0.0f, in_path, "y");
    process(POSTPROC_BW, 0.1f, in_path, "bw10");
    process(POSTPROC_BW, 0.5f, in_path, "bw50");
    process(POSTPROC_BW, 0.9f, in_path, "bw90");
    
    /* cleanup */
    cudaFreeHost(in_buffer);
    cudaFreeHost(out_buffer);
    cuda_postproc_v210_destroy(postproc);
    printf("Done.\n");
    return 0;
}

