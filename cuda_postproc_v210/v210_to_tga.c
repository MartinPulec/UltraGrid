/// 
/// @file    v210_to_tga.c
/// @author  Martin Jirman (jirman@cesnet.cz)
/// @brief   Decodes 10bit YUYV v210 encoded image into 8bit RGB image.
///

#define MAX_PATH_SIZE (16 * 1024)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


static void usage(const char * const command, const char * const message) {
    printf("ERROR: %s\n", message);
    printf("Usage: %s width height image.v210\n", command);
    exit(-1);
}


static float unpack(const uint32_t bits, const int shift) {
    return ((bits >> shift) & 1023) * 0.000977517106549f;  // divide by 1023
}


static uint8_t clamp(const float s) {
    const int res = (int)(s * 255.0f);
    if(res < 0)
        return 0;
    if(res > 255)
        return 255;
    return res;
}


/// @return 0 for success, nonzero otherwise
static int convert(FILE * in, FILE * out, const int size_x, int size_y) {
    // groups loaded per line
    const int grps_x = (size_x + 5) / 6;
    
    // number of padding groups to be skipped at the end of each line
    const int padding_grps = (1 + ~grps_x) & 7;
    
    // buffers for input and output group
    uint32_t yuv[4];
    uint8_t bgr[3 * 6];
    float ys[6], us[3], vs[3];
    
    // process all rows
    while(size_y--) {
        // process all groups
        for(int x = size_x; x > 0; x -= 6) {
            // read the group
            if(1 != fread(yuv, sizeof(yuv), 1, in)) {
                return -1;
            }
            
            // unpack
            us[0] = unpack(yuv[0], 0) - 0.5f;
            ys[0] = unpack(yuv[0], 10) - 0.0627450980392f;
            vs[0] = unpack(yuv[0], 20) - 0.5f;
            ys[1] = unpack(yuv[1], 0) - 0.0627450980392f;
            us[1] = unpack(yuv[1], 10) - 0.5f;
            ys[2] = unpack(yuv[1], 20) - 0.0627450980392f;
            vs[1] = unpack(yuv[2], 0) - 0.5f;
            ys[3] = unpack(yuv[2], 10) - 0.0627450980392f;
            us[2] = unpack(yuv[2], 20) - 0.5f;
            ys[4] = unpack(yuv[3], 0) - 0.0627450980392f;
            vs[2] = unpack(yuv[3], 10) - 0.5f;
            ys[5] = unpack(yuv[3], 20) - 0.0627450980392f;
            
            // convert yuv to rgb
            for(int i = 6; i--; ) {
                const float y = ys[i];
                const float u = us[i >> 1];
                const float v = vs[i >> 1];
                const float r = 1.164f * y + 1.793f * v;
                const float g = 1.164f * y - 0.534f * v - 0.213f * u;
                const float b = 1.164f * y + 2.115f * u;
                bgr[i * 3 + 2] = clamp(r);
                bgr[i * 3 + 1] = clamp(g);
                bgr[i * 3 + 0] = clamp(b);
            }
            
            // write results
            if(1 != fwrite(bgr, (x > 6 ? 6 : x) * 3, 1, out)) {
                printf("Error writing %d bytes (%d lines remaining).\n",
                       (x > 6 ? 6 : x) * 3, size_y);
                return -2;
            }
        }
        
        // skip padding groups
        for(int i = padding_grps; i--; ) {
            if(1 != fread(yuv, sizeof(yuv), 1, in)) {
                return -3;
            }
        }
    }
    
    return 0;
}


/// Writes the TARGA header for 3 component RGB image.
/// @return 0 for success, nonzero otherwise
static int write_tga_header(FILE * const out, int size_x, int size_y) {
    // compose the header
    uint8_t header[18];
    header[0] = 0; /* no ID field */
    header[1] = 0; /* no color map */
    header[2] = 2; /* uncompressed RGB */
    header[3] = 0; /* ignored if no color map */
    header[4] = 0; /* ignored if no color map */
    header[5] = 0; /* ignored if no color map */
    header[6] = 0; /* ignored if no color map */
    header[7] = 0; /* ignored if no color map */
    *((uint16_t*)(header + 8)) = 0; /* image origin X */
    *((uint16_t*)(header + 10)) = 0; /* image origin Y */
    *((uint16_t*)(header + 12)) = size_x; /* image width */
    *((uint16_t*)(header + 14)) = size_y; /* image height */
    header[16] = 24; /* 24 bits per pixel */
    header[17] = 1 << 5; /* origin in top-left corner */
    
    // write it into the file
    return 1 == fwrite(header, sizeof(header), 1, out) ? 0 : -1;
}


int main(int argn, char ** argv) {
    // check arguments
    if(argn != 4) {
        usage(argv[0], "Incorrect argument count.");
    }
    
    // get and check size
    const int size_x = atoi(argv[1]);
    const int size_y = atoi(argv[2]);
    if(size_x < 1 || size_y < 1) {
        usage(argv[0], "Both width and height must be positive integers.");
    }
    
    // get input and output filenames
    const char * const in_path = argv[3];
    char out_path[MAX_PATH_SIZE + 1];
    snprintf(out_path, MAX_PATH_SIZE, "%s.tga", in_path);
    
    // open input file
    FILE * const in_file = fopen(in_path, "r");
    if(in_file == 0) {
        usage(argv[0], "ERROR: could not open input file for reading.");
    }
    
    // open output file
    FILE * const out_file = fopen(out_path, "w");
    if(out_file == 0) {
        fclose(in_file);
        usage(argv[0], "ERROR: could not open output file for writing.");
    }
    
    // write targa header for 8bit RGB image and converted data
    if(write_tga_header(out_file, size_x, size_y)) {
        printf("ERROR: cannot write header to output file %s.\n", out_path);
    }
    if(convert(in_file, out_file, size_x, size_y)) {
        printf("ERROR: read/write error.\n");
    }

    // close both files before returning
    fclose(in_file);
    fclose(out_file);
    return 0;
}
