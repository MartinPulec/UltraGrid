/**
 * @file    test.c
 * @author  Martin Jirman <jirman@cesnet.cz>
 * @brief   2012 JPEG encoder demo simple test app.
 */


#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "../demo_enc.h"


#define MAX_PATH_LEN (1024 * 16)


/** Chained work item type. */
struct work_item {
    /* next work item pointer */
    struct work_item * next;
    
    /* page-locked buffer */
    void * buffer;
    
    /* output filename */
    char path[MAX_PATH_LEN + 1];
};


/* global encoder pointer */
struct demo_enc * enc = 0;

/* first argument pointer */
const char * prog_name = 0;

/* unused work item stack with locks */
struct work_item * volatile unused = 0;
pthread_mutex_t mutex;
pthread_cond_t cond;

/* image size */
int size_x = -1, size_y = -1;

/* buffers size */
int buffer_size = -1;


/** Shows correct usage and exits */
static void usage(const char * const message) {
    printf("ERROR: %s.\n", message);
    printf("Usage: %s width height file1.dpx [file2.dpx ...]\n", prog_name);
    exit(-1);
}


/** @return time in milliseconds */
static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec * 1000.0 + tv.tv_usec * 0.001;
}


/** Saving thread implementation. */
static void * saving_thread_impl(void * param) {
    int size;
    struct work_item * item;
    FILE * file = 0;
    
    /* continue as long as there are more encoded work items */
    while(1) {
        size = demo_enc_wait(enc, (void**)&item, 0, 0);
        if(0 == size) /* encoder stopped */
            break; 
        if(size < 0)
            printf("Error encoding %s.\n", item->path);
        if(size > 0) {
            /* save the codestream */
            if(file = fopen(item->path, "w")) {
                if(1 == fwrite(item->buffer, size, 1, file))
                    printf("Output file saved OK: %s.\n", item->path);
                else 
                    printf("Can't write to file %s.\n", item->path);
                fclose(file);
            } else 
                printf("Can't open output file %s.\n", item->path);
        }
        
        /* return work item with buffer back to loading thread to be reused */
        pthread_mutex_lock(&mutex);
        item->next = unused;
        unused = item;
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mutex);
    }
    
    /* return value is unused */
    return 0;
}


/** Reads given file and submits it for encoding. */
static void submit_input(const char * const filename, const int dwt_level_count) {
    static int subsampling;
    struct work_item * item = 0;
    FILE * file = 0;
    int load_size = size_x * size_y * 4; /* 4 bytes per pixel */
    
    /* open the file for reading and check the size. */
    if(file = fopen(filename, "r")) {
        /* get file size and set read offset */
        fseek(file, -load_size, SEEK_END);
        if(ftell(file) >= 0) {
            for(subsampling = 0; subsampling <= dwt_level_count; subsampling++) {
                /* get unused work item. */
                pthread_mutex_lock(&mutex);
                while(0 == unused)
                    pthread_cond_wait(&cond, &mutex);
                item = unused;
                unused = item->next;
                pthread_mutex_unlock(&mutex);
                
                /* compose output filename */
                snprintf(item->path, MAX_PATH_LEN, "%s.sub%d.j2k", filename, subsampling);
                
                /* load the file */
                fseek(file, -load_size, SEEK_END);
                if(1 == fread(item->buffer, load_size, 1, file)) {
                    /* show status message */
                    printf("Loaded %d bytes from file %s.\n", load_size, filename);
                    
                    /* submit the work item for encoding */
                    demo_enc_submit(enc, item, item->buffer, buffer_size, 
                                    item->buffer, 0, 1.0f, subsampling);
                } else {
                    /* show error message */
                    printf("Cannot read %d bytes from file %s.\n", load_size, filename);
                    
                    /* return the item back to queue */
                    pthread_mutex_lock(&mutex);
                    item->next = unused;
                    unused = item;
                    pthread_cond_signal(&cond);
                    pthread_mutex_unlock(&mutex);
                }
            }
        } else 
            printf("File too short or not seekable: %s.\n", filename);
        
        /* close the file */
        fclose(file);
    } else 
        printf("Cannot open file %s for reading.\n", filename);
}


/** Traverses given list of arguments and adds them to encoder. */
static void load_input(char ** argv, int dwt_level_count) {
    char path[MAX_PATH_LEN + 1], *newline;
    
    /* submit all command line arguments first. */
    while(*argv) {
        /* submit and advance to next argument */
        submit_input(*(argv++), dwt_level_count);
    }
}


/** Main - allocates stuff, starts encoding and then frees the stuff */
int main(int argn, char ** argv) {
    pthread_t saving_thread;
    const int dwt_level_count = 4;
    const int gpu_idx_array[] = {0};
    const int gpu_idx_count = sizeof(gpu_idx_array) / sizeof(*gpu_idx_array);
    int i;
    struct work_item * item;
    
    /* remember first argument */
    prog_name = *argv;
    
    /* get and check image size */
    if(argn < 3)
        usage("Expected at least 2 arguments.");
    size_x = atoi(argv[1]);
    size_y = atoi(argv[2]);
    if(size_x <= 0)
        usage("Width must be positive integer.");
    if(size_y <= 0)
        usage("Height must be positive integer.");
    
    /* initialize encoder instance */
    enc = demo_enc_create(gpu_idx_array, gpu_idx_count, size_x, size_y,
                          dwt_level_count, 0.8f);
    if(0 == enc)
        usage("Encoder initialization ERROR.");
    
    /* initialize synchronization stuff */
    pthread_mutex_init(&mutex, 0);
    pthread_cond_init(&cond, 0);
    
    /* allocate work_items - 6 per each GPU */
    buffer_size = size_x * size_y * 4 + 1024 * 1024;
    for(i = gpu_idx_count * 6; i--; ) {
        item = malloc(sizeof(struct work_item));
        item->next = unused;
        unused = item;
        cudaMallocHost(&item->buffer, buffer_size);
    }
    
    /* start saving thread */
    pthread_create(&saving_thread, 0, saving_thread_impl, 0);
    
    /* traverse all inputs and push them to encoder */
    load_input(argv + 3, dwt_level_count);
        
    /* destroy all work items */
    for(i = gpu_idx_count * 6; i--; ) {
        /* wait for next work item */
        pthread_mutex_lock(&mutex);
        while(0 == unused)
            pthread_cond_wait(&cond, &mutex);
        item = unused;
        unused = item->next;
        pthread_mutex_unlock(&mutex);
        
        /* release all resources of the work item */
        cudaFreeHost(item->buffer);
        free(item);
    }
    
    /* signalize encoder to stop (this unblocks the saving thread) */
    demo_enc_stop(enc);
    
    /* join the saving thread */
    pthread_join(saving_thread, 0);
    
    /* destroy the encoder */
    demo_enc_destroy(enc);
    
    /* destroy synchronization stuff */
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    
    /* bye */
    printf("Bye.\n");
    return 0;
}

