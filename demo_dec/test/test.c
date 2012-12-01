/**
 * @file    test.c
 * @author  Martin Jirman <jirman@cesnet.cz>
 * @brief   2012 JPEG decoder demo simple test app.
 */


#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "../demo_dec.h"


#define MAX_PATH_LEN (1024 * 16)


/** Chained work item type. */
struct work_item {
    /* next work item pointer */
    struct work_item * next;
    
    /* page-locked buffer poiner or NULL if not allocated */
    void * buffer_ptr;
    
    /* buffer size */
    int buffer_size;
    
    /* output image dimensions */
    int size_x, size_y;
    
    /* current output size */
    int output_size;
    
    /* output filename */
    char path[MAX_PATH_LEN + 1];
};


/* global decoder pointer */
struct demo_dec * dec = 0;

/* first argument pointer */
const char * prog_name = 0;

/* unused work item stack with locks */
struct work_item * volatile unused = 0;
pthread_mutex_t mutex;
pthread_cond_t cond;


/** Shows correct usage and exits */
static void usage(const char * const message) {
    printf("ERROR: %s.\n", message);
    printf("Usage: %s file1.j2k [file2.j2k ...]\n", prog_name);
    exit(-1);
}


/** Saving thread implementation. */
static void * saving_thread_impl(void * param) {
    int status;
    struct work_item * item;
    FILE * file = 0;
    
    /* continue as long as there are more decoded work items */
    while(1) {
        status = demo_dec_wait(dec, (void**)&item, 0, 0);
        if(1 == status) /* decoder stopped */
            break; 
        else if(2 == status)
            printf("Error decoding %s.\n", item->path);
        else if(0 == status) {
            /* save the image */
            if(file = fopen(item->path, "w")) {
                if(1 == fwrite(item->buffer_ptr, item->output_size, 1, file))
                    printf("Output file saved OK: %s (%dx%d).\n", 
                           item->path, item->size_x, item->size_y);
                else 
                    printf("Can't write to file %s.\n", item->path);
                fclose(file);
            } else 
                printf("Can't open output file %s.\n", item->path);
        } else
            printf("Unknown decoder status %d!?\n", status);
        
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


/** Gets nonzero if failed to submit the item to decoder. */
static int load_and_submit(FILE * const file,
                           struct work_item * const item,
                           const char * const filename) {
    int file_size;
    void * new_buffer_ptr;
    const int double_sized = 0;
    
    /* get file size */
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if(file_size <= 0) {
        printf("Could not get file size: %s.\n", filename);
        return -1;
    }
    
    /* possibly resize the buffer of work item to file size */
    if(item->buffer_size < file_size) {
        if(item->buffer_ptr && cudaSuccess != cudaFreeHost(item->buffer_ptr)) {
            printf("cudaFreeHost error (%s).\n", filename);
            return -1;
        }
        item->buffer_size = 0;
        item->buffer_ptr = 0;
        if(cudaSuccess != cudaMallocHost(&item->buffer_ptr, file_size)) {
            printf("cudaMallocHost error (%s).\n", filename);
            return -1;
        }
        item->buffer_size = file_size;
    }
    
    /* load data from the file */
    if(1 != fread(item->buffer_ptr, file_size, 1, file)) {
        printf("Cannot read %d bytes from file %s.\n", file_size, filename);
        return -1;
    }
    
    /* get image info */
    if(0 == demo_dec_image_info(item->buffer_ptr, file_size, 0,
                                &item->size_x, &item->size_y)) {
        printf("File %s isn't valid JPEG 2000 codestream.\n", filename);
        return -1;
    }
    if(double_sized) {
        item->size_x *= 2;
        item->size_y *= 2;
    }
    item->output_size = demo_dec_v210_size(item->size_x, item->size_y);
    
    /* check buffer size again (this time for output size) */
    if(item->buffer_size < item->output_size) {
        /* allocate new buffer */
        if(cudaSuccess != cudaMallocHost(&new_buffer_ptr, item->output_size)) {
            printf("cudaMallocHost error (%s).\n", filename);
            return -1;
        }
        
        /* copy codestream */
        memcpy(new_buffer_ptr, item->buffer_ptr, item->buffer_size);
        
        /* delete old buffer */
        if(cudaSuccess != cudaFreeHost(item->buffer_ptr)) {
            printf("cudaFreeHost error (%s).\n", filename);
        }
        item->buffer_ptr = new_buffer_ptr;
        item->buffer_size = item->output_size;
    }
    
    /* show status message */
    printf("Loaded %d bytes from file %s.\n", file_size, filename);
    
    /* submit the work item for decoding and indicate success. */
    demo_dec_submit(dec, item, item->buffer_ptr, item->buffer_ptr, file_size,
                    double_sized);
    return 0;
}


/** Reads given file and submits it for decoding. */
static void submit_input(const char * const filename) {
    struct work_item * item = 0;
    FILE * file = 0;
    long int file_size = -1;
        
    /* open the file for reading */
    if(file = fopen(filename, "r")) {
        /* get unused work item. */
        pthread_mutex_lock(&mutex);
        while(0 == unused)
            pthread_cond_wait(&cond, &mutex);
        item = unused;
        unused = item->next;
        pthread_mutex_unlock(&mutex);
        
        /* compose output filename */
        snprintf(item->path, MAX_PATH_LEN, "%s.v210", filename);
        
        /* load and submit or return work item to queue if failed */
        if(load_and_submit(file, item, filename)) {
            /* return the item back to queue */
            pthread_mutex_lock(&mutex);
            item->next = unused;
            unused = item;
            pthread_cond_signal(&cond);
            pthread_mutex_unlock(&mutex);
        }
        
        /* close the file */
        fclose(file);
    } else 
        printf("Cannot open file %s for reading.\n", filename);
}


/** Main - allocates stuff, starts decoding and then frees the stuff */
int main(int argn, char ** argv) {
    pthread_t saving_thread;
    int i;
    struct work_item * item;
    
    /* remember first argument */
    prog_name = *argv;
    
    /* check argument count */
    if(argn <= 1)
        usage("expected at least 1 input filename");
    
    /* initialize decoder instance */
    dec = demo_dec_create(0, 0);
    if(0 == dec)
        usage("decoder initialization ERROR");
    
    /* initialize synchronization stuff */
    pthread_mutex_init(&mutex, 0);
    pthread_cond_init(&cond, 0);
    
    /* allocate work_items - 6 per each GPU */
    for(i = demo_dec_gpu_count(dec) * 6; i--; ) {
        item = malloc(sizeof(struct work_item));
        item->next = unused;
        unused = item;
        item->buffer_size = -1;
        item->buffer_ptr = 0;
    }
    
    /* start saving thread */
    pthread_create(&saving_thread, 0, saving_thread_impl, 0);
    
    /* traverse all inputs and push them to decoder */
    for(i = 1; i < argn; i++) {
        submit_input(argv[i]);
    }
        
    /* destroy all work items */
    for(i = demo_dec_gpu_count(dec) * 6; i--; ) {
        /* wait for next work item */
        pthread_mutex_lock(&mutex);
        while(0 == unused)
            pthread_cond_wait(&cond, &mutex);
        item = unused;
        unused = item->next;
        pthread_mutex_unlock(&mutex);
        
        /* release all resources of the work item */
        if(item->buffer_ptr)
            cudaFreeHost(item->buffer_ptr);
        free(item);
    }
    
    /* signalize decoder to stop (this unblocks the saving thread) */
    demo_dec_stop(dec);
    
    /* join the saving thread */
    pthread_join(saving_thread, 0);
    
    /* destroy the decoder */
    demo_dec_destroy(dec);
    
    /* destroy synchronization stuff */
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    
    /* bye */
    printf("Bye.\n");
    return 0;
}

