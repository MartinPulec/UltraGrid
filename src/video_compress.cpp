/**
 * @file   video_compress.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @ingroup video_compress
 *
 * @brief Video compress functions.
 */
/*
 * Copyright (c) 2011-2013 CESNET z.s.p.o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <stdio.h>
#include <string.h>
#include "messaging.h"
#include "module.h"
#include "video.h"
#include "video_compress.h"
#include "video_compress/cuda_dxt.h"
#include "video_compress/dxt_glsl.h"
#include "video_compress/fastdxt.h"
#include "video_compress/libavcodec.h"
#include "video_compress/j2k.h"
#include "video_compress/jpeg.h"
#include "video_compress/none.h"
#include "video_compress/uyvy.h"
#include "lib_common.h"
#include "utils/message_queue.h"
#include "utils/worker.h"

namespace {
/* *_str are symbol names inside library */
/**
 * @brief This struct describes individual compress module
 *
 * Initially, in this struct are either callbacks or functions names.
 * For actual initialization of the callbacks/names, @ref MK_STATIC and @ref MK_NAME
 * macros should be used. After initialization, callbacks are set.
 *
 * There are 2 APIs available - drivers are required to implement one of them. They
 * can implement either @ref compress_frame_func or @ref compress_tile_func function.
 * The other* shall then be NULL.
 */
struct compress_t {
        const char        * name;         ///< compress (unique) name
        const char        * library_name; ///< If module is dynamically loadable, this is the name of library.

        compress_init_t     init_func;           ///< compress driver initialization function
        const char        * init_str;
        compress_frame_t    compress_frame_func; ///< compress function for Frame API
        const char         *compress_frame_str;
        compress_tile_t     compress_tile_func;  ///< compress function for Tile API
        const char         *compress_tile_str;
        /**
         * Async readback function.
         * If not defined, synchronous API is supposed.
         */
        compress_pop_t      compress_pop_func;
        const char         *compress_pop_str;

        void *handle;                     ///< for modular build, dynamically loaded library handle
};

/**
 * @brief This structure represents real internal compress state
 */
struct compress_state_real {
        struct compress_t  *handle;                 ///< handle for the driver
        struct module     **state;                  ///< driver internal states
        unsigned int        state_count;            ///< count of compress states (equal to tiles' count)
        char                compress_options[1024]; ///< compress options (for reconfiguration)

        struct video_frame *out_frame[2];           /**< @brief allocated output frame.
                                                     * This member is only used if the compress driver
                                                     * uses tile API. In this case returned tiles are
                                                     * arranged to frame.
                                                     * @see compress_t */
        message_queue      *queue;

        int                 buffer_index;
};

struct msg_frame : public msg {
        msg_frame(struct video_frame *f) : frame(f) {}
        struct video_frame *frame;
};
}

/**
 * @brief Video compress state.
 *
 * This structure represents external video compress state. This is basically a proxy for real
 * state. The point of doing this is to allow dynamic reconfiguration of the real state.
 */
struct compress_state {
        struct module mod;               ///< compress module data
        struct compress_state_real *ptr; ///< pointer to real compress state
};

typedef struct compress_state compress_state_proxy; ///< Used to emphasize that the state is actually a proxy.

/**
 * This is placeholder state returned by compression module meaning that the initialization was
 * successful but no state was create. This is the case eg. when the module only displayed help.
 */
struct module compress_init_noerr;

static void init_compressions(void);
static struct video_frame *compress_frame_tiles(struct compress_state_real *s, struct video_frame *frame,
                int buffer_index, struct module *parent);
static int compress_init_real(struct module *parent, const char *config_string,
                struct compress_state_real **state);
static void compress_done_real(struct compress_state_real *s);
static void compress_done(struct module *mod);

/**
 * @brief This table contains list of video compress devices compiled with this UltraGrid version.
 * @copydetails decoders
 */
struct compress_t compress_modules[] = {
#if defined HAVE_FASTDXT || defined BUILD_LIBRARIES
        {
                "FastDXT",
                "fastdxt",
                MK_NAME(fastdxt_init),
                MK_NAME(NULL),
                MK_NAME(fastdxt_compress_tile),
                MK_NAME(NULL),
                NULL
        },
#endif
#if defined HAVE_DXT_GLSL || defined BUILD_LIBRARIES
        {
                "RTDXT",
                "rtdxt",
                MK_NAME(dxt_glsl_compress_init),
                MK_NAME(dxt_glsl_compress),
                MK_NAME(NULL),
                MK_NAME(NULL),
                NULL
        },
#endif
        {
                "J2K",
                "j2k",
                MK_NAME(j2k_compress_init),
                MK_NAME(NULL),
                MK_NAME(j2k_compress_tile),
                MK_NAME(NULL),
                NULL
        },
#if defined HAVE_JPEG || defined  BUILD_LIBRARIES
        {
                "JPEG",
                "jpeg",
                MK_NAME(jpeg_compress_init),
                MK_NAME(jpeg_compress),
                MK_NAME(NULL),
                MK_NAME(NULL),
                NULL
        },
#endif
#if defined HAVE_COMPRESS_UYVY || defined  BUILD_LIBRARIES
        {
                "UYVY",
                "uyvy",
                MK_NAME(uyvy_compress_init),
                MK_NAME(uyvy_compress),
                MK_NAME(NULL),
                MK_NAME(NULL),
                NULL
        },
#endif
#if defined HAVE_LAVC || defined  BUILD_LIBRARIES
        {
                "libavcodec",
                "libavcodec",
                MK_NAME(libavcodec_compress_init),
                MK_NAME(NULL),
                MK_NAME(libavcodec_compress_tile),
                MK_NAME(NULL),
                NULL
        },
#endif
#if defined HAVE_CUDA_DXT || defined  BUILD_LIBRARIES
        {
                "cuda_dxt",
                "cuda_dxt",
                MK_NAME(cuda_dxt_compress_init),
                MK_NAME(NULL),
                MK_NAME(cuda_dxt_compress_tile),
                MK_NAME(NULL),
                NULL
        },
#endif
        {
                "none",
                NULL,
                MK_STATIC(none_compress_init),
                MK_STATIC(none_compress),
                MK_STATIC(NULL),
                MK_NAME(NULL),
                NULL
        },
};

#define MAX_COMPRESS_MODULES (sizeof(compress_modules)/sizeof(struct compress_t))

/// @brief List of available display devices.
///
/// initialized automatically
static struct compress_t *available_compress_modules[MAX_COMPRESS_MODULES];
/// @brief Count of @ref available_compress_modules.
///
/// initialized automatically
static int compress_modules_count = 0;

#ifdef BUILD_LIBRARIES
/** Opens compress library of given name. */
static void *compress_open_library(const char *compress_name)
{
        char name[128];
        snprintf(name, sizeof(name), "vcompress_%s.so.%d", compress_name, VIDEO_COMPRESS_ABI_VERSION);

        return open_library(name);
}

/** For a given device, load individual functions from library handle (previously opened). */
static int compress_fill_symbols(struct compress_t *compression)
{
        void *handle = compression->handle;

        compression->init_func = (compress_init_t)
                dlsym(handle, compression->init_str);
        compression->compress_frame_func = (compress_frame_t)
                dlsym(handle, compression->compress_frame_str);
        compression->compress_tile_func = (compress_tile_t)
                dlsym(handle, compression->compress_tile_str);

        if(!compression->init_func ||
                        (compression->compress_frame_func == 0 && compression->compress_tile_func == 0)
                        ) {
                fprintf(stderr, "Library %s opening error: %s \n", compression->library_name, dlerror());
                return FALSE;
        }
        return TRUE;
}
#endif

/// @brief guard of @ref available_compress_modules initialization
static pthread_once_t compression_list_initialized = PTHREAD_ONCE_INIT;

/// @brief initializes @ref available_compress_modules initialization
static void init_compressions(void)
{
        unsigned int i;
        for(i = 0; i < sizeof(compress_modules)/sizeof(struct compress_t); ++i) {
#ifdef BUILD_LIBRARIES
                if(compress_modules[i].library_name) {
                        int ret;
                        compress_modules[i].handle = compress_open_library(compress_modules[i].library_name);
                        if(!compress_modules[i].handle) continue;
                        ret = compress_fill_symbols(&compress_modules[i]);
                        if(!ret) {
                                fprintf(stderr, "Opening symbols from library %s failed.\n", compress_modules[i].library_name);
                                continue;
                        }
                }
#endif
                available_compress_modules[compress_modules_count] = &compress_modules[i];
                compress_modules_count++;
        }
}

/// @brief Displays list of available compressions.
void show_compress_help()
{
        int i;
        pthread_once(&compression_list_initialized, init_compressions);
        printf("Possible compression modules (see '-c <module>:help' for options):\n");
        for(i = 0; i < compress_modules_count; ++i) {
                printf("\t%s\n", available_compress_modules[i]->name);
        }
}

/**
 * @brief Processes message.
 *
 * This function is a callback called from control thread to change some parameters of
 * compression.
 *
 * @param[in] receiver pointer to the compress module
 * @param[in] msg      message to process
 */
static void compress_process_message(compress_state_proxy *proxy, struct msg_change_compress_data *data)
{
        /* In this case we are only changing some parameter of compression.
         * This means that we pass the parameter to compress driver. */
        if(data->what == CHANGE_PARAMS) {
                for(unsigned int i = 0; i < proxy->ptr->state_count; ++i) {
                        struct msg_change_compress_data *tmp_data =
                                (struct msg_change_compress_data *)
                                new_message(sizeof(struct msg_change_compress_data));
                        tmp_data->what = data->what;
                        strncpy(tmp_data->config_string, data->config_string,
                                        sizeof(tmp_data->config_string) - 1);
                        struct response *resp = send_message_to_receiver(proxy->ptr->state[i],
                                        (struct message *) tmp_data);
                        resp->deleter(resp);
                }

        } else {
                struct compress_state_real *new_state;
                char config[1024];
                strncpy(config, data->config_string, sizeof(config));

                int ret = compress_init_real(&proxy->mod, config, &new_state);
                if(ret == 0) {
                        struct compress_state_real *old = proxy->ptr;
                        proxy->ptr = new_state;
                        compress_done_real(old);
                }
        }

        free_message((struct message *) data);
}

/**
 * @brief This function initializes video compression.
 *
 * This function wrapps the call of compress_init_real().
 * @param[in] parent        parent module
 * @param[in] config_string configuration (in format <driver>:<options>)
 * @param[out] state        created state
 * @retval     0            if state created sucessfully
 * @retval    <0            if error occured
 * @retval    >0            finished successfully, no state created (eg. displayed help)
 */
int compress_init(struct module *parent, const char *config_string, struct compress_state **state) {
        struct compress_state_real *s;

        compress_state_proxy *proxy;
        proxy = (compress_state_proxy *) malloc(sizeof(compress_state_proxy));

        module_init_default(&proxy->mod);
        proxy->mod.cls = MODULE_CLASS_COMPRESS;
        proxy->mod.priv_data = proxy;
        proxy->mod.deleter = compress_done;

        int ret = compress_init_real(&proxy->mod, config_string, &s);
        if(ret == 0) {
                proxy->ptr = s;

                *state = proxy;
                module_register(&proxy->mod, parent);
        } else {
                free(proxy);
        }

        return ret;
}

/**
 * @brief This is compression initialization function that really does the stuff.
 * @param[in] parent        parent module
 * @param[in] config_string configuration (in format <driver>:<options>)
 * @param[out] state        created state
 * @retval     0            if state created sucessfully
 * @retval    <0            if error occured
 * @retval    >0            finished successfully, no state created (eg. displayed help)
 */
static int compress_init_real(struct module *parent, const char *config_string,
                struct compress_state_real **state)
{
        struct compress_state_real *s;
        struct video_compress_params params;
        const char *compress_options = NULL;

        if(!config_string)
                return -1;

        if(strcmp(config_string, "help") == 0)
        {
                show_compress_help();
                return 1;
        }

        pthread_once(&compression_list_initialized, init_compressions);

        memset(&params, 0, sizeof(params));

        s = (struct compress_state_real *) calloc(1, sizeof(struct compress_state_real));
        s->queue = new message_queue(1);

        s->state_count = 1;
        int i;
        for(i = 0; i < compress_modules_count; ++i) {
                if(strncasecmp(config_string, available_compress_modules[i]->name,
                                strlen(available_compress_modules[i]->name)) == 0) {
                        s->handle = available_compress_modules[i];
                        if(config_string[strlen(available_compress_modules[i]->name)] == ':')
                                compress_options = config_string +
                                        strlen(available_compress_modules[i]->name) + 1;
                        else
                                compress_options = "";
                }
        }
        if(!s->handle) {
                fprintf(stderr, "Unknown compression: %s\n", config_string);
                free(s);
                return -1;
        }
        strncpy(s->compress_options, compress_options, sizeof(s->compress_options) - 1);
        s->compress_options[sizeof(s->compress_options) - 1] = '\0';
        params.cfg = s->compress_options;
        if(s->handle->init_func) {
                s->state = (struct module **) calloc(1, sizeof(struct module *));
                s->state[0] = s->handle->init_func(parent, &params);
                if(!s->state[0]) {
                        fprintf(stderr, "Compression initialization failed: %s\n", config_string);
                        free(s->state);
                        free(s);
                        return -1;
                }
                if(s->state[0] == &compress_init_noerr) {
                        free(s->state);
                        free(s);
                        return 1;
                }
                for(int i = 0; i < 2; ++i) {
                        s->out_frame[i] = vf_alloc(1);
                }
        } else {
                return -1;
        }

        *state = s;
        return 0;
}

/**
 * @brief Returns name of compression module
 *
 * @param proxy compress state
 * @returns     compress name
 */
const char *get_compress_name(compress_state_proxy *proxy)
{
        if(proxy)
                return proxy->ptr->handle->name;
        else
                return NULL;
}

/**
 * @brief Compressses frame
 *
 * @param proxy        compress state
 * @param frame        uncompressed frame to be compressed
 * @param buffer_index 0 or 1 - driver should have 2 output buffers, filling the selected one.
 *                     Returned video frame should stay valid until requesting compress with the
 *                     same index.
 * @return             compressed frame, may be NULL if compression failed
 */
void compress_frame(compress_state_proxy *proxy, struct video_frame *frame)
{
        if (!proxy)
                abort();

        struct compress_state_real *s = proxy->ptr;

        if (frame == NULL) { // pass poisoned pill
                s->queue->push(new msg_frame(NULL));
                return;
        }

        struct msg_change_compress_data *msg = NULL;
        while ((msg = (struct msg_change_compress_data *) check_message(&proxy->mod))) {
                compress_process_message(proxy, msg);
        }

        struct video_frame *sync_api_frame;
        if(s->handle->compress_frame_func) {
                sync_api_frame = s->handle->compress_frame_func(s->state[0], frame, s->buffer_index);
        } else if(s->handle->compress_tile_func) {
                sync_api_frame = compress_frame_tiles(s, frame, s->buffer_index, &proxy->mod);
        } else {
                sync_api_frame = NULL;
        }

        s->buffer_index = (s->buffer_index + 1) % 2;
        // returned frame is different, so we may dispose the previous one
        if (sync_api_frame != frame) {
                frame->dispose(frame);
        }

        // NULL has special meaning - it is poisoned pill so we must
        // prevent passing it further
        if (sync_api_frame == NULL) {
                return;
        }

        msg_frame *frame_msg;
        if (s->handle->compress_pop_func) {
                abort(); // not yet implemented
        } else {
                frame_msg = new msg_frame(sync_api_frame);
        }
        s->queue->push(frame_msg);
}

/**
 * @name Tile API Routines
 * The worker callbacks here are optimization - all tiles are processed concurrently.
 * @{
 */
/**
 * @brief Auxiliary structure passed to worker thread.
 */
struct compress_worker_data {
        struct module *state;      ///< compress driver status
        struct video_frame *frame; ///< uncompressed tile to be compressed
        int tile_idx;              ///< tile to be compressed
        int buffer_index;          ///< buffer index @see compress_frame

        compress_tile_t callback;  ///< tile compress callback
        void *ret;                 ///< OUT - returned compressed tile, NULL if failed
};

/**
 * @brief This function is callback passed to a "thread pool"
 * @param arg @ref compress_worker_data
 * @return @ref compress_worker_data (same as input)
 */
static void *compress_tile_callback(void *arg) {
        struct compress_worker_data *s = (struct compress_worker_data *) arg;

        s->ret = s->callback(s->state, s->frame, s->tile_idx, s->buffer_index);

        return s;
}

/**
 * Compresses video frame with tiles API
 *
 * @param[in]     s             compress state
 * @param[in]     frame         uncompressed frame
 * @param         buffer_index  0 or 1 - driver should have 2 output buffers, filling the selected one.
 *                Returned video frame should stay valid until requesting compress with the
 *                same index.
 * @param         parent        parent module (for the case when there is a need to reconfigure)
 * @return                      compressed video frame, may be NULL if compression failed
 */
static struct video_frame *compress_frame_tiles(struct compress_state_real *s, struct video_frame *frame,
                int buffer_index, struct module *parent)
{
        struct video_compress_params params;
        memset(&params, 0, sizeof(params));
        params.cfg = s->compress_options;
        if(frame->tile_count != s->state_count) {
                s->state = (struct module **) realloc(s->state, frame->tile_count * sizeof(struct module *));
                for(unsigned int i = s->state_count; i < frame->tile_count; ++i) {
                        s->state[i] = s->handle->init_func(parent, &params);
                        if(!s->state[i]) {
                                fprintf(stderr, "Compression initialization failed\n");
                                return NULL;
                        }
                }
                for(int i = 0; i < 2; ++i) {
                        vf_free(s->out_frame[i]);
                        s->out_frame[i] = vf_alloc(frame->tile_count);
                }
                s->state_count = frame->tile_count;
        }

        task_result_handle_t task_handle[frame->tile_count];

        struct compress_worker_data data_tile[frame->tile_count];
        for(unsigned int i = 0; i < frame->tile_count; ++i) {
                struct compress_worker_data *data = &data_tile[i];
                data->state = s->state[i];
                data->frame = frame;
                data->tile_idx = i;
                data->buffer_index = buffer_index;;
                data->callback = s->handle->compress_tile_func;

                task_handle[i] = task_run_async(compress_tile_callback, data);
        }

        for(unsigned int i = 0; i < frame->tile_count; ++i) {
                struct compress_worker_data *data = (struct compress_worker_data *)
                        wait_task(task_handle[i]);

                if(!data->ret) {
                        return NULL;
                }

                if(i == 0) { // update metadata from first tile
                        struct video_desc desc =
                                video_desc_from_frame((struct video_frame *) data->ret);
                        desc.tile_count = frame->tile_count;
                        vf_write_desc(s->out_frame[buffer_index], desc);
                }

                memcpy(&s->out_frame[buffer_index]->tiles[i],
                                &((struct video_frame *) data->ret)->tiles[0],
                                sizeof(struct tile));
        }

        return s->out_frame[buffer_index];
}
/**
 * @}
 */

/**
 * @brief Video compression cleanup function.
 * @param mod video compress module
 */
static void compress_done(struct module *mod)
{
        if(!mod)
                return;

        compress_state_proxy *proxy = (compress_state_proxy *) mod->priv_data;
        struct compress_state_real *s = proxy->ptr;
        compress_done_real(s);

        free(proxy);
}

/**
 * Video compression cleanup function.
 * This destroys contained real video compress state.
 * @param mod video compress module
 */
static void compress_done_real(struct compress_state_real *s)
{
        if(!s)
                return;

        for(unsigned int i = 0; i < s->state_count; ++i) {
                module_done(s->state[i]);
        }
        free(s->state);
        delete s->queue;
        free(s);
}

struct video_frame *compress_pop(compress_state_proxy *proxy)
{
        struct video_frame *ret = NULL;
        if(!proxy)
                return NULL;

        struct compress_state_real *s = proxy->ptr;

        if(s->handle->compress_pop_func) {
                // ret = s->handle->compress_pop_func(s->state[0], frame, buffer_index);
        } else {
                msg *message = s->queue->pop();
                msg_frame *frame_msg = dynamic_cast<msg_frame *>(message);
                assert(frame_msg != NULL);
                ret = frame_msg->frame;
                delete frame_msg;
        }

        return ret;
}

