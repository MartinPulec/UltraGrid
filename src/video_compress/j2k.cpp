/**
 * @file   video_compress/j2k.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013 CESNET z.s.p.o.
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

#include "cmpto_j2k.h"
#include "debug.h"
#include "host.h"
#include "module.h"
#include "video_compress.h"
#include "video_compress/j2k.h"
#include "video.h"

#include <queue>
#include <utility>

using namespace std;

struct encoded_image {
        char *data;
        int len;
        struct video_desc *desc;
};

struct state_video_compress_j2k {
        struct module module_data;

        struct CMPTO_J2K_Enc_Context *context;
        struct CMPTO_J2K_Enc_Settings *enc_settings;

        pthread_cond_t frame_ready;
        pthread_mutex_t lock;
        queue<struct encoded_image *> *encoded_images;

        pthread_t thread_id;
};

static void j2k_compress_done(struct module *mod);
static void *j2k_compress_worker(void *args);

static void *j2k_compress_worker(void *args)
{
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) args;
        enum CMPTO_J2K_Error j2k_error;

        while (true) {
                struct CMPTO_J2K_Enc_Image *img;
                j2k_error = CMPTO_J2K_Enc_Context_Get_Encoded_Image(
                                s->context,
                                &img /* Set to NULL if encoder stopped */);
                if (j2k_error != CMPTO_J2K_OK) {
                        // some better solution?
                        continue;
                }

                if (img == NULL) {
                        break;
                }
                struct video_desc *desc;
                j2k_error = CMPTO_J2K_Enc_Image_Get_Custom_Data(
                                img, (void **) &desc);
                if (j2k_error != CMPTO_J2K_OK) {
                        continue;
                }
                size_t size;
                void * ptr;
                j2k_error = CMPTO_J2K_Enc_Image_Get_Codestream(
                                img,
                                &size,
                                &ptr);
                if (j2k_error != CMPTO_J2K_OK) {
                        continue;
                }
                struct encoded_image *encoded = (struct encoded_image *)
                        malloc(sizeof(struct encoded_image));
                encoded->data = (char *) malloc(size);
                memcpy(encoded->data, ptr, size);
                encoded->len = size;
                encoded->desc = desc;
                encoded->desc->color_spec = J2K;
                CMPTO_J2K_Enc_Context_Return_Unused_Image(
                                s->context, img);

                pthread_mutex_lock(&s->lock);
                s->encoded_images->push(encoded);
                pthread_cond_signal(&s->frame_ready);
                pthread_mutex_unlock(&s->lock);
        }

        return NULL;
}

struct module * j2k_compress_init(struct module *parent, const struct video_compress_params *params)
{
        struct state_video_compress_j2k *s;
        enum CMPTO_J2K_Error j2k_error;
        
        s = (struct state_video_compress_j2k *) calloc(1, sizeof(struct state_video_compress_j2k));

        j2k_error = CMPTO_J2K_Enc_Context_Init_CUDA( 
                        (const int *) cuda_devices,
                        cuda_devices_count,
                        &s->context);
        if (j2k_error != CMPTO_J2K_OK) {
fprintf(stderr, "%s", CMPTO_J2K_Get_Error_Message(j2k_error));
                goto error;
        }

        j2k_error = CMPTO_J2K_Enc_Settings_Create( 
                        s->context,
                        &s->enc_settings);
        if (j2k_error != CMPTO_J2K_OK) {
                goto error;
        }
CMPTO_J2K_Enc_Settings_Quantization(
    s->enc_settings,
    0.7 /* 0.0 = poor quality, 1.0 = full quality */
);

CMPTO_J2K_Enc_Settings_Rate_Limit(s->enc_settings, 1300000);
CMPTO_J2K_Enc_Settings_Enable(s->enc_settings, CMPTO_J2K_Rate_Control);
CMPTO_J2K_Enc_Settings_Enable(s->enc_settings, CMPTO_J2K_MCT);


        j2k_error = CMPTO_J2K_Enc_Settings_DWT_Count(
                                s->enc_settings,
                                6);
        if (j2k_error != CMPTO_J2K_OK) {
                goto error;
        }
        assert(pthread_cond_init(&s->frame_ready, NULL) == 0);
        assert(pthread_mutex_init(&s->lock, NULL) == 0);

        module_init_default(&s->module_data);
        s->module_data.cls = MODULE_CLASS_DATA;
        s->module_data.priv_data = s;
        s->module_data.deleter = j2k_compress_done;
        module_register(&s->module_data, parent);

        s->encoded_images = new queue<struct encoded_image *>();

        assert(pthread_create(&s->thread_id, NULL, j2k_compress_worker,
                        (void *) s) == 0);

        return &s->module_data;

error:
        if (s) {
                free(s);
        }
        return NULL;
}

static void j2k_compressed_frame_dispose(struct video_frame *frame)
{
	free(frame->tiles[0].data);
	vf_free(frame);
}


struct video_frame  *j2k_compress(struct module *mod, struct video_frame *tx,
                int buffer_idx)
{
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) mod->priv_data;
        struct CMPTO_J2K_Enc_Image *img;
        enum CMPTO_J2K_Error j2k_error;
        void *ptr;
        struct video_desc desc;
        void *udata;

	if (tx == NULL)
		goto get_frame_from_queue;

        assert(tx->tile_count == 1); // TODO

        j2k_error = CMPTO_J2K_Enc_Context_Get_Free_Image(
                        s->context,
                                tx->tiles[0].width,
                                tx->tiles[0].height,
                                CMPTO_J2K_444_u8_p012,
                                &img);
        if (j2k_error != CMPTO_J2K_OK) {
                return NULL;
        }

        j2k_error = CMPTO_J2K_Enc_Image_Get_Source_Data_Ptr(img,
                        &ptr);
        if (j2k_error != CMPTO_J2K_OK) {
                return NULL;
        }
        memcpy(ptr, tx->tiles[0].data,
                        tx->tiles[0].data_len);
        desc = video_desc_from_frame(tx);
        udata = malloc(sizeof(desc));
        memcpy(udata, &desc, sizeof(desc));

        j2k_error = CMPTO_J2K_Enc_Image_Set_Custom_Data(
                        img, udata);
        if (j2k_error != CMPTO_J2K_OK) {
                return NULL;
        }

        j2k_error = CMPTO_J2K_Enc_Context_Encode_Image(s->context,
                        img, s->enc_settings);
        if (j2k_error != CMPTO_J2K_OK) {
                return NULL;
        }

get_frame_from_queue:
        pthread_mutex_lock(&s->lock);
        struct encoded_image *encoded_img = NULL;
        if (s->encoded_images->size() > 0) {
                encoded_img = s->encoded_images->front();
                s->encoded_images->pop();
        }
        pthread_mutex_unlock(&s->lock);

        if (encoded_img != NULL) {
		struct video_frame *out = vf_alloc_desc(*(encoded_img->desc));

                free(encoded_img->desc);
                out->tiles[0].data = encoded_img->data;
                out->tiles[0].data_len =
                        encoded_img->len;
		out->dispose = j2k_compressed_frame_dispose;
                free(encoded_img);
                return out;
        } else {
                return NULL;
        }
}


static void j2k_compress_done(struct module *mod)
{
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) mod->priv_data;

        CMPTO_J2K_Enc_Context_Destroy(s->context);
        pthread_cond_destroy(&s->frame_ready);
        pthread_mutex_destroy(&s->lock);
        delete s->encoded_images;

        free(s);
}

