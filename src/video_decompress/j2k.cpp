/**
 * @file   video_decompress/j2k.cpp
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
#include "debug.h"
#include "host.h"
#include "video.h"
#include "video_decompress.h"

#include "cmpto_j2k.h"
#include "video_decompress/j2k.h"

#include <queue>

using std::queue;

struct state_decompress_j2k {
        CMPTO_J2K_Dec_Context *decoder;
        CMPTO_J2K_Dec_Settings *settings;

        struct video_desc desc;
        codec_t out_codec;

        pthread_mutex_t lock;
        queue<char *> *decompressed_frames;
        pthread_t thread_id;
};

static void *decompress_j2k_worker(void *args);

static void *decompress_j2k_worker(void *args)
{
        struct state_decompress_j2k *s =
                (struct state_decompress_j2k *) args;
        enum CMPTO_J2K_Error j2k_error;

        while (true) {
                struct CMPTO_J2K_Dec_Image *img;
                j2k_error = CMPTO_J2K_Dec_Context_Get_Decoded_Image(
                                s->decoder, &img);
                if (j2k_error != CMPTO_J2K_OK) {
                        continue;
                }

                if (img == NULL) {
                        /// @todo what about reconfiguration
                        break;
                }

                void *dec_data;
                j2k_error = CMPTO_J2K_Dec_Image_Get_Decoded_Data_Ptr(
                                img, &dec_data);
                if (j2k_error != CMPTO_J2K_OK) {
                        continue;
                }
                int data_len = s->desc.height * vc_get_linesize(
                                        s->desc.width, s->out_codec);
                char *buffer = (char *) malloc(data_len);
                memcpy(buffer, dec_data, data_len);

                pthread_mutex_lock(&s->lock);
                s->decompressed_frames->push(buffer);
                pthread_mutex_unlock(&s->lock);
                j2k_error = CMPTO_J2K_Dec_Context_Return_Unused_Image(
                                s->decoder,
                                img);
                if (j2k_error != CMPTO_J2K_OK) {
                        fprintf(stderr, "[J2K decompress] unable to "
                                        "to return processed image!\n");
                }
        }
}

void * j2k_decompress_init(void)
{
        struct state_decompress_j2k *s = NULL;
        enum CMPTO_J2K_Error j2k_error;

        s = (struct state_decompress_j2k *)
                calloc(1, sizeof(struct state_decompress_j2k));
        assert(pthread_mutex_init(&s->lock, NULL) == 0);

        j2k_error = CMPTO_J2K_Dec_Context_Init_CUDA(
                        (const int *) cuda_devices,
                        cuda_devices_count,
                        &s->decoder);
        if (j2k_error != CMPTO_J2K_OK) {
                goto error;
        }

        j2k_error = CMPTO_J2K_Dec_Settings_Create( 
                        s->decoder,
                        &s->settings);
        if (j2k_error != CMPTO_J2K_OK) {
                goto error;
        }

        s->decompressed_frames = new queue<char *>();

        assert(pthread_create(&s->thread_id, NULL, decompress_j2k_worker,
                                (void *) s) == 0);

        return s;

error:
        delete s->decompressed_frames;
        if (s->settings) {
                CMPTO_J2K_Dec_Settings_Destroy(s->settings);
        }
        if (s->decoder) {
                CMPTO_J2K_Dec_Context_Destroy(s->decoder);
        }
        if (s) {
                pthread_mutex_destroy(&s->lock);
                free(s);
        }
        return NULL;
}

int j2k_decompress_reconfigure(void *state, struct video_desc desc, 
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_j2k *s = (struct state_decompress_j2k *) state;
        
        assert(out_codec == RGB);
        assert((rshift == 0 && gshift == 8 && bshift == 16) ||
                        (rshift == 16 && gshift == 8 && bshift == 0));
        assert(pitch == vc_get_linesize(desc.width, out_codec));

        int j2k_error = CMPTO_J2K_Dec_Settings_Data_Format(
                                s->settings,
                                (rshift == 0) ?
                                CMPTO_J2K_444_u8_p012 :
                                CMPTO_J2K_444_u8_p210);

        if (j2k_error != CMPTO_J2K_OK) {
                return FALSE;
        }

        s->desc = desc;
        s->out_codec = out_codec;

        return TRUE;
}

int j2k_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int frame_seq)
{
        struct state_decompress_j2k *s =
                (struct state_decompress_j2k *) state;
        enum CMPTO_J2K_Error j2k_error;
        struct CMPTO_J2K_Dec_Image *img;

        j2k_error = CMPTO_J2K_Dec_Context_Get_Free_Image(
                        s->decoder,
                        src_len,
                        &img);
        if (j2k_error != CMPTO_J2K_OK) {
                return FALSE;
        }

        void *ptr;
        j2k_error = CMPTO_J2K_Dec_Image_Get_Codestream_Ptr(
                        img, &ptr);
        if (j2k_error != CMPTO_J2K_OK) {
                return FALSE;
        }
        memcpy(ptr, buffer, src_len);

        j2k_error = CMPTO_J2K_Dec_Context_Decode_Image(
                        s->decoder,
                        img,
                        s->settings);
        if (j2k_error != CMPTO_J2K_OK) {
                return FALSE;
        }
        
        pthread_mutex_lock(&s->lock);
        if (s->decompressed_frames->size() == 0) {
                pthread_mutex_unlock(&s->lock);
                return FALSE;
        }
        char *decoded = s->decompressed_frames->front();
        s->decompressed_frames->pop();
        pthread_mutex_unlock(&s->lock);

        memcpy(dst, decoded, s->desc.height *
                        vc_get_linesize(s->desc.width, s->out_codec));

        free(decoded);

        return TRUE;
}

int j2k_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        int ret = FALSE;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
                                *(int *) val = FALSE;
                                *len = sizeof(int);
                                ret = TRUE;
                        }
                        break;
                default:
                        ret = FALSE;
        }

        return ret;
}

void j2k_decompress_done(void *state)
{
        struct state_decompress_j2k *s = (struct state_decompress_j2k *) state;

        CMPTO_J2K_Dec_Settings_Destroy(s->settings);
        CMPTO_J2K_Dec_Context_Destroy(s->decoder);

        pthread_mutex_destroy(&s->lock);

        delete s->decompressed_frames;

        free(s);
}

