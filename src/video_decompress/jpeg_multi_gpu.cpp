/**
 * @file   video_decompress/jpeg_multi_gpu.cpp
 * @author Martin Pulec  <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2016 CESNET z.s.p.o.
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

#include <pthread.h>
#include <stdlib.h>

#include "libgpujpeg/gpujpeg_decoder.h"

#include "debug.h"
#include "lib_common.h"
#include "host.h"
#include "utils/synchronized_queue.h"
#include "video.h"
#include "video_decompress.h"

namespace {

struct thread_data {
        thread_data() :
                jpeg_decoder(0), desc(), out_codec(), out_buff(0),
                cuda_dev_index(-1)
        {}
        synchronized_queue<msg *, 1> m_in;
        // currently only for output frames
        synchronized_queue<msg *, 1> m_out;

        struct gpujpeg_decoder  *jpeg_decoder;
        struct video_desc        desc;
        codec_t                  out_codec;
        char                    *out_buff;

        int                      cuda_dev_index;
};

struct msg_reconfigure : public msg {
        struct video_desc        desc;
        codec_t                  out_codec;
};

struct msg_reconfigure_status : public msg {
        msg_reconfigure_status(bool val) : success(val) {}

        bool                     success;
};

struct msg_frame : public msg {
        msg_frame(int len) {
                data = new char[len];
                data_len = len;
        }
        ~msg_frame() {
                delete data;
        }
        char *data;
        int data_len;
};

struct state_decompress_jpeg_to_dxt {
        struct thread_data       thread_data[MAX_CUDA_DEVICES];
        pthread_t                thread_id[MAX_CUDA_DEVICES];

        struct video_desc        desc;

        // which card is free to process next image
        unsigned int             free;
        unsigned int             occupied_count; // number of GPUs occupied
};

static int reconfigure_thread(struct thread_data *s, struct video_desc desc);

static void *worker_thread(void *arg)
{
        struct thread_data *s =
                (struct thread_data *) arg;

        while(1) {
                msg *message = s->m_in.pop();

                if (dynamic_cast<msg_quit *>(message)) {
                        delete message;
                        break;
                } else if (dynamic_cast<msg_reconfigure *>(message)) {
                        msg_reconfigure *reconf = dynamic_cast<msg_reconfigure *>(message);
                        bool ret = reconfigure_thread(s, reconf->desc);
                        s->desc = reconf->desc;
                        s->out_codec = reconf->out_codec;

                        msg_reconfigure_status *status = new msg_reconfigure_status(ret);
                        s->m_out.push(status);

                        delete message;
                } else if (dynamic_cast<msg_frame *>(message)) {
                        msg_frame *frame_msg = dynamic_cast<msg_frame *>(message);
                        struct gpujpeg_decoder_output decoder_output;

                        msg_frame *output_frame = new msg_frame(vc_get_linesize(s->desc.width, s->desc.color_spec) * s->desc.height);

                        gpujpeg_decoder_output_set_custom(&decoder_output, (uint8_t *) output_frame->data);


                        if (gpujpeg_decoder_decode(s->jpeg_decoder, (uint8_t *) frame_msg->data, frame_msg->data_len,
                                        &decoder_output) != 0) {
                                log_msg(LOG_LEVEL_ERROR, "JPEG decoding error!\n");
                                delete output_frame;
                                continue;
                        }

                        s->m_out.push(output_frame);

                        delete frame_msg;
                }
        }

        if(s->jpeg_decoder) {
                gpujpeg_decoder_destroy(s->jpeg_decoder);
        }

        free(s->out_buff);

        return NULL;
}

void * jpeg_to_dxt_decompress_init(void)
{
        struct state_decompress_jpeg_to_dxt *s;

        s = new state_decompress_jpeg_to_dxt;

        s->free = 0;
        s->occupied_count = 0;

        for(unsigned int i = 0; i < cuda_devices_count; ++i) {
                int ret = gpujpeg_init_device(cuda_devices[i], TRUE);
                if(ret != 0) {
                        fprintf(stderr, "[JPEG] initializing CUDA device %d failed.\n", cuda_devices[0]);
                        delete s;
                        return NULL;
                }
        }

        for(unsigned int i = 0; i < cuda_devices_count; ++i) {
                s->thread_data[i].cuda_dev_index = i;
                int ret = pthread_create(&s->thread_id[i], NULL, worker_thread,  &s->thread_data[i]);
                assert(ret == 0);
        }

        return s;
}

static void flush(struct state_decompress_jpeg_to_dxt *s)
{
        if (s->occupied_count > 0) {
                for (unsigned int i = 0; i < cuda_devices_count; ++i) {
                        if (i == s->free)
                                continue;
                        struct msg_frame *completed =
                                dynamic_cast<msg_frame *>(s->thread_data[(s->free+1)%cuda_devices_count].m_out.pop());
                        delete completed;
                }
                s->occupied_count = 0;
        }
        s->free = 0;
}

/**
 * Reconfigureation function
 *
 * @return 0 to indicate error
 *         otherwise maximal buffer size which ins needed for image of given codec, width, and height
 */
int jpeg_to_dxt_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_jpeg_to_dxt *s = (struct state_decompress_jpeg_to_dxt *) state;

        assert(rshift == 0 && gshift == 8 && bshift == 16);
        assert(out_codec == UYVY || out_codec == RGB);
        /// TODO TODO TODO TODO !
        assert(pitch == (int) vc_get_linesize(desc.width, out_codec)); // default

        s->desc = desc;

        flush(s);

        for(unsigned int i = 0; i < cuda_devices_count; ++i) {
                msg_reconfigure *reconf = new msg_reconfigure;
                reconf->desc = desc;
                reconf->out_codec = out_codec;

                s->thread_data[i].m_in.push(reconf);

                msg_reconfigure_status *status =
                        dynamic_cast<msg_reconfigure_status *>(s->thread_data[i].m_out.pop());
                assert(status != NULL);
                if (!status->success) {
                        delete status;
                        return FALSE;
                }
                delete status;
        }

        return TRUE;
}

/**
 * @return maximal buffer size needed to image with such a properties
 */
static int reconfigure_thread(struct thread_data *s, struct video_desc desc)
{
        if(s->jpeg_decoder != NULL) {
                gpujpeg_decoder_destroy(s->jpeg_decoder);
                s->jpeg_decoder = NULL;
        } else {
                gpujpeg_init_device(cuda_devices[s->cuda_dev_index], 0);
        }

        if(s->out_buff != NULL) {
                free(s->out_buff);
                s->out_buff = NULL;
        }

        s->out_buff = (char *) malloc(vc_get_linesize(desc.width, desc.color_spec) * desc.height);
        if (s->out_buff == NULL) {
                log_msg(LOG_LEVEL_ERROR, "Could not allocate output buffer.\n");
                return false;
        }
        //gpujpeg_init_device(cuda_device, GPUJPEG_OPENGL_INTEROPERABILITY);

        s->jpeg_decoder = gpujpeg_decoder_create();
        if (!s->jpeg_decoder) {
                log_msg(LOG_LEVEL_ERROR, "Creating JPEG decoder failed.\n");
                return false;
        }

        return true;
}

int jpeg_to_dxt_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int frame_seq)
{
        struct state_decompress_jpeg_to_dxt *s = (struct state_decompress_jpeg_to_dxt *) state;
        UNUSED(frame_seq);

        msg_frame *message = new msg_frame(src_len);
        memcpy(message->data, buffer, src_len);

        if(s->occupied_count < cuda_devices_count - 1) {
                s->thread_data[s->free].m_in.push(message);
                s->free = s->free + 1; // should not exceed cuda_device_count
                s->occupied_count += 1;
        } else {
                s->thread_data[s->free].m_in.push(message);

                s->free = (s->free + 1) % cuda_devices_count;

                struct msg_frame *completed =
                        dynamic_cast<msg_frame *>(s->thread_data[s->free].m_out.pop());
                assert(completed != NULL);
                memcpy(dst, completed->data, vc_get_linesize(s->desc.width, s->desc.color_spec) * s->desc.height);

                delete completed;
        }

        return TRUE;
}

int jpeg_to_dxt_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        UNUSED(property);
        UNUSED(val);
        UNUSED(len);

        return FALSE;
}

void jpeg_to_dxt_decompress_done(void *state)
{
        struct state_decompress_jpeg_to_dxt *s = (struct state_decompress_jpeg_to_dxt *) state;

        if(!state) {
                return;
        }

        flush(s);

        for(unsigned int i = 0; i < cuda_devices_count; ++i) {
                msg_quit *quit_msg = new msg_quit;
                s->thread_data[i].m_in.push(quit_msg);
        }

        for(unsigned int i = 0; i < cuda_devices_count; ++i) {
                pthread_join(s->thread_id[i], NULL);
        }

        delete s;
}

static const struct decode_from_to jpeg_to_dxt_decoders[] = {
        { JPEG, UYVY, 200 },
        { JPEG, RGB, 200 },
        { VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, 0 },
};

static const struct video_decompress_info jpeg_to_dxt_info = {
        jpeg_to_dxt_decompress_init,
        jpeg_to_dxt_decompress_reconfigure,
        jpeg_to_dxt_decompress,
        jpeg_to_dxt_decompress_get_property,
        jpeg_to_dxt_decompress_done,
        jpeg_to_dxt_decoders,
};

REGISTER_MODULE(jpeg_multi_gpu, &jpeg_to_dxt_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

} // namespace

