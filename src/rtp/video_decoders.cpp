/*
 * Copyright (c) 2003-2004 University of Southern California
 * Copyright (c) 2005-2013 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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
/**
 * @file    rtp/video_decoders.cpp
 * @author  Colin Perkins
 * @author  Ladan Gharai
 * @author  Martin Pulec <pulec@cesnet.cz>
 *
 * @ingroup video_rtp_decoder
 *
 * Normal workflow through threads is following:
 * 1. decode data from in context of receiving thread with function decode_frame(),
 *    it will decode it to framebuffer, passes it to FEC thread
 * 2. fec_thread() passes frame to decompress thread
 * 3. thread running decompress_thread(), displays the frame
 *
 * ### Uncompressed video (without FEC) ###
 * In step one, the decoder is a linedecoder and framebuffer is the display framebuffer
 *
 * ### Compressed video ###
 * Data is saved to decompress buffer. The decompression itself is done by decompress_thread().
 *
 * ### video with FEC ###
 * Data is saved to FEC buffer. Decoded with fec_thread().
 *
 * ### Encrypted video (without FEC) ###
 * Prior to decoding, packet is decompressed.
 *
 * ### Encrypted video (with FEC) ###
 * After FEC decoding, the whole block is decompressed.
 *
 * @todo
 * This code is very very messy, it needs to be rewritten.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "crypto/openssl_decrypt.h"
#include "debug.h"
#include "host.h"
#include "messaging.h"
#include "perf.h"
#include "rtp/fec.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/video_decoders.h"
#include "utils/message_queue.h"
#include "utils/timed_message.h"
#include "video.h"
#include "video_decompress.h"
#include "video_display.h"
#include "vo_postprocess.h"

#include <future>
#include <iostream>
#include <map>
#include <queue>

using std::map;
using std::queue;

struct state_video_decoder;

/**
 * Interlacing changing function protoype. The function should be able to change buffer
 * in place, that is when dst and src are the same.
 */
typedef void (*change_il_t)(char *dst, char *src, int linesize, int height);

// prototypes
static bool reconfigure_decoder(struct state_video_decoder *decoder,
                struct video_desc desc);
static void restrict_returned_codecs(codec_t *display_codecs,
                size_t *display_codecs_count, codec_t *pp_codecs,
                int pp_codecs_count);
static int check_for_mode_change(struct state_video_decoder *decoder, uint32_t *hdr);
static void wait_for_framebuffer_swap(struct state_video_decoder *decoder);
static void *fec_thread(void *args);
static void *decompress_thread(void *args);
static void cleanup(struct state_video_decoder *decoder);
static bool parse_video_hdr(uint32_t *hdr, struct video_desc *desc);

static int sum_map(map<int, int> const & m) {
        int ret = 0;
        for (map<int, int>::const_iterator it = m.begin();
                        it != m.end(); ++it) {
                ret += it->second;
        }
        return ret;
}

namespace {
/**
 * Enumerates 2 possibilities how to decode arriving data.
 */
enum decoder_type_t {
        UNSET,
        LINE_DECODER,    ///< This is simple decoder, that decodes incoming data per line (pixelformats only).
        EXTERNAL_DECODER /**< real decompress, received buffer is opaque and has to be decompressed as
                          * a whole. */
};

/**
 * This structure holds data needed to use a linedecoder.
 */
struct line_decoder {
        int                  base_offset;  ///< from the beginning of buffer. Nonzero if decoding from mutiple tiles.
        double               src_bpp;      ///< Source pixelformat BPP (bytes)
        double               dst_bpp;      ///< Destination pixelformat BPP (bytes)
        int                  shifts[3];    ///< requested red,green and blue shift (in bits)
        decoder_t            decode_line;  ///< actual decoding function
        unsigned int         dst_linesize; ///< destination linesize
        unsigned int         dst_pitch;    ///< framebuffer pitch - it can be larger if SDL resolution is larger than data */
        unsigned int         src_linesize; ///< source linesize
};

struct decompress_msg;
struct main_msg;

// message definitions
typedef char *char_p;
struct fec_msg : public msg {
        fec_msg(int count) : substream_count(count) {
                buffer_len = new int[count];
                buffer_num = new int[count];
                recv_buffers = new char_p[count];
                pckt_list = 0;
        }
        ~fec_msg() {
                delete [] buffer_len;
                delete [] buffer_num;
                delete [] recv_buffers;
                delete [] pckt_list;
        }
        int *buffer_len;
        int *buffer_num;
        char **recv_buffers;
        map<int, int> *pckt_list;
        fec_desc fec_description;
        int substream_count;
        bool poisoned;
};

struct decompress_msg : public msg {
        decompress_msg(int count) :
                        poisoned(false) {
                decompress_buffer = new char_p[count];
                fec_buffers = new char_p[count];
                buffer_len = new int[count];
                buffer_num = new int[count];
        }
        ~decompress_msg() {
                delete [] decompress_buffer;
                delete [] fec_buffers;
                delete [] buffer_len;
                delete [] buffer_num;
        }
        char **decompress_buffer;
        int *buffer_len;
        int *buffer_num;
        char **fec_buffers;
        bool poisoned;
};

struct main_msg : public msg {
        virtual ~main_msg(){}
};

struct main_msg_reconfigure : public main_msg {
        main_msg_reconfigure(struct video_desc d) : desc(d) {}
        struct video_desc   desc;
};
}

/**
 * @brief Decoder state
 */
struct state_video_decoder
{
        state_video_decoder() :
                decompress_queue(1), fec_queue(1)
        {}
        virtual ~state_video_decoder() {}

        struct module *parent;

        pthread_t decompress_thread_id,
                  fec_thread_id;
        struct video_desc received_vid_desc; ///< description of the network video
        struct video_desc display_desc;      ///< description of the mode that display is currently configured to

        struct video_frame *frame; ///< @todo rewrite this more reasonably

        struct display   *display; ///< assigned display device
        /// @{
        int               display_requested_pitch;
        int               display_requested_rgb_shift[3];
        codec_t          *native_codecs; ///< list of native codecs
        size_t            native_count;  ///< count of @ref native_codecs
        enum interlacing_t *disp_supported_il; ///< display supported interlacing mode
        size_t            disp_supported_il_cnt; ///< count of @ref disp_supported_il
        /// @}

        unsigned int      max_substreams; ///< maximal number of expected substreams
        change_il_t       change_il;      ///< function to change interlacing, if needed. Otherwise NULL.

        pthread_mutex_t lock;

        enum decoder_type_t decoder_type;  ///< how will the video data be decoded
        struct line_decoder *line_decoder; ///< if the video is uncompressed and only pixelformat change
                                           ///< is neeeded, use this structure
        struct state_decompress **decompress_state; ///< state of the decompress (for every substream)
        unsigned int accepts_corrupted_frame:1;     ///< whether we should pass corrupted frame to decompress
        bool buffer_swapped; /**< variable indicating that display buffer
                              * has been processed and we can write to a new one */
        pthread_cond_t buffer_swapped_cv; ///< condition variable associated with @ref buffer_swapped

        message_queue     decompress_queue;

        codec_t           out_codec;
        // display or postprocessor
        int               pitch;
        // display pitch, can differ from previous if we have postprocessor
        int               display_pitch;

        /// @{
        struct vo_postprocess_state *postprocess;
        struct video_frame *pp_frame;
        int pp_output_frames_count;
        /// @}

        message_queue     fec_queue;

        enum video_mode   video_mode;  ///< video mode set for this decoder
        unsigned          merged_fb:1; ///< flag if the display device driver requires tiled video or not

        // for statistics
        /// @{
        volatile unsigned long int displayed, dropped, corrupted, missing;
        volatile unsigned long int fec_ok, fec_nok;
        long int last_buffer_number;
        /// @}
        timed_message       slow_msg; ///< shows warning ony in certain interval

        message_queue       msg_queue;

        struct openssl_decrypt      *decrypt; ///< decrypt state

        std::future<bool> reconfiguration_future;
        bool             reconfiguration_in_progress;
};

/**
 * This function blocks until video frame is displayed and decoder::frame
 * can be filled with new data. Until this point, the video frame is not considered
 * valid.
 */
static void wait_for_framebuffer_swap(struct state_video_decoder *decoder) {
        pthread_mutex_lock(&decoder->lock);
        {
                while (!decoder->buffer_swapped) {
                        pthread_cond_wait(&decoder->buffer_swapped_cv, &decoder->lock);
                }
        }
        pthread_mutex_unlock(&decoder->lock);
}

#define ENCRYPTED_ERR "Receiving encrypted video data but " \
        "no decryption key entered!\n"
#define NOT_ENCRYPTED_ERR "Receiving unencrypted video data " \
        "while expecting encrypted.\n"
#define ERROR_GOTO_CLEANUP ret = FALSE; goto cleanup;

static void *fec_thread(void *args) {
        struct state_video_decoder *decoder =
                (struct state_video_decoder *) args;

        fec *fec_state = NULL;
        struct fec_desc desc(FEC_NONE);

        while(1) {
                struct fec_msg *data = NULL;
                data = dynamic_cast<fec_msg *>(decoder->fec_queue.pop());

                if(data->poisoned) {
                        decompress_msg *msg = new decompress_msg(0);
                        msg->poisoned = true;
                        decoder->decompress_queue.push(msg);
                        delete data;
                        break; // exit from loop
                }

                struct video_frame *frame = decoder->frame;
                struct tile *tile = NULL;
                int ret = TRUE;
                struct decompress_msg *decompress_msg = new struct decompress_msg( data->substream_count);
                memcpy(decompress_msg->buffer_num, data->buffer_num, data->substream_count * sizeof(int));
                memcpy(decompress_msg->fec_buffers, data->recv_buffers, data->substream_count * sizeof(char *));

                if (data->fec_description.type != FEC_NONE) {
                        if(!fec_state || desc.k != data->fec_description.k ||
                                        desc.m != data->fec_description.m ||
                                        desc.c != data->fec_description.c ||
                                        desc.seed != data->fec_description.seed
                          ) {
                                delete fec_state;
                                desc = data->fec_description;
                                fec_state = fec::create_from_desc(desc);
                                if(fec_state == NULL) {
                                        fprintf(stderr, "[decoder] Unable to initialize LDGM.\n");
                                        exit_uv(1);
                                        goto cleanup;
                                }
                        }
                }

                if (data->fec_description.type != FEC_NONE) {
                        int pos;
                        for (pos = 0; pos < get_video_mode_tiles_x(decoder->video_mode)
                                        * get_video_mode_tiles_y(decoder->video_mode); ++pos) {
                                char *fec_out_buffer = NULL;
                                int fec_out_len = 0;

                                fec_state->decode(data->recv_buffers[pos],
                                                data->buffer_len[pos],
                                                &fec_out_buffer, &fec_out_len, data->pckt_list[pos]);

                                if (data->buffer_len[pos] != (int) sum_map(data->pckt_list[pos])) {
                                        verbose_msg("Frame incomplete - substream %d, buffer %d: expected %u bytes, got %u.\n", pos,
                                                        (unsigned int) data->buffer_num[pos],
                                                        data->buffer_len[pos],
                                                        (unsigned int) sum_map(data->pckt_list[pos]));
                                }

                                if(fec_out_len == 0) {
                                        ret = FALSE;
                                        fprintf(stderr, "[decoder] FEC: unable to reconstruct data.\n");
                                        decoder->fec_nok += 1;
                                        goto cleanup;
                                }
                                decoder->fec_ok += 1;

                                video_payload_hdr_t video_hdr;
                                memcpy(&video_hdr, fec_out_buffer,
                                                sizeof(video_payload_hdr_t));
                                fec_out_buffer += sizeof(video_payload_hdr_t);
                                fec_out_len -= sizeof(video_payload_hdr_t);

                                struct video_desc network_desc;
                                parse_video_hdr(video_hdr, &network_desc);
                                if (!video_desc_eq_excl_param(decoder->received_vid_desc,
                                                        network_desc, PARAM_TILE_COUNT)) {
                                        main_msg *msg = new main_msg_reconfigure(network_desc);
                                        decoder->msg_queue.push(msg);
                                        ERROR_GOTO_CLEANUP
                                }

                                if(!frame) {
                                        ERROR_GOTO_CLEANUP
                                }

                                if(decoder->decoder_type == EXTERNAL_DECODER) {
                                        decompress_msg->buffer_len[pos] = fec_out_len;
                                        decompress_msg->decompress_buffer[pos] = fec_out_buffer;
                                } else { // linedecoder
                                        wait_for_framebuffer_swap(decoder);

                                        struct video_frame *out_frame;
                                        int divisor;

                                        if(!decoder->postprocess) {
                                                out_frame = frame;
                                        } else {
                                                out_frame = decoder->pp_frame;
                                        }

                                        if (!decoder->merged_fb) {
                                                divisor = decoder->max_substreams;
                                        } else {
                                                divisor = 1;
                                        }

                                        tile = vf_get_tile(out_frame, pos % divisor);

                                        struct line_decoder *line_decoder =
                                                &decoder->line_decoder[pos];

                                        int data_pos = 0;
                                        char *src = fec_out_buffer;
                                        char *dst = tile->data + line_decoder->base_offset;
                                        while(data_pos < (int) fec_out_len) {
                                                line_decoder->decode_line((unsigned char*)dst, (unsigned char *) src, line_decoder->src_linesize,
                                                                line_decoder->shifts[0],
                                                                line_decoder->shifts[1],
                                                                line_decoder->shifts[2]);
                                                src += line_decoder->src_linesize;
                                                dst += vc_get_linesize(tile->width ,frame->color_spec);
                                                data_pos += line_decoder->src_linesize;
                                        }
                                }
                        }
                } else { /* PT_VIDEO */
                        bool corrupted_frame_counted = false;
                        for(int i = 0; i < (int) decoder->max_substreams; ++i) {
                                decompress_msg->buffer_len[i] = data->buffer_len[i];
                                decompress_msg->decompress_buffer[i] = data->recv_buffers[i];

                                if (data->buffer_len[i] != (int) sum_map(data->pckt_list[i])) {
                                        verbose_msg("Frame incomplete - substream %d, buffer %d: expected %u bytes, got %u. ", i,
                                                        (unsigned int) data->buffer_num[i],
                                                        data->buffer_len[i],
                                                        (unsigned int) sum_map(data->pckt_list[i]));
                                        if(decoder->decoder_type == EXTERNAL_DECODER && !decoder->accepts_corrupted_frame) {
                                                ret = FALSE;
                                                verbose_msg("dropped.\n");
                                                goto cleanup;
                                        } else if (!corrupted_frame_counted) {
                                                corrupted_frame_counted = true;
                                                // count it here because decoder accepts corrupted frames
                                                decoder->corrupted++;
                                        }
                                        verbose_msg("\n");
                                }
                        }
                }

                pthread_mutex_lock(&decoder->lock);
                {
                        decoder->buffer_swapped = false;
                }
                pthread_mutex_unlock(&decoder->lock);
                decoder->decompress_queue.push(decompress_msg);

cleanup:
                if(ret == FALSE) {
                        for(int i = 0; i < data->substream_count; ++i) {
                                free(data->recv_buffers[i]);
                        }
                        decoder->corrupted++;
                        decoder->dropped++;
                }
                delete data;
        }

        delete fec_state;

        return NULL;
}

static void *decompress_thread(void *args) {
        struct state_video_decoder *decoder =
                (struct state_video_decoder *) args;

        struct tile *tile;

        while(1) {
                decompress_msg *msg = dynamic_cast<decompress_msg *>(decoder->decompress_queue.pop());

                if(msg->poisoned) {
                        delete msg;
                        break;
                }

                if(decoder->decoder_type == EXTERNAL_DECODER) {
                        int tile_width = decoder->received_vid_desc.width; // get_video_mode_tiles_x(decoder->video_mode);
                        int tile_height = decoder->received_vid_desc.height; // get_video_mode_tiles_y(decoder->video_mode);
                        int x, y;
                        struct video_frame *output;
                        if(decoder->postprocess) {
                                output = decoder->pp_frame;
                        } else {
                                output = decoder->frame;
                        }
                        for (x = 0; x < get_video_mode_tiles_x(decoder->video_mode); ++x) {
                                for (y = 0; y < get_video_mode_tiles_y(decoder->video_mode); ++y) {
                                        int pos = x + get_video_mode_tiles_x(decoder->video_mode) * y;
                                        char *out;
                                        if(decoder->merged_fb) {
                                                tile = vf_get_tile(output, 0);
                                                // TODO: OK when rendering directly to display FB, otherwise, do not reflect pitch (we use PP)
                                                out = tile->data + y * decoder->pitch * tile_height +
                                                        vc_get_linesize(tile_width, decoder->out_codec) * x;
                                        } else {
                                                tile = vf_get_tile(output, x);
                                                out = tile->data;
                                        }
                                        if(!msg->decompress_buffer[pos])
                                                continue;
                                        decompress_frame(decoder->decompress_state[pos],
                                                        (unsigned char *) out,
                                                        (unsigned char *) msg->decompress_buffer[pos],
                                                        msg->buffer_len[pos],
                                                        msg->buffer_num[pos]);
                                }
                        }
                }

                if(decoder->postprocess) {
                        int i;
                        bool pp_ret;

                        pp_ret = vo_postprocess(decoder->postprocess,
                                        decoder->pp_frame,
                                        decoder->frame,
                                        decoder->display_pitch);

                        if(!pp_ret) {
                                decoder->pp_frame = vo_postprocess_getf(decoder->postprocess);
                                goto skip_frame;
                        }

                        for (i = 1; i < decoder->pp_output_frames_count; ++i) {
                                display_put_frame(decoder->display, decoder->frame, PUTF_BLOCKING);
                                decoder->frame = display_get_frame(decoder->display);
                                pp_ret = vo_postprocess(decoder->postprocess,
                                                NULL,
                                                decoder->frame,
                                                decoder->display_pitch);
                                if(!pp_ret) {
                                        decoder->pp_frame = vo_postprocess_getf(decoder->postprocess);
                                        goto skip_frame;
                                }
                        }

                        /* get new postprocess frame */
                        decoder->pp_frame = vo_postprocess_getf(decoder->postprocess);
                }

                if(decoder->change_il) {
                        unsigned int i;
                        for(i = 0; i < decoder->frame->tile_count; ++i) {
                                struct tile *tile = vf_get_tile(decoder->frame, i);
                                decoder->change_il(tile->data, tile->data, vc_get_linesize(tile->width,
                                                        decoder->out_codec), tile->height);
                        }
                }

                {
                        int putf_flags = 0;

                        if(is_codec_interframe(decoder->received_vid_desc.color_spec)) {
                                putf_flags = PUTF_NONBLOCK;
                        }

                        int ret = display_put_frame(decoder->display,
                                        decoder->frame, putf_flags);
                        if (ret == 0) {
                                decoder->displayed++;
                                decoder->frame =
                                        display_get_frame(decoder->display);
                        } else {
                                decoder->dropped++;
                        }
                }

skip_frame:

                for(unsigned int i = 0; i < decoder->max_substreams; ++i) {
                        free(msg->fec_buffers[i]);
                }
                delete msg;

                pthread_mutex_lock(&decoder->lock);
                {
                        // we have put the video frame and requested another one which is
                        // writable so on
                        decoder->buffer_swapped = true;
                        pthread_cond_signal(&decoder->buffer_swapped_cv);
                }
                pthread_mutex_unlock(&decoder->lock);
        }

        return NULL;
}

static void decoder_set_video_mode(struct state_video_decoder *decoder, enum video_mode video_mode)
{
        decoder->video_mode = video_mode;
        decoder->max_substreams = get_video_mode_tiles_x(decoder->video_mode)
                        * get_video_mode_tiles_y(decoder->video_mode);
}

/**
 * @brief Initializes video decompress state.
 * @param video_mode  video_mode expected to be received from network
 * @param postprocess postprocessing specification, may be NULL
 * @param display     Video display that is supposed to be controlled from decoder.
 *                    display_get_frame(), display_put_frame(), display_get_propert()
 *                    and display_reconfigure() functions may be used. If set to NULL,
 *                    no decoding will take place.
 * @param encryption  Encryption config string. Currently, this is a passphrase to be
 *                    used. This may change eventually.
 * @return Newly created decoder state. If an error occured, returns NULL.
 * @ingroup video_rtp_decoder
 */
struct state_video_decoder *video_decoder_init(struct module *parent,
                enum video_mode video_mode, const char *postprocess,
                struct display *display, const char *encryption)
{
        struct state_video_decoder *s;

        s = (state_video_decoder *) calloc(1, sizeof(state_video_decoder));
        s = new(s) state_video_decoder; // call the constructor

        s->parent = parent;

        s->native_codecs = NULL;
        s->disp_supported_il = NULL;
        s->postprocess = NULL;
        s->change_il = NULL;

        s->displayed = s->dropped = s->corrupted = 0ul;
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->buffer_swapped_cv, NULL);

        s->buffer_swapped = true;
        s->last_buffer_number = -1;

        if (encryption) {
                if(openssl_decrypt_init(&s->decrypt,
                                                encryption, MODE_AES128_CTR) != 0) {
                        fprintf(stderr, "Unable to create decompress!\n");
                        return NULL;
                }
        }

        decoder_set_video_mode(s, video_mode);

        if(postprocess) {
                char *tmp_pp_config = strdup(postprocess);
                s->postprocess = vo_postprocess_init(tmp_pp_config);
                free(tmp_pp_config);
                if(strcmp(postprocess, "help") == 0) {
                        exit_uv(0);
                        return NULL;
                }
                if(!s->postprocess) {
                        fprintf(stderr, "Initializing postprocessor \"%s\" failed.\n", postprocess);
                        free(s);
                        exit_uv(129);
                        return NULL;
                }
        }

        if(!video_decoder_register_display(s, display))
                return NULL;

        return s;
}

static void restrict_returned_codecs(codec_t *display_codecs,
                size_t *display_codecs_count, codec_t *pp_codecs,
                int pp_codecs_count)
{
        int i;

        for (i = 0; i < (int) *display_codecs_count; ++i) {
                int j;

                int found = FALSE;

                for (j = 0; j < pp_codecs_count; ++j) {
                        if(display_codecs[i] == pp_codecs[j]) {
                                found = TRUE;
                        }
                }

                if(!found) {
                        memmove(&display_codecs[i], (const void *) &display_codecs[i + 1],
                                        sizeof(codec_t) * (*display_codecs_count - i - 1));
                        --*display_codecs_count;
                        --i;
                }
        }
}

/**
 * @brief Registers video display to be used for displaying decoded video frames.
 *
 * No display should be managed by this decoder when this function is called.
 *
 * @see decoder_remove_display
 *
 * @param decoder decoder state
 * @param display new display to be registered
 * @return        whether registration was successful
 */
bool video_decoder_register_display(struct state_video_decoder *decoder, struct display *display)
{
        assert(display != NULL);
        assert(decoder->display == NULL);

        int ret;

        decoder->display = display;

        free(decoder->native_codecs);
        decoder->native_count = 20 * sizeof(codec_t);
        decoder->native_codecs = (codec_t *)
                malloc(decoder->native_count * sizeof(codec_t));
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_CODECS, decoder->native_codecs, &decoder->native_count);
        decoder->native_count /= sizeof(codec_t);
        if(!ret) {
                fprintf(stderr, "Failed to query codecs from video display.\n");
                return false;
        }

        if(decoder->postprocess) {
                codec_t postprocess_supported_codecs[20];
                size_t len = sizeof(postprocess_supported_codecs);
                bool ret = vo_postprocess_get_property(decoder->postprocess, VO_PP_PROPERTY_CODECS,
                                (void *) postprocess_supported_codecs, &len);

                if(ret) {
                        if(len == 0) { // problem detected
                                fprintf(stderr, "[Decoder] Unable to get supported codecs.\n");
                                return false;

                        }
                        restrict_returned_codecs(decoder->native_codecs, &decoder->native_count,
                                        postprocess_supported_codecs, len / sizeof(codec_t));
                }
        }


        free(decoder->disp_supported_il);
        decoder->disp_supported_il_cnt = 20 * sizeof(enum interlacing_t);
        decoder->disp_supported_il = (enum interlacing_t*) calloc(decoder->disp_supported_il_cnt,
                        sizeof(enum interlacing_t));
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_SUPPORTED_IL_MODES, decoder->disp_supported_il, &decoder->disp_supported_il_cnt);
        if(ret) {
                decoder->disp_supported_il_cnt /= sizeof(enum interlacing_t);
        } else {
                enum interlacing_t tmp[] = { PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME}; /* default if not said othervise */
                memcpy(decoder->disp_supported_il, tmp, sizeof(tmp));
                decoder->disp_supported_il_cnt = sizeof(tmp) / sizeof(enum interlacing_t);
        }

        // Start decompress and ldmg threads
        if(pthread_create(&decoder->decompress_thread_id, NULL, decompress_thread,
                                decoder) != 0) {
                perror("Unable to create thread");
                return false;
        }

        if(pthread_create(&decoder->fec_thread_id, NULL, fec_thread,
                                decoder) != 0) {
                perror("Unable to create thread");
                return false;
        }

        return true;
}

/**
 * @brief This removes display from current decoder.
 *
 * From now on no video frames will be decoded with current decoder.
 * @see decoder_register_display - the counterpart of this functon
 *
 * @param decoder decoder from which will the decoder be removed
 */
void video_decoder_remove_display(struct state_video_decoder *decoder)
{
        if(decoder->display) {
                fec_msg *msg = new fec_msg(0);
                msg->poisoned = true;
                decoder->fec_queue.push(msg);

                pthread_join(decoder->fec_thread_id, NULL);
                pthread_join(decoder->decompress_thread_id, NULL);

                if (decoder->frame)
                        display_put_frame(decoder->display, decoder->frame,
                                        PUTF_NONBLOCK);

                decoder->display = NULL;
                decoder->frame = NULL;
                memset(&decoder->display_desc, 0, sizeof(decoder->display_desc));
        }
}

static void cleanup(struct state_video_decoder *decoder)
{
        decoder->decoder_type = UNSET;
        if(decoder->decompress_state) {
                for(unsigned int i = 0; i < decoder->max_substreams; ++i) {
                        decompress_done(decoder->decompress_state[i]);
                }
                free(decoder->decompress_state);
                decoder->decompress_state = NULL;
        }
        if(decoder->line_decoder) {
                free(decoder->line_decoder);
                decoder->line_decoder = NULL;
        }
}

#define PRINT_STATISTICS fprintf(stderr, "Video decoder statistics: %lu total: %lu displayed / %lu "\
                                "dropped / %lu corrupted / %lu missing frames.",\
                                decoder->displayed + decoder->dropped + decoder->missing, \
                                decoder->displayed, decoder->dropped, decoder->corrupted,\
                                decoder->missing); \
                         if (decoder->fec_ok + decoder->fec_nok > 0) fprintf(stderr, " LDGM OK/NOK: %ld/%ld", \
                         decoder->fec_ok, decoder->fec_nok); \
                         fprintf(stderr, "\n");

/**
 * @brief Destroys decoder created with decoder_init()
 * @param decoder decoder to be destroyed. If NULL, no action is performed.
 */
void video_decoder_destroy(struct state_video_decoder *decoder)
{
        if(!decoder)
                return;

        openssl_decrypt_destroy(decoder->decrypt);

        video_decoder_remove_display(decoder);

        pthread_mutex_destroy(&decoder->lock);

        cleanup(decoder);

        if(decoder->pp_frame) {
                vo_postprocess_done(decoder->postprocess);
                decoder->pp_frame = NULL;
        }

        free(decoder->native_codecs);
        free(decoder->disp_supported_il);

        PRINT_STATISTICS

        free(decoder);
}

/**
 * Attemps to initialize decompress of given magic
 *
 * @param[in]  magic      magic of the requested decompressor
 * @param[out] decompress_state decoder state
 * @param[in]  substreams number of decoders to be created
 * @return     true if initialization succeeded
 */
static bool try_initialize_decompress(uint32_t magic,
                struct state_decompress **decompress_state, int substreams)
{
        for(int i = 0; i < substreams; ++i) {
                decompress_state[i] = decompress_init(magic);

                if(!decompress_state[i]) {
                        debug_msg("Decompressor with magic %x was not found.\n");
                        for(int j = 0; j < substreams; ++j) {
                                if (decompress_state[i] != NULL)
                                        decompress_done(decompress_state[i]);
                                decompress_state[i] = NULL;
                        }
                        return false;
                }
        }

        return true;
}

/**
 * @param[in] in_codec input codec
 * @param[in] out_codec output codec
 * @param[in] prio_min minimal priority that can be probed
 * @param[in] prio_max maximal priority that can be probed
 * @param[out] magic if decompressor was found here is stored its magic
 * @retval -1       if no found
 * @retval priority best decoder's priority
 */
static int find_best_decompress(codec_t in_codec, codec_t out_codec,
                int prio_min, int prio_max, uint32_t *magic) {
        int trans;
        int best_priority = prio_max + 1;
        // first pass - find the one with best priority (least)
        for(trans = 0; trans < decoders_for_codec_count;
                        ++trans) {
                if(in_codec == decoders_for_codec[trans].from &&
                                out_codec == decoders_for_codec[trans].to) {
                        int priority = decoders_for_codec[trans].priority;
                        if(priority <= prio_max &&
                                        priority >= prio_min &&
                                        priority < best_priority) {
                                if(decompress_is_available(
                                                        decoders_for_codec[trans].decompress_index)) {
                                        best_priority = priority;
                                        *magic = decoders_for_codec[trans].decompress_index;
                                }
                        }
                }
        }

        if(best_priority == prio_max + 1)
                return -1;
        return best_priority;
}

/**
 * @brief Finds (best) decompress module for specified compression.
 *
 * If more than one decompress module is available, load the one with highest priority.
 *
 * @param[in] in_codec    source compression
 * @param[in] out_codec   requested destination pixelformat
 * @param[out] state      pointer (array) to be filled with state_count instances of decompressor
 * @param[in] state_count number of decompress states to be created.
 * This is important mainly for interlrame compressions which keeps internal state between individual
 * frames. Different tiles need to have different states then.
 * @retval true           if state_count members of state is filled with valid decompressor
 * @retval false          if initialization failed
 */
bool init_decompress(codec_t in_codec, codec_t out_codec,
                struct state_decompress **state, int state_count)
{
        int prio_max = 1000;
        int prio_min = 0;
        int prio_cur;
        uint32_t decompress_magic = 0u;

        while(1) {
                prio_cur = find_best_decompress(in_codec, out_codec,
                                prio_min, prio_max, &decompress_magic);
                // if found, init decoder
                if(prio_cur != -1) {
                        if(try_initialize_decompress(decompress_magic, state, state_count)) {
                                return true;
                        } else {
                                // failed, try to find another one
                                prio_min = prio_cur + 1;
                                continue;
                        }
                } else {
                        break;
                }
        }
        return false;
}

/**
 * This function selects, according to given video description, appropriate
 *
 * @param[in]  decoder     decoder to be taken parameters from (video mode, native codecs etc.)
 * @param[in]  desc        incoming video description
 * @param[out] decode_line If chosen decoder is a linedecoder, this variable contains the
 *                         decoding function.
 * @return                 Output codec, if no decoding function found, -1 is returned.
 */
static codec_t choose_codec_and_decoder(struct state_video_decoder *decoder, struct video_desc desc,
                                decoder_t *decode_line)
{
        codec_t out_codec = (codec_t) -1;
        *decode_line = NULL;

        size_t native;
        /* first check if the codec is natively supported */
        for(native = 0u; native < decoder->native_count; ++native)
        {
                out_codec = decoder->native_codecs[native];
                if(desc.color_spec == out_codec) {
                        if((out_codec == DXT1 || out_codec == DXT1_YUV ||
                                        out_codec == DXT5)
                                        && decoder->video_mode != VIDEO_NORMAL)
                                continue; /* it is a exception, see NOTES #1 */
                        if(desc.color_spec == RGBA || /* another exception - we may change shifts */
                                        desc.color_spec == RGB)
                                continue;

                        *decode_line = (decoder_t) memcpy;
                        decoder->decoder_type = LINE_DECODER;

                        goto after_linedecoder_lookup;
                }
        }
        /* otherwise if we have line decoder */
        int trans;
        for(trans = 0; line_decoders[trans].line_decoder != NULL;
                                ++trans) {
                for(native = 0; native < decoder->native_count; ++native)
                {
                        out_codec = decoder->native_codecs[native];
                        if(desc.color_spec == line_decoders[trans].from &&
                                        out_codec == line_decoders[trans].to) {

                                *decode_line = line_decoders[trans].line_decoder;

                                decoder->decoder_type = LINE_DECODER;
                                goto after_linedecoder_lookup;
                        }
                }
        }

after_linedecoder_lookup:

        /* we didn't find line decoder. So try now regular (aka DXT) decoder */
        if(*decode_line == NULL) {
                for(native = 0; native < decoder->native_count; ++native)
                {
                        out_codec = decoder->native_codecs[native];
                        decoder->decompress_state = (struct state_decompress **)
                                calloc(decoder->max_substreams, sizeof(struct state_decompress *));
                        if(init_decompress(desc.color_spec, decoder->native_codecs[native],
                                                decoder->decompress_state,
                                                decoder->max_substreams)) {
                                int res = 0, ret;
                                size_t size = sizeof(res);
                                ret = decompress_get_property(decoder->decompress_state[0],
                                                DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME,
                                                &res,
                                                &size);
                                if(ret && res) {
                                        decoder->accepts_corrupted_frame = TRUE;
                                } else {
                                        decoder->accepts_corrupted_frame = FALSE;
                                }

                                decoder->decoder_type = EXTERNAL_DECODER;
                                goto after_decoder_lookup;
                        } else {
                                free(decoder->decompress_state);
                                decoder->decompress_state = 0;
                        }
                }
        }
after_decoder_lookup:

        if(decoder->decoder_type == UNSET) {
                fprintf(stderr, "Unable to find decoder for input codec \"%s\"!!!\n", get_codec_name(desc.color_spec));
                exit_uv(128);
                return (codec_t) -1;
        }

        return out_codec;
}

/**
 * This function finds interlacing mode changing function.
 *
 * @param[in]  in_il       input_interlacing
 * @param[in]  supported   list of supported output interlacing modes
 * @param[in]  il_out_cnt  count of supported items
 * @param[out] out_il      selected output interlacing
 * @return                 selected interlacing changing function, NULL if not needed or not found
 */
static change_il_t select_il_func(enum interlacing_t in_il, enum interlacing_t *supported,
                int il_out_cnt, /*out*/ enum interlacing_t *out_il)
{
        struct transcode_t { enum interlacing_t in; enum interlacing_t out; change_il_t func; };

        struct transcode_t transcode[] = {
                {UPPER_FIELD_FIRST, INTERLACED_MERGED, il_upper_to_merged},
                {INTERLACED_MERGED, UPPER_FIELD_FIRST, il_merged_to_upper}
        };

        int i;
        /* first try to check if it can be nativelly displayed */
        for (i = 0; i < il_out_cnt; ++i) {
                if(in_il == supported[i]) {
                        *out_il = in_il;
                        return NULL;
                }
        }

        for (i = 0; i < il_out_cnt; ++i) {
                size_t j;
                for (j = 0; j < sizeof(transcode) / sizeof(struct transcode_t); ++j) {
                        if(in_il == transcode[j].in && supported[i] == transcode[j].out) {
                                *out_il = transcode[j].out;
                                return transcode[j].func;
                        }
                }
        }

        return NULL;
}

/**
 * Reconfigures decoder if network received video data format has changed.
 *
 * @param decoder       the video decoder state
 * @param desc          new video description to be reconfigured to
 * @return              boolean value if reconfiguration was successful
 */
static bool reconfigure_decoder(struct state_video_decoder *decoder,
                struct video_desc desc)
{
        codec_t out_codec;
        decoder_t decode_line;
        enum interlacing_t display_il = PROGRESSIVE;
        //struct video_frame *frame;
        int render_mode;

        // this code forces flushing the pipelined data
        struct display *tmp_display = decoder->display;
        video_decoder_remove_display(decoder);
        video_decoder_register_display(decoder, tmp_display);

        assert(decoder != NULL);
        assert(decoder->native_codecs != NULL);

        cleanup(decoder);

        desc.tile_count = get_video_mode_tiles_x(decoder->video_mode)
                        * get_video_mode_tiles_y(decoder->video_mode);

        out_codec = choose_codec_and_decoder(decoder, desc, &decode_line);
        if(out_codec == (codec_t) -1)
                return false;
        else
                decoder->out_codec = out_codec;
        struct video_desc display_desc = desc;

        int display_mode;
        size_t len = sizeof(int);
        int ret;

        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_VIDEO_MODE,
                        &display_mode, &len);
        if(!ret) {
                debug_msg("Failed to get video display mode.");
                display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        }

        bool pp_does_change_tiling_mode = false;

        if (decoder->postprocess) {
                size_t len = sizeof(pp_does_change_tiling_mode);
                if(vo_postprocess_get_property(decoder->postprocess, VO_PP_DOES_CHANGE_TILING_MODE,
                                        &pp_does_change_tiling_mode, &len)) {
                        if(len == 0) {
                                // just for sake of completness since it shouldn't be a case
                                fprintf(stderr, "[Decoder] Warning: unable to get pp tiling mode!\n");
                        }
                }
        }

        if(decoder->postprocess) {
                struct video_desc pp_desc = desc;
                pp_desc.color_spec = out_codec;
                if(!pp_does_change_tiling_mode) {
                        pp_desc.width *= get_video_mode_tiles_x(decoder->video_mode);
                        pp_desc.height *= get_video_mode_tiles_y(decoder->video_mode);
                        pp_desc.tile_count = 1;
                }
                vo_postprocess_reconfigure(decoder->postprocess, pp_desc);
                decoder->pp_frame = vo_postprocess_getf(decoder->postprocess);
                vo_postprocess_get_out_desc(decoder->postprocess, &display_desc, &render_mode, &decoder->pp_output_frames_count);
        } else {
                if(display_mode == DISPLAY_PROPERTY_VIDEO_MERGED) {
                        display_desc.width *= get_video_mode_tiles_x(decoder->video_mode);
                        display_desc.height *= get_video_mode_tiles_y(decoder->video_mode);
                        display_desc.tile_count = 1;
                }
        }

        decoder->change_il = select_il_func(desc.interlacing, decoder->disp_supported_il,
                        decoder->disp_supported_il_cnt, &display_il);

        if (!decoder->postprocess || !pp_does_change_tiling_mode) { /* otherwise we need postprocessor mode, which we obtained before */
                render_mode = display_mode;
        }

        display_desc.color_spec = out_codec;
        display_desc.interlacing = display_il;

        if(!video_desc_eq(decoder->display_desc, display_desc))
        {
                int ret;
                /*
                 * TODO: put frame should be definitely here. On the other hand, we cannot be sure
                 * that vo driver is initialized so far:(
                 */
                //display_put_frame(decoder->display, frame);
                /* reconfigure VO and give it opportunity to pass us pitch */
                ret = display_reconfigure(decoder->display, display_desc);
                if(!ret) {
                        fprintf(stderr, "[decoder] Unable to reconfigure display.\n");
                        exit_uv(128);
                        return false;
                }
                decoder->display_desc = display_desc;
        }
        /*if(decoder->postprocess) {
                frame = decoder->pp_frame;
        } else {
                frame = frame_display;
        }*/

        len = sizeof(decoder->display_requested_rgb_shift);
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_RGB_SHIFT,
                        &decoder->display_requested_rgb_shift, &len);
        if(!ret) {
                debug_msg("Failed to get r,g,b shift property from video driver.\n");
                int rgb_shift[3] = {0, 8, 16};
                memcpy(&decoder->display_requested_rgb_shift, rgb_shift, sizeof(rgb_shift));
        }

        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_BUF_PITCH,
                        &decoder->display_requested_pitch, &len);
        if(!ret) {
                debug_msg("Failed to get pitch from video driver.\n");
                decoder->display_requested_pitch = PITCH_DEFAULT;
        }

        int linewidth;
        if(render_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
                linewidth = desc.width;
        } else {
                linewidth = desc.width * get_video_mode_tiles_x(decoder->video_mode);
        }

        if(!decoder->postprocess) {
                if(decoder->display_requested_pitch == PITCH_DEFAULT)
                        decoder->pitch = vc_get_linesize(linewidth, out_codec);
                else
                        decoder->pitch = decoder->display_requested_pitch;
        } else {
                decoder->pitch = vc_get_linesize(linewidth, out_codec);
        }

        if(decoder->display_requested_pitch == PITCH_DEFAULT) {
                decoder->display_pitch = vc_get_linesize(display_desc.width, out_codec);
        } else {
                decoder->display_pitch = decoder->display_requested_pitch;
        }

        int src_x_tiles = get_video_mode_tiles_x(decoder->video_mode);
        int src_y_tiles = get_video_mode_tiles_y(decoder->video_mode);

        if(decoder->decoder_type == LINE_DECODER) {
                decoder->line_decoder = (struct line_decoder *) malloc(src_x_tiles * src_y_tiles *
                                        sizeof(struct line_decoder));
                if(render_mode == DISPLAY_PROPERTY_VIDEO_MERGED && decoder->video_mode == VIDEO_NORMAL) {
                        struct line_decoder *out = &decoder->line_decoder[0];
                        out->base_offset = 0;
                        out->src_bpp = get_bpp(desc.color_spec);
                        out->dst_bpp = get_bpp(out_codec);
                        memcpy(out->shifts, decoder->display_requested_rgb_shift, 3 * sizeof(int));

                        out->decode_line = decode_line;
                        out->dst_pitch = decoder->pitch;
                        out->src_linesize = vc_get_linesize(desc.width, desc.color_spec);
                        out->dst_linesize = vc_get_linesize(desc.width, out_codec);
                        decoder->merged_fb = TRUE;
                } else if(render_mode == DISPLAY_PROPERTY_VIDEO_MERGED
                                && decoder->video_mode != VIDEO_NORMAL) {
                        int x, y;
                        for(x = 0; x < src_x_tiles; ++x) {
                                for(y = 0; y < src_y_tiles; ++y) {
                                        struct line_decoder *out = &decoder->line_decoder[x +
                                                        src_x_tiles * y];
                                        out->base_offset = y * (desc.height)
                                                        * decoder->pitch +
                                                        vc_get_linesize(x * desc.width, out_codec);

                                        out->src_bpp = get_bpp(desc.color_spec);
                                        out->dst_bpp = get_bpp(out_codec);
                                        memcpy(out->shifts, decoder->display_requested_rgb_shift,
                                                        3 * sizeof(int));

                                        out->decode_line = decode_line;

                                        out->dst_pitch = decoder->pitch;
                                        out->src_linesize =
                                                vc_get_linesize(desc.width, desc.color_spec);
                                        out->dst_linesize =
                                                vc_get_linesize(desc.width, out_codec);
                                }
                        }
                        decoder->merged_fb = TRUE;
                } else if (render_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
                        int x, y;
                        for(x = 0; x < src_x_tiles; ++x) {
                                for(y = 0; y < src_y_tiles; ++y) {
                                        struct line_decoder *out = &decoder->line_decoder[x +
                                                        src_x_tiles * y];
                                        out->base_offset = 0;
                                        out->src_bpp = get_bpp(desc.color_spec);
                                        out->dst_bpp = get_bpp(out_codec);
                                        memcpy(out->shifts, decoder->display_requested_rgb_shift,
                                                        3 * sizeof(int));

                                        out->decode_line = decode_line;
                                        out->src_linesize =
                                                vc_get_linesize(desc.width, desc.color_spec);
                                        out->dst_pitch =
                                                out->dst_linesize =
                                                vc_get_linesize(desc.width, out_codec);
                                }
                        }
                        decoder->merged_fb = FALSE;
                }
        } else if (decoder->decoder_type == EXTERNAL_DECODER) {
                int buf_size;

                for(unsigned int i = 0; i < decoder->max_substreams; ++i) {
                        buf_size = decompress_reconfigure(decoder->decompress_state[i], desc,
                                        decoder->display_requested_rgb_shift[0],
                                        decoder->display_requested_rgb_shift[1],
                                        decoder->display_requested_rgb_shift[2],
                                        decoder->pitch,
                                        out_codec);
                        if(!buf_size) {
                                return false;
                        }
                }
                if(render_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
                        decoder->merged_fb = FALSE;
                } else {
                        decoder->merged_fb = TRUE;
                }
        }

        // Pass metadata to receiver thread (it can tweak parameters)
        struct msg_receiver *msg = (struct msg_receiver *)
                new_message(sizeof(struct msg_receiver));
        msg->type = RECEIVER_MSG_VIDEO_PROP_CHANGED;
        msg->new_desc = decoder->received_vid_desc;
        struct response *resp =
                send_message_to_receiver(decoder->parent, (struct message *) msg);
        resp->deleter(resp);

        return true;
}

static bool parse_video_hdr(uint32_t *hdr, struct video_desc *desc)
{
        uint32_t tmp;
        int fps_pt, fpsd, fd, fi;

        desc->width = ntohl(hdr[3]) >> 16;
        desc->height = ntohl(hdr[3]) & 0xffff;
        desc->color_spec = get_codec_from_fcc(hdr[4]);
        if(desc->color_spec == (codec_t) -1) {
                fprintf(stderr, "Unknown FourCC \"%4s\"!\n", (char *) &hdr[4]);
                return false;
        }

        tmp = ntohl(hdr[5]);
        desc->interlacing = (enum interlacing_t) (tmp >> 29);
        fps_pt = (tmp >> 19) & 0x3ff;
        fpsd = (tmp >> 15) & 0xf;
        fd = (tmp >> 14) & 0x1;
        fi = (tmp >> 13) & 0x1;

        desc->fps = compute_fps(fps_pt, fpsd, fd, fi);

        return true;
}

static int reconfigure_if_needed(struct state_video_decoder *decoder,
                struct video_desc network_desc)
{
        if (!video_desc_eq_excl_param(decoder->received_vid_desc, network_desc, PARAM_TILE_COUNT)) {
                printf("New incoming video format detected: %dx%d @%.2f%s, codec %s\n",
                                network_desc.width, network_desc.height, network_desc.fps,
                                get_interlacing_suffix(network_desc.interlacing),
                                get_codec_name(network_desc.color_spec));
                decoder->received_vid_desc = network_desc;

                decoder->reconfiguration_in_progress = true;
                decoder->reconfiguration_future = std::async(std::launch::async,
                                [&decoder](){ return reconfigure_decoder(decoder, decoder->received_vid_desc); });

                return TRUE;
        }
        return FALSE;
}
/**
 * Checks if network format has changed.
 *
 * @param decoder decoder state
 * @param hdr     raw RTP payload header
 *
 * @return TRUE if format changed (and reconfiguration was successful), FALSE if not
 */
static int check_for_mode_change(struct state_video_decoder *decoder,
                uint32_t *hdr)
{
        struct video_desc network_desc;

        parse_video_hdr(hdr, &network_desc);
        return reconfigure_if_needed(decoder, network_desc);
}

#define ERROR_GOTO_CLEANUP ret = FALSE; goto cleanup;
#define max(a, b)       (((a) > (b))? (a): (b))

/**
 * @brief Decodes a participant buffer representing one video frame.
 * @param cdata        PBUF buffer
 * @param decoder_data @ref vcodec_state containing decoder state and some additional data
 * @retval TRUE        if decoding was successful.
 *                     It stil doesn't mean that the frame will be correctly displayed,
 *                     decoding may fail in some subsequent (asynchronous) steps.
 * @retval FALSE       if decoding failed
 */
int decode_video_frame(struct coded_data *cdata, void *decoder_data)
{
        struct vcodec_state *pbuf_data = (struct vcodec_state *) decoder_data;
        struct state_video_decoder *decoder = pbuf_data->decoder;

        int ret = TRUE;
        uint32_t offset;
        int len;
        rtp_packet *pckt = NULL;
        unsigned char *source;
        char *data;
        uint32_t data_pos;
        int prints=0;
        struct tile *tile = NULL;
        uint32_t tmp;
        uint32_t substream;
        int max_substreams = decoder->max_substreams;

        int i;
        map<int, int> *pckt_list;
        uint32_t buffer_len[max_substreams];
        uint32_t buffer_num[max_substreams];
        // the following is just LDGM related optimalization - normally we fill up
        // allocated buffers when we have compressed data. But in case of LDGM, there
        // is just the LDGM buffer present, so we point to it instead to copying
        char *recv_buffers[max_substreams]; // for FEC or compressed data
        pckt_list = new map<int, int>[max_substreams];
        for (i = 0; i < (int) max_substreams; ++i) {
                buffer_len[i] = 0;
                buffer_num[i] = 0;
                recv_buffers[i] = NULL;
        }

        perf_record(UVP_DECODEFRAME, cdata);

        // We have no framebuffer assigned, exitting
        if(!decoder->display) {
                return FALSE;
        }

        int k = 0, m = 0, c = 0, seed = 0; // LDGM
        int buffer_number, buffer_length;

        // check if we are not in the middle of reconfiguration
        if (decoder->reconfiguration_in_progress) {
#if defined __GNUC__ && __GNUC__ == 4 && __GNUC_MINOR__ == 6
                bool status =
#else
                std::future_status status =
#endif
                        decoder->reconfiguration_future.wait_until(std::chrono::system_clock::now());
#if defined __GNUC__ && __GNUC__ == 4 && __GNUC_MINOR__ == 6
                if (status) {
#else
                if (status == std::future_status::ready) {
#endif
                        bool ret = decoder->reconfiguration_future.get();
                        if (ret) {
                                decoder->frame = display_get_frame(decoder->display);
                        } else {
                                fprintf(stderr, "Decoder reconfiguration failed!!!\n");
                                decoder->frame = NULL;
                        }
                        decoder->reconfiguration_in_progress = false;
                } else {
                        // skip the frame if we are not yet reconfigured
                        return FALSE;
                }
        }

        main_msg *msg;
        while ((msg = dynamic_cast<main_msg *>(decoder->msg_queue.pop(true /* nonblock */)))) {
                if (dynamic_cast<main_msg_reconfigure *>(msg)) {
                        main_msg_reconfigure *msg_reconf = dynamic_cast<main_msg_reconfigure *>(msg);
                        if (reconfigure_if_needed(decoder, msg_reconf->desc)) {
                                return FALSE;
                        }
                }
                delete msg;
        }

        int pt;
        bool buffer_swapped = false;
        struct fec_msg *fec_msg = NULL;

        while (cdata != NULL) {
                uint32_t *hdr;
                pckt = cdata->data;

                pt = pckt->pt;
                hdr = (uint32_t *)(void *) pckt->data;
                data_pos = ntohl(hdr[1]);
                tmp = ntohl(hdr[0]);
                substream = tmp >> 22;
                buffer_number = tmp & 0x3ffff;
                buffer_length = ntohl(hdr[2]);

                if (pt == PT_VIDEO_LDGM || pt == PT_ENCRYPT_VIDEO_LDGM) {
                        tmp = ntohl(hdr[3]);
                        k = tmp >> 19;
                        m = 0x1fff & (tmp >> 6);
                        c = 0x3f & tmp;
                        seed = ntohl(hdr[4]);
                }

                if (pt == PT_ENCRYPT_VIDEO || pt == PT_ENCRYPT_VIDEO_LDGM) {
                        if(!decoder->decrypt) {
                                fprintf(stderr, ENCRYPTED_ERR);
                                ERROR_GOTO_CLEANUP
                        }
                } else {
                        if(decoder->decrypt) {
                                fprintf(stderr, NOT_ENCRYPTED_ERR);
                                ERROR_GOTO_CLEANUP
                        }
                }

                switch (pt) {
                case PT_VIDEO:
                        len = pckt->data_len - sizeof(video_payload_hdr_t);
                        data = (char *) hdr + sizeof(video_payload_hdr_t);
                        break;
                case PT_VIDEO_LDGM:
                        len = pckt->data_len - sizeof(fec_video_payload_hdr_t);
                        data = (char *) hdr + sizeof(fec_video_payload_hdr_t);
                        break;
                case PT_ENCRYPT_VIDEO:
                        len = pckt->data_len - sizeof(video_payload_hdr_t)
                                - sizeof(crypto_payload_hdr_t);
                        data = (char *) hdr + sizeof(video_payload_hdr_t)
                                + sizeof(crypto_payload_hdr_t);
                        break;
                case PT_ENCRYPT_VIDEO_LDGM:
                        len = pckt->data_len - sizeof(fec_video_payload_hdr_t)
                                - sizeof(crypto_payload_hdr_t);
                        data = (char *) hdr + sizeof(fec_video_payload_hdr_t)
                                + sizeof(crypto_payload_hdr_t);
                        break;
                default:
                        fprintf(stderr, "[decoder] Unknown packet type: %d.\n", pckt->pt);
                        exit_uv(1);
                        ret = FALSE;
                        goto cleanup;
                }

                if ((int) substream >= max_substreams) {
                        fprintf(stderr, "[decoder] received substream ID %d. Expecting at most %d substreams. Did you set -M option?\n",
                                        substream, max_substreams);
                        // the guess is valid - we start with highest substream number (anytime - since it holds a m-bit)
                        // in next iterations, index is valid
                        enum video_mode video_mode =
                                guess_video_mode(substream + 1);
                        if (video_mode != VIDEO_UNKNOWN) {
                                fprintf(stderr, "[decoder] Guessing mode: ");
                                decoder_set_video_mode(decoder, video_mode);
                                decoder->received_vid_desc.width = 0; // just for sure, that we reconfigure in next iteration
                                fprintf(stderr, "%s. Check if it is correct.\n", get_video_mode_description(decoder->video_mode));
                        } else {
                                exit_uv(1);
                        }
                        // we need skip this frame (variables are illegal in this iteration
                        // and in case that we got unrecognized number of substreams - exit
                        ret = FALSE;
                        goto cleanup;
                }

                if(!recv_buffers[substream]) {
                        recv_buffers[substream] = (char *) malloc(buffer_length);
                }

                buffer_num[substream] = buffer_number;
                buffer_len[substream] = buffer_length;

                char plaintext[len]; // will be actually shorter
                if(pt == PT_ENCRYPT_VIDEO || pt == PT_ENCRYPT_VIDEO_LDGM) {
                        int data_len;

                        if((data_len = openssl_decrypt(decoder->decrypt,
                                        data, len,
                                        (char *) hdr, pt == PT_ENCRYPT_VIDEO ?
                                        sizeof(video_payload_hdr_t) : sizeof(fec_video_payload_hdr_t),
                                        plaintext)) == 0) {
                                fprintf(stderr, "Warning: Packet dropped AES - wrong CRC!\n");
                                goto next_packet;
                        }
                        data = (char *) plaintext;
                        len = data_len;
                }

                if (pt == PT_VIDEO || pt == PT_ENCRYPT_VIDEO)
                {
                        /* Critical section
                         * each thread *MUST* wait here if this condition is true
                         */
                        if (check_for_mode_change(decoder, hdr)) {
                                return FALSE;
                        }

                        // hereafter, display framebuffer can be used, so we
                        // check if we got it
                        if (decoder->frame == NULL) {
                                return FALSE;
                        }
                }

                pckt_list[substream][data_pos] = len;

                if ((pt == PT_VIDEO || pt == PT_ENCRYPT_VIDEO) && decoder->decoder_type == LINE_DECODER) {
                        if(!buffer_swapped) {
                                wait_for_framebuffer_swap(decoder);
                                buffer_swapped = true;
                        }

                        if(!decoder->postprocess) {
                                if (!decoder->merged_fb) {
                                        tile = vf_get_tile(decoder->frame, substream);
                                } else {
                                        tile = vf_get_tile(decoder->frame, 0);
                                }
                        } else {
                                if (!decoder->merged_fb) {
                                        tile = vf_get_tile(decoder->pp_frame, substream);
                                } else {
                                        tile = vf_get_tile(decoder->pp_frame, 0);
                                }
                        }

                        struct line_decoder *line_decoder =
                                &decoder->line_decoder[substream];

                        /* End of critical section */

                        /* MAGIC, don't touch it, you definitely break it
                         *  *source* is data from network, *destination* is frame buffer
                         */

                        /* compute Y pos in source frame and convert it to
                         * byte offset in the destination frame
                         */
                        int y = (data_pos / line_decoder->src_linesize) * line_decoder->dst_pitch;

                        /* compute X pos in source frame */
                        int s_x = data_pos % line_decoder->src_linesize;

                        /* convert X pos from source frame into the destination frame.
                         * it is byte offset from the beginning of a line.
                         */
                        int d_x = ((int)((s_x) / line_decoder->src_bpp)) *
                                line_decoder->dst_bpp;

                        /* pointer to data payload in packet */
                        source = (unsigned char*)(data);

                        /* copy whole packet that can span several lines.
                         * we need to clip data (v210 case) or center data (RGBA, R10k cases)
                         */
                        while (len > 0) {
                                /* len id payload length in source BPP
                                 * decoder needs len in destination BPP, so convert it
                                 */
                                int l = ((int)(len / line_decoder->src_bpp)) * line_decoder->dst_bpp;

                                /* do not copy multiple lines, we need to
                                 * copy (& clip, center) line by line
                                 */
                                if (l + d_x > (int) line_decoder->dst_linesize) {
                                        l = line_decoder->dst_linesize - d_x;
                                }

                                /* compute byte offset in destination frame */
                                offset = y + d_x;

                                /* watch the SEGV */
                                if (l + line_decoder->base_offset + offset <= tile->data_len) {
                                        /*decode frame:
                                         * we have offset for destination
                                         * we update source contiguously
                                         * we pass {r,g,b}shifts */
                                        line_decoder->decode_line((unsigned char*)tile->data + line_decoder->base_offset + offset, source, l,
                                                        line_decoder->shifts[0], line_decoder->shifts[1],
                                                        line_decoder->shifts[2]);
                                        /* we decoded one line (or a part of one line) to the end of the line
                                         * so decrease *source* len by 1 line (or that part of the line */
                                        len -= line_decoder->src_linesize - s_x;
                                        /* jump in source by the same amount */
                                        source += line_decoder->src_linesize - s_x;
                                } else {
                                        /* this should not ever happen as we call reconfigure before each packet
                                         * iff reconfigure is needed. But if it still happens, something is terribly wrong
                                         * say it loudly
                                         */
                                        if((prints % 100) == 0) {
                                                fprintf(stderr, "WARNING!! Discarding input data as frame buffer is too small.\n"
                                                                "Well this should not happened. Expect troubles pretty soon.\n");
                                        }
                                        prints++;
                                        len = 0;
                                }
                                /* each new line continues from the beginning */
                                d_x = 0;        /* next line from beginning */
                                s_x = 0;
                                y += line_decoder->dst_pitch;  /* next line */
                        }
                } else { /* PT_VIDEO_LDGM or external decoder */
                        memcpy(recv_buffers[substream] + data_pos, (unsigned char*) data,
                                len);
                }

next_packet:
                cdata = cdata->nxt;
        }

        if(!pckt) {
                ret = FALSE;
                goto cleanup;
        }

        if (decoder->frame == NULL && (pt == PT_VIDEO || pt == PT_ENCRYPT_VIDEO)) {
                ret = FALSE;
                goto cleanup;
        }


        assert(ret == TRUE);

        // format message
        fec_msg = new struct fec_msg(max_substreams);
        fec_msg->fec_description = fec_desc(fec::fec_type_from_pt(pt), k, m, c, seed);
        fec_msg->poisoned = false;
        memcpy(fec_msg->buffer_len, buffer_len, sizeof(buffer_len));
        memcpy(fec_msg->buffer_num, buffer_num, sizeof(buffer_num));
        memcpy(fec_msg->recv_buffers, recv_buffers, sizeof(recv_buffers));
        fec_msg->pckt_list = pckt_list;

        if(decoder->fec_queue.size() > 0) {
                decoder->slow_msg.print("Your computer may be too SLOW to play this !!!");
        }
        decoder->fec_queue.push(fec_msg);
cleanup:
        ;
        unsigned int frame_size = 0;

        for(i = 0; i < (int) max_substreams; ++i) {
                if(ret != TRUE) {
                        free(recv_buffers[i]);
                }

                frame_size += buffer_len[i];
        }

        if(ret != TRUE) {
                delete [] pckt_list;
        }

        pbuf_data->max_frame_size = max(pbuf_data->max_frame_size, frame_size);
        pbuf_data->decoded++;

        /// @todo figure out multiple substreams
        if (decoder->last_buffer_number != -1) {
                long int missing = buffer_number -
                        ((decoder->last_buffer_number + 1) & 0x3fffff);
                missing = (missing + 0x3fffff) % 0x3fffff;
                decoder->missing += missing;
        }
        decoder->last_buffer_number = buffer_number;

        if(decoder->displayed % 600 == 599) {
                PRINT_STATISTICS
        }

        return ret;
}

