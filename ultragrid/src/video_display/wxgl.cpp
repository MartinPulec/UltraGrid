/*
 * FILE:   video_display/wxgl.c
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *
 * Copyright (c) 2001-2003 University of Southern California
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
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 *
 */


extern "C" {
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "video.h"
#include "video_codec.h"
#include "video_display.h"
}

#include <tr1/memory>

#include "video_display/wxgl.h"

#include "client-gui/include/Player.h"
#include "client-gui/client_guiMain.h"

#define MAGIC_WXGL	0x1f87bd3a

struct state_wxgl {
        uint32_t magic;
        Player *player;
        struct video_frame *frame;
        struct tile *tile;
        std::tr1::shared_ptr<char> buffer_data;
};

void *display_wxgl_init(char *fmt, unsigned int flags)
{
    UNUSED(flags);
    struct state_wxgl *s;

    s = (struct state_wxgl *)calloc(1, sizeof(struct state_wxgl));
    if (s != NULL) {
        s->magic = MAGIC_WXGL;
        s->player = (Player *) fmt;
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        s->tile->data = NULL;
    }
    return s;
}

void display_wxgl_run(void *arg)
{
    UNUSED(arg);
}

void display_wxgl_finish(void *state)
{
    UNUSED(state);
}

void display_wxgl_done(void *state)
{
    struct state_wxgl *s = (struct state_wxgl *)state;
    assert(s->magic == MAGIC_WXGL);
    free(s->tile->data);
    vf_free(s->frame);
    free(s);
}

struct video_frame *display_wxgl_getf(void *state)
{
    struct state_wxgl *s = (struct state_wxgl *)state;
    assert(s->magic == MAGIC_WXGL);
    s->buffer_data = s->player->getframe();
    s->tile->data = s->buffer_data.get();
    return s->frame;
}

int display_wxgl_putf(void *state, char *frame)
{
    struct state_wxgl *s = (struct state_wxgl *)state;
    assert(s->magic == MAGIC_WXGL);

    s->player->putframe(s->buffer_data, s->frame->frames);
    s->buffer_data = std::tr1::shared_ptr<char>();
    return 0;
}

display_type_t *display_wxgl_probe(void)
{
        display_type_t *dt;

        dt = (display_type_t *) malloc(sizeof(display_type_t));
        if (dt != NULL) {
                dt->id = DISPLAY_WXGL_ID;
                dt->name = "wxgl";
                dt->description = "Dummy WXGL device";

                dt->devices = (struct display_device *) malloc(sizeof(struct display_device));
                dt->devices->name = NULL;
        }
        return dt;
}

int display_wxgl_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_wxgl *s = (struct state_wxgl *)state;
        codec_t codecs[] = {UYVY, RGBA, RGB, DXT1, DXT1_YUV, DXT5};
        // UYVY - currently not needed. perhaps also broken with GLView

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }

                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RSHIFT:
                        *(int *) val = 0;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_GSHIFT:
                        *(int *) val = 8;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BSHIFT:
                        *(int *) val = 16;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        *(int *) val = DISPLAY_PROPERTY_VIDEO_MERGED;
                        break;

                default:
                        return FALSE;
        }
        return TRUE;
}

int display_wxgl_reconfigure(void *state, struct video_desc desc)
{
        struct state_wxgl *s = (struct state_wxgl *)state;
        assert(s->magic == MAGIC_WXGL);
        int dxt_height = (desc.height + 3) / 4 * 4;

        if(desc.color_spec == DXT1 || desc.color_spec == DXT5 || desc.color_spec == DXT1_YUV)
            s->tile->data_len = dxt_height;
        else
            s->tile->data_len = desc.height;
        s->tile->data_len *= vc_get_linesize(desc.width, desc.color_spec);

        s->player->reconfigure(desc.width, desc.height, (int) desc.color_spec, s->tile->data_len);

        return TRUE;
}

struct audio_frame * display_wxgl_get_audio_frame(void *state)
{
        UNUSED(state);
        return NULL;
}

void display_wxgl_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

int display_wxgl_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

