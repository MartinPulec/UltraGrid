/*
 * FILE:    video_codec.h
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */
#ifndef __video_decompress_h

#define __video_decompress_h
#include "video_codec.h"

#include "Frame.h"
#include "video_decompress/jpeg.h"

struct state_decompress;

/**
 * initializes decompression and returns internal state
 */
typedef  void *(*decompress_init_t)();
/**
 * Recompresses decompression for specified video description
 */
typedef  int (*decompress_reconfigure_t)(void * state, struct video_desc desc, 
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec);
/**
 * Decompresses data from buffer of src_len into dst
 */
typedef void (*decompress_push_t)(void *, std::tr1::shared_ptr<Frame> buffer);
typedef std::tr1::shared_ptr<Frame> (*decompress_pop_t)(void *);
/**
 * Cleanup function
 */
typedef  void (*decompress_done_t)(void *);


struct decode_from_to {
        codec_t from;
        codec_t to;

        uint32_t decompress_index;
};
extern struct decode_from_to decoders_for_codec[];
extern const int decoders_for_codec_count;


void initialize_video_decompress(void);

struct state_decompress *decompress_init(unsigned int decoder_index);
int decompress_reconfigure(struct state_decompress *, struct video_desc, int rshift, int gshift, int bshift, int pitch, codec_t out_codec);
void decompress_push(struct state_decompress *, std::tr1::shared_ptr<Frame> buffer);
std::tr1::shared_ptr<Frame> decompress_pop(struct state_decompress *);
void decompress_done(struct state_decompress *);

#endif /* __video_decompress_h */
