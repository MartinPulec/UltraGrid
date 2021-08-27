/**
 * @file   utils/packet_counter.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012 CESNET, z. s. p. o.
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
 
#ifndef __PACKET_COUNTER_H
#define __PACKET_COUNTER_H

struct packet_counter;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct packet_counter *packet_counter_init(int num_substreams);
void packet_counter_destroy(struct packet_counter *state);
void packet_counter_register_packet(struct packet_counter *state, unsigned int substream_id,
                unsigned int bufnum, unsigned int offset, unsigned int len);
bool packet_counter_has_packet(struct packet_counter *state, unsigned int substream_id,
                unsigned int bufnum, unsigned int offset, unsigned int len);
int packet_counter_get_total_bytes(struct packet_counter *state);
int packet_counter_get_all_bytes(struct packet_counter *state);
int packet_counter_get_channels(struct packet_counter *state);
void packet_counter_clear(struct packet_counter *state);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __PACKET_COUNTER_H */
