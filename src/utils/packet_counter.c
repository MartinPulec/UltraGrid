/**
 * @file   utils/packet_counter.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2026 CESNET, zájmové sdružení právnických osob
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

#include "utils/packet_counter.h"

#include <assert.h>   // for assert
#include <inttypes.h> // for uint32_t, uint16_t
#include <limits.h>   // for ULONG_MAX
#include <stdint.h>
#include <stdlib.h> // for free, size_t, calloc, qsort, realloc

#include "compat/c23.h"   // IWYU pragma: keep
#include "utils/macros.h" // for to_fourcc

#define MAGIC to_fourcc('U', 'T', 'p', 'c')

enum {
        PC_MAX_SUBSTREAMS = 256,
};

typedef struct pc_packet packet;

struct packet_counter {
        uint32_t magic;
        int      num_substreams;

        struct pc_packet *packets;
        size_t            packets_allocated;
        size_t            packets_count;

        bool stats_generated;
        struct {
                long          expected;
                long          received;
                const packet *past_last;
        } stats[PC_MAX_SUBSTREAMS];
        long expected_cumul;
        long received_cumul;
};

struct packet_counter *
packet_counter_init()
{
        struct packet_counter *s = calloc(1, sizeof *s);
        s->magic                 = MAGIC;
        return s;
}

void
packet_counter_destroy(struct packet_counter *s)
{
        if (s == nullptr) {
                return;
        }
        assert(s->magic == MAGIC);
        free(s->packets);
        free(s);
}

void
packet_counter_register_packet(struct packet_counter *s,
                               unsigned int substream_id, unsigned int bufnum,
                               unsigned int offset, unsigned int len)
{
        assert(len <= UINT16_MAX);
        if (s->packets_count == s->packets_allocated) {
                s->packets_allocated = 2 * (s->packets_allocated + 1);
                s->packets = realloc(s->packets, s->packets_allocated *
                                                     sizeof(struct pc_packet));
                assert(s->packets != nullptr);
        }
        s->packets[s->packets_count].substream_id  = substream_id;
        s->packets[s->packets_count].packet_len    = len;
        s->packets[s->packets_count].buffer_number = bufnum;
        s->packets[s->packets_count].offset        = offset;
        s->packets_count += 1;
}

static int
compare(const void *a, const void *b)
{
        const struct pc_packet *packet_a = a;
        const struct pc_packet *packet_b = b;
        if (packet_a->substream_id != packet_b->substream_id) {
                return packet_a->substream_id - packet_b->substream_id;
        }
        if (packet_a->buffer_number != packet_b->buffer_number) {
                return packet_a->buffer_number < packet_b->buffer_number ? -1
                                                                         : 1;
        }
        if (packet_a->offset != packet_b->offset) {
                return packet_a->offset < packet_b->offset ? -1 : 1;
        }
        if (packet_a->packet_len != packet_b->packet_len) {
                return packet_a->packet_len - packet_b->packet_len;
        }

        return 0;
}

/**
 * sort, rm dups, generate stats and substream pointers
 */
static void
process_packets(struct packet_counter *s)
{
        if (s->stats_generated) {
                return;
        }

        qsort(s->packets, s->packets_count, sizeof s->packets[0], compare);
        // remove duplicates
        if (s->packets_count > 1) {
                unsigned long wr_index = 1;
                for (unsigned long i = 1; i < s->packets_count; ++i) {
                        const struct pc_packet *p = s->packets + i;
                        if (p->substream_id != p[-1].substream_id ||
                            p->buffer_number != p[-1].buffer_number ||
                            p->offset != p[-1].offset) {
                                s->packets[wr_index++] = *p;
                        }
                }
                s->packets_count = wr_index;
        }

        long expected = 0;
        long received = 0;


        memset(s->stats, 0, sizeof s->stats);
        s->expected_cumul = 0;
        s->received_cumul = 0;

        for (unsigned long i = 0; i < s->packets_count; ++i) {
                const struct pc_packet *p = s->packets + i;
                received += p->packet_len;
                // last packet of a buffer in a substeram (count expected as its
                // offset+len); buffers with no packes can therefor not be
                // counted, also not detected are missing packets at the end of
                // the buffer
                if (i == s->packets_count - 1 ||
                    p[1].substream_id != p->substream_id ||
                    p[1].buffer_number != p->buffer_number) {
                        assert(p->substream_id < PC_MAX_SUBSTREAMS);
                        expected += p->offset + p->packet_len;
                        s->stats[p->substream_id].expected = expected;
                        s->stats[p->substream_id].received = received;
                        s->stats[p->substream_id].past_last = p + 1;
                        s->expected_cumul += expected;
                        s->received_cumul += received;
                        expected = 0;
                        received = 0;
                }
        }
        s->stats_generated = true;
}

/**
 * get cumulative status for all non-empty substreams (see the note)
 *
 * @see packet_counter_get_bytes_per_ss
 *
 * @note
 * The reported number of _expected_ bytes may be lees than the actual, see the
 * inline comment in process_packets() because off+len of last packet in substream
 * is used but this may be lost.
 * <br>
 * Usually it works ok for big buffers but not for small containing few or even
 * one packet (as in audio).
 * <br>
 * If possible, prefer `data_len` header from packet, which is reliable.
 */
void
packet_counter_get_bytes(struct packet_counter *s, long *expected,
                         long *received) {
        process_packets(s);
        *expected = s->expected_cumul;
        *received = s->received_cumul;
}

/**
 * get packets stats for given substream ID
 *
 * @note
 * see the note in packet_counter_get_bytes()
 *
 * @param[in]  substream_id  substream ID
 * @param[out] expected      number of expected bytes
 * @param[out] received      actual number of received bytes
 */
void
packet_counter_get_bytes_per_ss(struct packet_counter *s, unsigned substream_id,
                                long *expected, long *received)
{
        process_packets(s);
        *expected = s->stats[substream_id].expected;
        *received = s->stats[substream_id].received;
}

void
packet_counter_clear(struct packet_counter *s)
{
        s->packets_count   = 0;
        s->stats_generated = false;
}

unsigned
packet_counter_get_packets(struct packet_counter *s, unsigned substream_id,
                           const struct pc_packet **packets)
{
        process_packets(s);
        const packet *const past_last = s->stats[substream_id].past_last;
        if (!past_last) {
                return 0;
        }
        if (substream_id == 0) {
                *packets = s->packets;
                return past_last - s->packets;
        }
        // find our first packet (typically the previous substream past last if
        // substream not empty)
        const packet *prev_end          = s->stats[0].past_last;
        unsigned      prev_substream_id = substream_id - 1;
        while (prev_substream_id > 0) { // find prev non-empty channel
                if (s->stats[prev_substream_id].past_last) {
                        prev_end = s->stats[prev_substream_id].past_last;
                        break;
                }
                prev_substream_id -= 1;
        }
        *packets = prev_end;
        return past_last - prev_end;
}
