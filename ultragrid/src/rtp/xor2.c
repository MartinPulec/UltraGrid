/*
 * FILE:     xor2.c
 * AUTHOR:   Martin Pulec <pulec@cesnet.cz>
 *
 * The routines in this file implement the Real-time Transport Protocol,
 * RTP, as specified in RFC1889 with current updates under discussion in
 * the IETF audio/video transport working group. Portions of the code are
 * derived from the algorithms published in that specification.
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
 * Copyright (c) 1998-2001 University College London
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
 *      This product includes software developed by the Computer Science
 *      Department at University College London and by the University of
 *      Southern California Information Sciences Institute. This product also
 *      includes software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University, Department, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *    
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *
 */

/**
 * Algorithm implemented as described here:
 * http://blogs.oracle.com/ahl/entry/double_parity_raid_z
 *
 * Reference implementation:
 * http://fxr.watson.org/fxr/source/common/fs/zfs/vdev_raidz.c?v=OPENSOLARIS
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "memory.h"
#include "debug.h"
#include "net_udp.h"
#include "crypto/random.h"
#include "compat/drand48.h"
#include "compat/gettimeofday.h"
#include "crypto/crypt_des.h"
#include "crypto/crypt_aes.h"
#include "tv.h"
#include "crypto/md5.h"
#include "ntp.h"
#include "xor2.h"

/*
 * These two tables represent powers and logs of 2 in the Galois field defined
 * above. These values were computed by repeatedly multiplying by 2 as above.
 */
static const uint8_t vdev_raidz_pow2[256] = {
	0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
	0x1d, 0x3a, 0x74, 0xe8, 0xcd, 0x87, 0x13, 0x26,
	0x4c, 0x98, 0x2d, 0x5a, 0xb4, 0x75, 0xea, 0xc9,
	0x8f, 0x03, 0x06, 0x0c, 0x18, 0x30, 0x60, 0xc0,
	0x9d, 0x27, 0x4e, 0x9c, 0x25, 0x4a, 0x94, 0x35,
	0x6a, 0xd4, 0xb5, 0x77, 0xee, 0xc1, 0x9f, 0x23,
	0x46, 0x8c, 0x05, 0x0a, 0x14, 0x28, 0x50, 0xa0,
	0x5d, 0xba, 0x69, 0xd2, 0xb9, 0x6f, 0xde, 0xa1,
	0x5f, 0xbe, 0x61, 0xc2, 0x99, 0x2f, 0x5e, 0xbc,
	0x65, 0xca, 0x89, 0x0f, 0x1e, 0x3c, 0x78, 0xf0,
	0xfd, 0xe7, 0xd3, 0xbb, 0x6b, 0xd6, 0xb1, 0x7f,
	0xfe, 0xe1, 0xdf, 0xa3, 0x5b, 0xb6, 0x71, 0xe2,
	0xd9, 0xaf, 0x43, 0x86, 0x11, 0x22, 0x44, 0x88,
	0x0d, 0x1a, 0x34, 0x68, 0xd0, 0xbd, 0x67, 0xce,
	0x81, 0x1f, 0x3e, 0x7c, 0xf8, 0xed, 0xc7, 0x93,
	0x3b, 0x76, 0xec, 0xc5, 0x97, 0x33, 0x66, 0xcc,
	0x85, 0x17, 0x2e, 0x5c, 0xb8, 0x6d, 0xda, 0xa9,
	0x4f, 0x9e, 0x21, 0x42, 0x84, 0x15, 0x2a, 0x54,
	0xa8, 0x4d, 0x9a, 0x29, 0x52, 0xa4, 0x55, 0xaa,
	0x49, 0x92, 0x39, 0x72, 0xe4, 0xd5, 0xb7, 0x73,
	0xe6, 0xd1, 0xbf, 0x63, 0xc6, 0x91, 0x3f, 0x7e,
	0xfc, 0xe5, 0xd7, 0xb3, 0x7b, 0xf6, 0xf1, 0xff,
	0xe3, 0xdb, 0xab, 0x4b, 0x96, 0x31, 0x62, 0xc4,
	0x95, 0x37, 0x6e, 0xdc, 0xa5, 0x57, 0xae, 0x41,
	0x82, 0x19, 0x32, 0x64, 0xc8, 0x8d, 0x07, 0x0e,
	0x1c, 0x38, 0x70, 0xe0, 0xdd, 0xa7, 0x53, 0xa6,
	0x51, 0xa2, 0x59, 0xb2, 0x79, 0xf2, 0xf9, 0xef,
	0xc3, 0x9b, 0x2b, 0x56, 0xac, 0x45, 0x8a, 0x09,
	0x12, 0x24, 0x48, 0x90, 0x3d, 0x7a, 0xf4, 0xf5,
	0xf7, 0xf3, 0xfb, 0xeb, 0xcb, 0x8b, 0x0b, 0x16,
	0x2c, 0x58, 0xb0, 0x7d, 0xfa, 0xe9, 0xcf, 0x83,
	0x1b, 0x36, 0x6c, 0xd8, 0xad, 0x47, 0x8e, 0x01
};
static const uint8_t vdev_raidz_log2[256] = {
	0x00, 0x00, 0x01, 0x19, 0x02, 0x32, 0x1a, 0xc6,
	0x03, 0xdf, 0x33, 0xee, 0x1b, 0x68, 0xc7, 0x4b,
	0x04, 0x64, 0xe0, 0x0e, 0x34, 0x8d, 0xef, 0x81,
	0x1c, 0xc1, 0x69, 0xf8, 0xc8, 0x08, 0x4c, 0x71,
	0x05, 0x8a, 0x65, 0x2f, 0xe1, 0x24, 0x0f, 0x21,
	0x35, 0x93, 0x8e, 0xda, 0xf0, 0x12, 0x82, 0x45,
	0x1d, 0xb5, 0xc2, 0x7d, 0x6a, 0x27, 0xf9, 0xb9,
	0xc9, 0x9a, 0x09, 0x78, 0x4d, 0xe4, 0x72, 0xa6,
	0x06, 0xbf, 0x8b, 0x62, 0x66, 0xdd, 0x30, 0xfd,
	0xe2, 0x98, 0x25, 0xb3, 0x10, 0x91, 0x22, 0x88,
	0x36, 0xd0, 0x94, 0xce, 0x8f, 0x96, 0xdb, 0xbd,
	0xf1, 0xd2, 0x13, 0x5c, 0x83, 0x38, 0x46, 0x40,
	0x1e, 0x42, 0xb6, 0xa3, 0xc3, 0x48, 0x7e, 0x6e,
	0x6b, 0x3a, 0x28, 0x54, 0xfa, 0x85, 0xba, 0x3d,
	0xca, 0x5e, 0x9b, 0x9f, 0x0a, 0x15, 0x79, 0x2b,
	0x4e, 0xd4, 0xe5, 0xac, 0x73, 0xf3, 0xa7, 0x57,
	0x07, 0x70, 0xc0, 0xf7, 0x8c, 0x80, 0x63, 0x0d,
	0x67, 0x4a, 0xde, 0xed, 0x31, 0xc5, 0xfe, 0x18,
	0xe3, 0xa5, 0x99, 0x77, 0x26, 0xb8, 0xb4, 0x7c,
	0x11, 0x44, 0x92, 0xd9, 0x23, 0x20, 0x89, 0x2e,
	0x37, 0x3f, 0xd1, 0x5b, 0x95, 0xbc, 0xcf, 0xcd,
	0x90, 0x87, 0x97, 0xb2, 0xdc, 0xfc, 0xbe, 0x61,
	0xf2, 0x56, 0xd3, 0xab, 0x14, 0x2a, 0x5d, 0x9e,
	0x84, 0x3c, 0x39, 0x53, 0x47, 0x6d, 0x41, 0xa2,
	0x1f, 0x2d, 0x43, 0xd8, 0xb7, 0x7b, 0xa4, 0x76,
	0xc4, 0x17, 0x49, 0xec, 0x7f, 0x0c, 0x6f, 0xf6,
	0x6c, 0xa1, 0x3b, 0x52, 0x29, 0x9d, 0x55, 0xaa,
	0xfb, 0x60, 0x86, 0xb1, 0xbb, 0xcc, 0x3e, 0x5a,
	0xcb, 0x59, 0x5f, 0xb0, 0x9c, 0xa9, 0xa0, 0x51,
	0x0b, 0xf5, 0x16, 0xeb, 0x7a, 0x75, 0x2c, 0xd7,
	0x4f, 0xae, 0xd5, 0xe9, 0xe6, 0xe7, 0xad, 0xe8,
	0x74, 0xd6, 0xf4, 0xea, 0xa8, 0x50, 0x58, 0xaf,
};

/*
 * Multiply a given number by 2 raised to the given power.
 */
static inline uint8_t
vdev_raidz_exp2(uint8_t a, int exp)
{
	if (a == 0u)
		return 0u;

	exp += vdev_raidz_log2[a];
	if (exp > 255u)
		exp -= 255u;

	return vdev_raidz_pow2[exp];
}

struct xor2_pkt_hdr {
        uint32_t pkt_count;
#ifdef WORDS_BIGENDIAN
        uint16_t header_len;
        uint16_t payload_len;
#else
        uint16_t payload_len;
        uint16_t header_len;
#endif
}__attribute__((packed));


struct xor2_session {
        char            *P;
        char            *Q;

        char            *tmp;

        int              header_len;
        int              max_payload_len;
        int              pkt_count;

        struct xor2_pkt_hdr xor2_hdr;
};

struct xor2_session * xor2_init(int header_len, int max_payload_len)
{
        struct xor2_session *s;
        assert (header_len % 4 == 0);
        /* TODO: check if it shouldn't be less */
        assert(header_len + max_payload_len + 40 + sizeof(struct xor2_pkt_hdr) < 9000);

        s = (struct xor2_session *) malloc(sizeof(struct xor2_session));
        s->pkt_count = 0;
        s->header_len = header_len;
        /* calloc is really needed here, because we will xor2 with this place incomming packets */
        s->P = (char *) calloc(1, header_len + max_payload_len);
        s->Q = (char *) calloc(1, header_len + max_payload_len);
        s->tmp = (char *) malloc(header_len + max_payload_len);
        s->max_payload_len = max_payload_len;

        return s;
}

/*
 * We add firstly D_{n-1} , then D_{n-2} etc. in order to support arbitrary number of packets.
 */
void xor2_add_packet(struct xor2_session *session, const char *hdr, const char *payload, int payload_len)
{
        int linepos;
        register unsigned int *line1;
        register const unsigned int *line2;

        session->pkt_count++;


        /* First compute P, which is quite straightforward (XOR) */

        line1 = (unsigned int *) session->P;
        line2 = (const unsigned int *) hdr;
        for(linepos = 0; linepos < session->header_len; linepos += 4) {
                *line1 ^= *line2;
                line1 += 1;
                line2 += 1;
        }

        line1 = (unsigned int *) ((char *) session->P + session->header_len);
        line2 = (const unsigned int *) payload;
        for(linepos = 0; linepos < (payload_len - 15); linepos += 16) {
                asm volatile ("movdqu (%0), %%xmm0\n"
                        "movdqu (%1), %%xmm1\n"
                        "pxor %%xmm1, %%xmm0\n"
                        "movdqu %%xmm0, (%0)\n"
                        ::"r" ((unsigned long *) line1),
                        "r"((unsigned long *) line2));
                line1 += 4;
                line2 += 4;
        }
        if(linepos != payload_len) {
                char *line1c = line1;
                char *line2c = line1;
                for(; linepos < payload_len; linepos += 1) {
                        *line1c ^= *line2c;
                        line1c += 1;
                        line2c += 1;
                }
        }

        /* then Q */
        line1 = (unsigned int *) session->tmp;
        line2 = (const unsigned int *) hdr;
        for(linepos = 0; linepos < session->header_len; linepos++) {

        }



}

void xor2_emit_xor2_packet(struct xor2_session *session, const char **hdr, size_t *hdr_len, const char **payload, size_t *payload_len)
{
        session->xor2_hdr.pkt_count = htonl(session->pkt_count);
        session->xor2_hdr.header_len = htons(session->header_len);
        session->xor2_hdr.payload_len = htons(session->max_payload_len);

        *hdr = (char *)  &session->xor2_hdr;
        *hdr_len = (size_t) sizeof(struct xor2_pkt_hdr);
        *payload = (const char *) session->header_xor2;
        *payload_len = (size_t) (session->header_len + session->max_payload_len);
}

void xor2_clear(struct xor2_session *session)
{
        session->pkt_count = 0;
        memset(session->header_xor2, 0, session->header_len + session->max_payload_len);
}

void xor2_destroy(struct xor2_session * session)
{
        free(session->header_xor2);
        free(session);
}


struct xor2_session *xor2_restore_init()
{
        struct xor2_session *session;
        session = (struct xor2_session *) malloc(sizeof(struct xor2_session));
        return session;
}

void xor2_restore_start(struct xor2_session *session, const char *data)
{
        session->pkt_count = - ntohl(((struct xor2_pkt_hdr *) data)->pkt_count);
        session->header_len = ntohs(((struct xor2_pkt_hdr *) data)->header_len);
        session->max_payload_len = ntohs(((struct xor2_pkt_hdr *) data)->payload_len);

        session->header_xor2 = data + sizeof(struct xor2_pkt_hdr);
        session->payload_xor2 = data + sizeof(struct xor2_pkt_hdr) + session->header_len;
}

int xor2_restore_packet(struct xor2_session *session, char **pkt)
{
        if(session->pkt_count == 0)
        {
                /* no packet restored */
                return FALSE;
        }
        if(session->pkt_count < -1) {
                debug_msg("Restoring packed failed - missing data.\n");
                return FALSE;
        }
        if(session->pkt_count > -1) {
                debug_msg("Restoring packed failed - missing xor2.\n");
                return FALSE;
        }

        debug_msg("Restoring packed.\n");
        *pkt = session->header_xor2;
        return TRUE;
}

void xor2_restore_destroy(struct xor2_session *xor2)
{
        free(xor2);
}

void xor2_restore_invalidate(struct xor2_session *session)
{
        session->pkt_count = 0;
}

