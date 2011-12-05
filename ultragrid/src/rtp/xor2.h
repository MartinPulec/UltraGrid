/*
 * FILE:   xor2.h
 * AUTHOR: Martin Pulec <pulec@cesnet.cz>
 *
 * Copyright (c) 1998-2000 University College London
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Computer Science
 *      Department at University College London.
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
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
 */

#ifndef __XOR2_H__
#define __XOR2_H__

struct xor2_session;

struct xor2_session * xor2_init(int header_len, int max_payload_len);
void xor2_add_packet(struct xor2_session *session, const char *hdr, const char *payload, int payload_len);
void xor2_add_packet(struct xor2_session *session, const char *hdr, const char *payload, int payload_len);
void xor2_emit_xor2_packet_p(struct xor2_session *session, const char **hdr, size_t *hdr_len, const char **payload, size_t *payload_len);
void xor2_emit_xor2_packet_q(struct xor2_session *session, const char **hdr, size_t *hdr_len, const char **payload, size_t *payload_len);
void xor2_clear(struct xor2_session *session);
void xor2_destroy(struct xor2_session * session);


struct xor2_session *xor2_restore_init();
void xor2_restore_start(struct xor2_session *session, const char *data);
int xor2_restore_packet(struct xor2_session *session, char **pkt);
void xor2_restore_destroy(struct xor2_session *xor2);
void xor2_restore_invalidate(struct xor2_session *xor2);

#endif /* __XOR2_H__ */
