/*
 * FILE:    aes_encrypt.c
 * AUTHOR:  Colin Perkins <csp@csperkins.org>
 *          Ladan Gharai
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2004 University of Southern California
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#ifdef HAVE_CRYPTO

#include "crypto/md5.h"
#include "crypto/openssl_aes_encrypt.h"

#include <string.h>
#ifndef HAVE_NETTLE
#include <openssl/aes.h>
#include <openssl/rand.h>
#else
#include <nettle/aes.h>
#endif
#include "crypto/random.h"

struct openssl_aes_encrypt {
        AES_KEY key;

        enum openssl_aes_mode mode;

        unsigned char ivec[16];
        unsigned int num;
        unsigned char ecount[16];
};

int openssl_aes_encrypt_init(struct openssl_aes_encrypt **state, const char *passphrase,
                enum openssl_aes_mode mode)
{
        struct openssl_aes_encrypt *s = (struct openssl_aes_encrypt *)
                calloc(1, sizeof(struct openssl_aes_encrypt));

        MD5_CTX context;
        unsigned char hash[16];

#ifndef HAVE_NETTLE
        if (!RAND_bytes(s->ivec, 8)) {
                return -1;
        }
#else
        lbl_srandom((int)(getpid() * 42) ^ (int)time((time_t *) 0));
        uint32_t rand_number = lbl_random();
        memcpy(&s->ivec, &rand_number, sizeof(uint32_t));
        rand_number = lbl_random();
        memcpy(&s->ivec, &rand_number, sizeof(uint32_t));
#endif

        MD5Init(&context);
        MD5Update(&context, (unsigned char *) passphrase,
                        strlen(passphrase));
        MD5Final(hash, &context);

        AES_set_encrypt_key(hash, 128, &s->key);
        s->mode = mode;

        *state = s;
        return 0;
}

void openssl_aes_encrypt_block(struct openssl_aes_encrypt *s, unsigned char *plaintext,
                unsigned char *ciphertext, char *nonce_plus_counter, int len)
{
        if(nonce_plus_counter) {
                memcpy(nonce_plus_counter, (char *) s->ivec, 16);
                /* We do start a new block so we zero the byte counter
                 * Please NB that counter doesn't need to be incremented
                 * because the counter is incremented everytime s->num == 0,
                 * presumably before encryption, so setting it to 0 forces
                 * counter increment as well.
                 */
                if(s->num != 0) {
                        s->num = 0;
                }
        }

        switch(s->mode) {
                case MODE_CTR:
                        AES_ctr128_encrypt(plaintext, ciphertext, len, &s->key, s->ivec,
                                        s->ecount, &s->num);
                        break;
                case MODE_ECB:
                        AES_ecb_encrypt(plaintext, ciphertext,
                                        &s->key
#ifndef HAVE_NETTLE
                                        , AES_ENCRYPT
#endif
                                        );
                        break;
                default:
                        abort();
        }
}

void openssl_aes_encrypt_destroy(struct openssl_aes_encrypt *s)
{
        free(s);
}

#endif //  HAVE_CRYPTO

