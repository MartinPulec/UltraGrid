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
#include "aes_encrypt.h"

#include <string.h>

#include "defs.h"

aes_encrypt::aes_encrypt()
{
        int i, nrounds = 5;
        unsigned char key[32], iv[32];
        unsigned int salt[] = AES_SALT;

        /*
         * Gen key & IV for AES 256 CBC mode. A SHA1 digest is used to hash the supplied key material.
         * nrounds is the number of times the we hash the material. More rounds are more secure but
         * slower.
         */
        i = EVP_BytesToKey(EVP_aes_256_cbc(), EVP_sha1(), (const unsigned char *) salt,
                        (const unsigned char *) AES_KEY_MATERIAL, strlen(AES_KEY_MATERIAL),
                        nrounds, key, iv);
        if (i != 32) {
                printf("Key size is %d bits - should be 256 bits\n", i);
                throw;
        }

        EVP_CIPHER_CTX_init(&m_context);
        EVP_EncryptInit_ex(&m_context, EVP_aes_256_cbc(), NULL, key, iv);
}

unsigned char *aes_encrypt::encrypt(unsigned char *plaintext, int *len)
{
        /* max ciphertext len for a n bytes of plaintext is n + AES_BLOCK_SIZE -1 bytes */
        int c_len = *len + AES_BLOCK_SIZE, f_len = 0;
        unsigned char *ciphertext = (unsigned char *) malloc(c_len);

        /* allows reusing of 'e' for multiple encryption cycles */
        EVP_EncryptInit_ex(&m_context, NULL, NULL, NULL, NULL);

        /* update ciphertext, c_len is filled with the length of ciphertext generated,
         *len is the size of plaintext in bytes */
        EVP_EncryptUpdate(&m_context, ciphertext, &c_len, plaintext, *len);

        /* update ciphertext with the final remaining bytes */
        EVP_EncryptFinal_ex(&m_context, ciphertext+c_len, &f_len);

        *len = c_len + f_len;
        return ciphertext;
}

