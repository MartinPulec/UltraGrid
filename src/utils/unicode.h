// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#ifndef UTILS_UNICODE_H_21A466F2_6B6F_4EF2_83CE_AE364FEFDEAE
#define UTILS_UNICODE_H_21A466F2_6B6F_4EF2_83CE_AE364FEFDEAE

#ifdef __cplusplus
#include <cstddef> // for size_t
#include <cwchar>  // for wchar_t
#else
#include <stddef.h>  // for size_t
#include <wchar.h>   // for wchar_t
#endif

#include "compat/c23.h" // IWYU pragma: keep for bool

#ifdef __cplusplus
extern "C" {
#endif

// the encoding of L"" is implementation defined but almost always UTF-8
#define W_REGISTERED_SIGN  L"\xC2\xAE"
#define W_DEGREE_SIGN      L"\xC2\xB0"
#define W_SUPERSCRIPT_ONE  L"\xC2\xB9"
#define W_LARGE_RED_CIRCLE L"\xF0\x9F\x94\xB4"

void        u8_out_init(bool is_win_utf8_terminal);
const char *wcs_to_mbs_buf(const wchar_t *wstr, size_t buflen,
                           char *out_fallback);
// convenience macro casting char8_t ptr to (const unsigned char *) + len from
// buf
#define wcs_to_mbs_fallb(wstr, out_fallback)                                       \
        wcs_to_mbs_buf((wstr), sizeof(out_fallback), \
                       (out_fallback))

#ifdef __cplusplus
}
#endif

#endif // defined UTILS_UNICODE_H_21A466F2_6B6F_4EF2_83CE_AE364FEFDEAE
