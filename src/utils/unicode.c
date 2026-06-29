// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

/**
 * @file
 * 
 */

#include "utils/unicode.h"

#ifdef HAVE_CONFIG_H
#include "config.h" // for HAVE_C8RTOMB
#endif

#include <assert.h> // for assert
#include <limits.h> // for INT_MAX, PATH_MAX
#include <locale.h> // for setlocale
#include <string.h> // for memcpy, strlen, strpbrk
#ifdef HAVE_C8RTOMB
#include <uchar.h> // for c8rtomb
#include <wchar.h> // for mbstate_t
#endif

#include "compat/c23.h" // IWYU pragma: keep
#include "debug.h"

#define MOD_NAME "[utils/unicode] "

static bool utf8_terminal = false;
void
u8_out_init(bool is_win_utf8_terminal)
{
#ifdef _WIN32
        utf8_terminal = is_win_utf8_terminal;
#else
        const char *lc_ctype = setlocale(LC_CTYPE, "");
        if (lc_ctype == nullptr) {
                MSG(WARNING, "Cannot set locale.");
        } else {
                MSG(DEBUG, "LC_CTYPE set to: %s\n", lc_ctype);
                utf8_terminal = strstr(lc_ctype, ".UTF-8") != nullptr;
        }
        (void) is_win_utf8_terminal;
#endif
}

/**
 * Tries to convert wide string to locale-specific multibyte string. If
 * conversion fails out_fallback is returned untouched (should contain fallback
 * text).
 *
 * @param buflen  out_fallback buffer length long enough to hold the converted
                  string
 * @param[in,out] out_fallback NUL-terminated fallback string. If conversion
 *                succeeds, it is rewritten by the u8_str converted to MBS
 * @returns out_fallback with converted data if conversion suceeds
 * @returns out_fallback unchanged if wstr not convertible
 *
 * u8_out_init() must be called otherwise fallback is always ret
 */
const char *
wcs_to_mbs_buf(const wchar_t *wstr, size_t buflen, char *out_fallback)
{
#ifdef _WIN32
        if (!utf8_terminal) {
                return out_fallback;
        }
#endif

        mbstate_t ps = { 0 };

        // check convertibility first
        size_t ret = wcsrtombs(nullptr, &wstr, 0, &ps);
        if (ret == (size_t) -1) { // not convertible
                return out_fallback;
        }
        // actual conversion
        wcsrtombs(out_fallback, &wstr, buflen, &ps);
        if (*wstr != L'\0') {
                MSG(WARNING, "wide string truncated.\n");
        }
        return out_fallback;
}
