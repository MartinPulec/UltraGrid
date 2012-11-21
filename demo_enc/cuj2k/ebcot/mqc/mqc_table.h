/* 
 * Copyright (c) 2011, Martin Srom
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MQC_TABLE_H
#define MQC_TABLE_H

/**
 * Lookup table for MQ-Coder
 *
 * Each row of lookup table is MQ-Coder's state that can be associated to any context.
 * Each context CX is associated to any state.
 */

#include <stdint.h>

/**
 * Context state
 */
struct mqc_cxstate {
    // The probability of the Less Probable Symbol (0.75->0x8000, 1.5->0xffff)
    uint32_t qeval;
    // The Most Probable Symbol (0 or 1)
    uint8_t mps;
    // Number of shifts needed to be at least 0x8000
    uint8_t ns;
    // Next state index if the next encoded symbol is the MPS
    uint8_t nmps;
    // Next state index if the next encoded symbol is the LPS
    uint8_t nlps;
};

/**
 * Size of lookup table
 */
#define mqc_table_size (47 * 2)

/**
 * Static definition of lookup table for MQ-Coder in CPU memory
 */
static struct mqc_cxstate mqc_table[mqc_table_size] = {
    {0x5601, 0, 1,  2,  3},     {0x5601, 1, 1,  3,  2},
    {0x3401, 0, 2,  4,  12},    {0x3401, 1, 2,  5,  13},
    {0x1801, 0, 3,  6,  18},    {0x1801, 1, 3,  7,  19},
    {0x0ac1, 0, 4,  8,  24},    {0x0ac1, 1, 4,  9,  25},
    {0x0521, 0, 5,  10, 58},    {0x0521, 1, 5,  11, 59},
    {0x0221, 0, 6,  76, 66},    {0x0221, 1, 6,  77, 67},
    {0x5601, 0, 1,  14, 13},    {0x5601, 1, 1,  15, 12},
    {0x5401, 0, 1,  16, 28},    {0x5401, 1, 1,  17, 29},
    {0x4801, 0, 1,  18, 28},    {0x4801, 1, 1,  19, 29},
    {0x3801, 0, 2,  20, 28},    {0x3801, 1, 2,  21, 29},
    {0x3001, 0, 2,  22, 34},    {0x3001, 1, 2,  23, 35},
    {0x2401, 0, 2,  24, 36},    {0x2401, 1, 2,  25, 37},
    {0x1c01, 0, 3,  26, 40},    {0x1c01, 1, 3,  27, 41},
    {0x1601, 0, 3,  58, 42},    {0x1601, 1, 3,  59, 43},
    {0x5601, 0, 1,  30, 29},    {0x5601, 1, 1,  31, 28},
    {0x5401, 0, 1,  32, 28},    {0x5401, 1, 1,  33, 29},
    {0x5101, 0, 1,  34, 30},    {0x5101, 1, 1,  35, 31},
    {0x4801, 0, 1,  36, 32},    {0x4801, 1, 1,  37, 33},
    {0x3801, 0, 2,  38, 34},    {0x3801, 1, 2,  39, 35},
    {0x3401, 0, 2,  40, 36},    {0x3401, 1, 2,  41, 37},
    {0x3001, 0, 2,  42, 38},    {0x3001, 1, 2,  43, 39},
    {0x2801, 0, 2,  44, 38},    {0x2801, 1, 2,  45, 39},
    {0x2401, 0, 2,  46, 40},    {0x2401, 1, 2,  47, 41},
    {0x2201, 0, 2,  48, 42},    {0x2201, 1, 2,  49, 43},
    {0x1c01, 0, 3,  50, 44},    {0x1c01, 1, 3,  51, 45},
    {0x1801, 0, 3,  52, 46},    {0x1801, 1, 3,  53, 47},
    {0x1601, 0, 3,  54, 48},    {0x1601, 1, 3,  55, 49},
    {0x1401, 0, 3,  56, 50},    {0x1401, 1, 3,  57, 51},
    {0x1201, 0, 3,  58, 52},    {0x1201, 1, 3,  59, 53},
    {0x1101, 0, 3,  60, 54},    {0x1101, 1, 3,  61, 55},
    {0x0ac1, 0, 4,  62, 56},    {0x0ac1, 1, 4,  63, 57},
    {0x09c1, 0, 4,  64, 58},    {0x09c1, 1, 4,  65, 59},
    {0x08a1, 0, 4,  66, 60},    {0x08a1, 1, 4,  67, 61},
    {0x0521, 0, 5,  68, 62},    {0x0521, 1, 5,  69, 63},
    {0x0441, 0, 5,  70, 64},    {0x0441, 1, 5,  71, 65},
    {0x02a1, 0, 6,  72, 66},    {0x02a1, 1, 6,  73, 67},
    {0x0221, 0, 6,  74, 68},    {0x0221, 1, 6,  75, 69},
    {0x0141, 0, 7,  76, 70},    {0x0141, 1, 7,  77, 71},
    {0x0111, 0, 7,  78, 72},    {0x0111, 1, 7,  79, 73},
    {0x0085, 0, 8,  80, 74},    {0x0085, 1, 8,  81, 75},
    {0x0049, 0, 9,  82, 76},    {0x0049, 1, 9,  83, 77},
    {0x0025, 0, 10, 84, 78},    {0x0025, 1, 10, 85, 79},
    {0x0015, 0, 11, 86, 80},    {0x0015, 1, 11, 87, 81},
    {0x0009, 0, 12, 88, 82},    {0x0009, 1, 12, 89, 83},
    {0x0005, 0, 13, 90, 84},    {0x0005, 1, 13, 91, 85},
    {0x0001, 0, 15, 90, 86},    {0x0001, 1, 15, 91, 87},
    {0x5601, 0, 1,  92, 92},    {0x5601, 1, 1,  93, 93}
};

#endif // MQC_TABLE_H
