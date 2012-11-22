///
/// @file    j2kd_seg_info.cpp
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Implementation of getting info about codestream segments in T2.
///


#include "j2kd_seg_info.h"


namespace cuj2kd {


// /// Info about one codestream segment.
// struct SegInfo {
//     int maxPassCount; ///< maximal number of passes in the segment
//     bool bypassAC;    ///< true if arithm. coding is not used for the segment
// };


/// Gets segment info for segment sstarting with given pass,
/// when each pass termination is NOT used, nor is used selective AC bypass.
/// @param passIdx  index of first pass of the segment
/// @return pointer to immutable structure with info about the segment
const SegInfo * getSegInfoNormal(u8) {
    static const SegInfo info = {255, false};
    return &info;
}



/// Gets segment info for segment sstarting with given pass,
/// when each pass termination is used.
/// @param passIdx  index of first pass of the segment
/// @return pointer to immutable structure with info about the segment
const SegInfo * getSegInfoTermAll(u8) {
    static const SegInfo info = {1, false};
    return &info;
}



/// Gets segment info for segment starting with given pass,
/// when selective AC bypass mode is used.
/// @param passIdx  index of first pass of the segment
/// @return pointer to immutable structure with info about the segment
const SegInfo * getSegInfoSelectiveBypass(u8 passIdx) {
    static const SegInfo info[256] = {
        {10, false}, // pass 0
        {9,  false}, // pass 1
        {8,  false}, // pass 2
        {7,  false}, // pass 3
        {6,  false}, // pass 4
        {5,  false}, // pass 5
        {4,  false}, // pass 6
        {3,  false}, // pass 7
        {2,  false}, // pass 8
        {1,  false}, // pass 9
        {2,  true},  // pass 10
        {1,  true},  // pass 11
        {1,  false}, // pass 12
        {2,  true},  // pass 13
        {1,  true},  // pass 14
        {1,  false}, // pass 15
        {2,  true},  // pass 16
        {1,  true},  // pass 17
        {1,  false}, // pass 18
        {2,  true},  // pass 19
        {1,  true},  // pass 20
        {1,  false}, // pass 21
        {2,  true},  // pass 22
        {1,  true},  // pass 23
        {1,  false}, // pass 24
        {2,  true},  // pass 25
        {1,  true},  // pass 26
        {1,  false}, // pass 27
        {2,  true},  // pass 28
        {1,  true},  // pass 29
        {1,  false}, // pass 30
        {2,  true},  // pass 31
        {1,  true},  // pass 32
        {1,  false}, // pass 33
        {2,  true},  // pass 34
        {1,  true},  // pass 35
        {1,  false}, // pass 36
        {2,  true},  // pass 37
        {1,  true},  // pass 38
        {1,  false}, // pass 39
        {2,  true},  // pass 40
        {1,  true},  // pass 41
        {1,  false}, // pass 42
        {2,  true},  // pass 43
        {1,  true},  // pass 44
        {1,  false}, // pass 45
        {2,  true},  // pass 46
        {1,  true},  // pass 47
        {1,  false}, // pass 48
        {2,  true},  // pass 49
        {1,  true},  // pass 50
        {1,  false}, // pass 51
        {2,  true},  // pass 52
        {1,  true},  // pass 53
        {1,  false}, // pass 54
        {2,  true},  // pass 55
        {1,  true},  // pass 56
        {1,  false}, // pass 57
        {2,  true},  // pass 58
        {1,  true},  // pass 59
        {1,  false}, // pass 60
        {2,  true},  // pass 61
        {1,  true},  // pass 62
        {1,  false}, // pass 63
        {2,  true},  // pass 64
        {1,  true},  // pass 65
        {1,  false}, // pass 66
        {2,  true},  // pass 67
        {1,  true},  // pass 68
        {1,  false}, // pass 69
        {2,  true},  // pass 70
        {1,  true},  // pass 71
        {1,  false}, // pass 72
        {2,  true},  // pass 73
        {1,  true},  // pass 74
        {1,  false}, // pass 75
        {2,  true},  // pass 76
        {1,  true},  // pass 77
        {1,  false}, // pass 78
        {2,  true},  // pass 79
        {1,  true},  // pass 80
        {1,  false}, // pass 81
        {2,  true},  // pass 82
        {1,  true},  // pass 83
        {1,  false}, // pass 84
        {2,  true},  // pass 85
        {1,  true},  // pass 86
        {1,  false}, // pass 87
        {2,  true},  // pass 88
        {1,  true},  // pass 89
        {1,  false}, // pass 90
        {2,  true},  // pass 91
        {1,  true},  // pass 92
        {1,  false}, // pass 93
        {2,  true},  // pass 94
        {1,  true},  // pass 95
        {1,  false}, // pass 96
        {2,  true},  // pass 97
        {1,  true},  // pass 98
        {1,  false}, // pass 99
        {2,  true},  // pass 100
        {1,  true},  // pass 101
        {1,  false}, // pass 102
        {2,  true},  // pass 103
        {1,  true},  // pass 104
        {1,  false}, // pass 105
        {2,  true},  // pass 106
        {1,  true},  // pass 107
        {1,  false}, // pass 108
        {2,  true},  // pass 109
        {1,  true},  // pass 110
        {1,  false}, // pass 111
        {2,  true},  // pass 112
        {1,  true},  // pass 113
        {1,  false}, // pass 114
        {2,  true},  // pass 115
        {1,  true},  // pass 116
        {1,  false}, // pass 117
        {2,  true},  // pass 118
        {1,  true},  // pass 119
        {1,  false}, // pass 120
        {2,  true},  // pass 121
        {1,  true},  // pass 122
        {1,  false}, // pass 123
        {2,  true},  // pass 124
        {1,  true},  // pass 125
        {1,  false}, // pass 126
        {2,  true},  // pass 127
        {1,  true},  // pass 128
        {1,  false}, // pass 129
        {2,  true},  // pass 130
        {1,  true},  // pass 131
        {1,  false}, // pass 132
        {2,  true},  // pass 133
        {1,  true},  // pass 134
        {1,  false}, // pass 135
        {2,  true},  // pass 136
        {1,  true},  // pass 137
        {1,  false}, // pass 138
        {2,  true},  // pass 139
        {1,  true},  // pass 140
        {1,  false}, // pass 141
        {2,  true},  // pass 142
        {1,  true},  // pass 143
        {1,  false}, // pass 144
        {2,  true},  // pass 145
        {1,  true},  // pass 146
        {1,  false}, // pass 147
        {2,  true},  // pass 148
        {1,  true},  // pass 149
        {1,  false}, // pass 150
        {2,  true},  // pass 151
        {1,  true},  // pass 152
        {1,  false}, // pass 153
        {2,  true},  // pass 154
        {1,  true},  // pass 155
        {1,  false}, // pass 156
        {2,  true},  // pass 157
        {1,  true},  // pass 158
        {1,  false}, // pass 159
        {2,  true},  // pass 160
        {1,  true},  // pass 161
        {1,  false}, // pass 162
        {2,  true},  // pass 163
        {1,  true},  // pass 164
        {1,  false}, // pass 165
        {2,  true},  // pass 166
        {1,  true},  // pass 167
        {1,  false}, // pass 168
        {2,  true},  // pass 169
        {1,  true},  // pass 170
        {1,  false}, // pass 171
        {2,  true},  // pass 172
        {1,  true},  // pass 173
        {1,  false}, // pass 174
        {2,  true},  // pass 175
        {1,  true},  // pass 176
        {1,  false}, // pass 177
        {2,  true},  // pass 178
        {1,  true},  // pass 179
        {1,  false}, // pass 180
        {2,  true},  // pass 181
        {1,  true},  // pass 182
        {1,  false}, // pass 183
        {2,  true},  // pass 184
        {1,  true},  // pass 185
        {1,  false}, // pass 186
        {2,  true},  // pass 187
        {1,  true},  // pass 188
        {1,  false}, // pass 189
        {2,  true},  // pass 190
        {1,  true},  // pass 191
        {1,  false}, // pass 192
        {2,  true},  // pass 193
        {1,  true},  // pass 194
        {1,  false}, // pass 195
        {2,  true},  // pass 196
        {1,  true},  // pass 197
        {1,  false}, // pass 198
        {2,  true},  // pass 199
        {1,  true},  // pass 200
        {1,  false}, // pass 201
        {2,  true},  // pass 202
        {1,  true},  // pass 203
        {1,  false}, // pass 204
        {2,  true},  // pass 205
        {1,  true},  // pass 206
        {1,  false}, // pass 207
        {2,  true},  // pass 208
        {1,  true},  // pass 209
        {1,  false}, // pass 210
        {2,  true},  // pass 211
        {1,  true},  // pass 212
        {1,  false}, // pass 213
        {2,  true},  // pass 214
        {1,  true},  // pass 215
        {1,  false}, // pass 216
        {2,  true},  // pass 217
        {1,  true},  // pass 218
        {1,  false}, // pass 219
        {2,  true},  // pass 220
        {1,  true},  // pass 221
        {1,  false}, // pass 222
        {2,  true},  // pass 223
        {1,  true},  // pass 224
        {1,  false}, // pass 225
        {2,  true},  // pass 226
        {1,  true},  // pass 227
        {1,  false}, // pass 228
        {2,  true},  // pass 229
        {1,  true},  // pass 230
        {1,  false}, // pass 231
        {2,  true},  // pass 232
        {1,  true},  // pass 233
        {1,  false}, // pass 234
        {2,  true},  // pass 235
        {1,  true},  // pass 236
        {1,  false}, // pass 237
        {2,  true},  // pass 238
        {1,  true},  // pass 239
        {1,  false}, // pass 240
        {2,  true},  // pass 241
        {1,  true},  // pass 242
        {1,  false}, // pass 243
        {2,  true},  // pass 244
        {1,  true},  // pass 245
        {1,  false}, // pass 246
        {2,  true},  // pass 247
        {1,  true},  // pass 248
        {1,  false}, // pass 249
        {2,  true},  // pass 250
        {1,  true},  // pass 251
        {1,  false}, // pass 252
        {2,  true},  // pass 253
        {1,  true},  // pass 254
        {1,  false}, // pass 255
    };
    return info + passIdx;
}



/// Gets segment info for segment starting with given pass,
/// when each pass termination is used together with selective AC bypass mode.
/// @param passIdx  index of first pass of the segment
/// @return pointer to immutable structure with info about the segment
const SegInfo * getSegInfoTermAllSelectiveBypass(u8 passIdx) {
    static const SegInfo info[256] = {
        // AC bypass + each pass termination
        {1, false}, // pass 0
        {1, false}, // pass 1
        {1, false}, // pass 2
        {1, false}, // pass 3
        {1, false}, // pass 4
        {1, false}, // pass 5
        {1, false}, // pass 6
        {1, false}, // pass 7
        {1, false}, // pass 8
        {1, false}, // pass 9
        {1, true},  // pass 10
        {1, true},  // pass 11
        {1, false}, // pass 12
        {1, true},  // pass 13
        {1, true},  // pass 14
        {1, false}, // pass 15
        {1, true},  // pass 16
        {1, true},  // pass 17
        {1, false}, // pass 18
        {1, true},  // pass 19
        {1, true},  // pass 20
        {1, false}, // pass 21
        {1, true},  // pass 22
        {1, true},  // pass 23
        {1, false}, // pass 24
        {1, true},  // pass 25
        {1, true},  // pass 26
        {1, false}, // pass 27
        {1, true},  // pass 28
        {1, true},  // pass 29
        {1, false}, // pass 30
        {1, true},  // pass 31
        {1, true},  // pass 32
        {1, false}, // pass 33
        {1, true},  // pass 34
        {1, true},  // pass 35
        {1, false}, // pass 36
        {1, true},  // pass 37
        {1, true},  // pass 38
        {1, false}, // pass 39
        {1, true},  // pass 40
        {1, true},  // pass 41
        {1, false}, // pass 42
        {1, true},  // pass 43
        {1, true},  // pass 44
        {1, false}, // pass 45
        {1, true},  // pass 46
        {1, true},  // pass 47
        {1, false}, // pass 48
        {1, true},  // pass 49
        {1, true},  // pass 50
        {1, false}, // pass 51
        {1, true},  // pass 52
        {1, true},  // pass 53
        {1, false}, // pass 54
        {1, true},  // pass 55
        {1, true},  // pass 56
        {1, false}, // pass 57
        {1, true},  // pass 58
        {1, true},  // pass 59
        {1, false}, // pass 60
        {1, true},  // pass 61
        {1, true},  // pass 62
        {1, false}, // pass 63
        {1, true},  // pass 64
        {1, true},  // pass 65
        {1, false}, // pass 66
        {1, true},  // pass 67
        {1, true},  // pass 68
        {1, false}, // pass 69
        {1, true},  // pass 70
        {1, true},  // pass 71
        {1, false}, // pass 72
        {1, true},  // pass 73
        {1, true},  // pass 74
        {1, false}, // pass 75
        {1, true},  // pass 76
        {1, true},  // pass 77
        {1, false}, // pass 78
        {1, true},  // pass 79
        {1, true},  // pass 80
        {1, false}, // pass 81
        {1, true},  // pass 82
        {1, true},  // pass 83
        {1, false}, // pass 84
        {1, true},  // pass 85
        {1, true},  // pass 86
        {1, false}, // pass 87
        {1, true},  // pass 88
        {1, true},  // pass 89
        {1, false}, // pass 90
        {1, true},  // pass 91
        {1, true},  // pass 92
        {1, false}, // pass 93
        {1, true},  // pass 94
        {1, true},  // pass 95
        {1, false}, // pass 96
        {1, true},  // pass 97
        {1, true},  // pass 98
        {1, false}, // pass 99
        {1, true},  // pass 100
        {1, true},  // pass 101
        {1, false}, // pass 102
        {1, true},  // pass 103
        {1, true},  // pass 104
        {1, false}, // pass 105
        {1, true},  // pass 106
        {1, true},  // pass 107
        {1, false}, // pass 108
        {1, true},  // pass 109
        {1, true},  // pass 110
        {1, false}, // pass 111
        {1, true},  // pass 112
        {1, true},  // pass 113
        {1, false}, // pass 114
        {1, true},  // pass 115
        {1, true},  // pass 116
        {1, false}, // pass 117
        {1, true},  // pass 118
        {1, true},  // pass 119
        {1, false}, // pass 120
        {1, true},  // pass 121
        {1, true},  // pass 122
        {1, false}, // pass 123
        {1, true},  // pass 124
        {1, true},  // pass 125
        {1, false}, // pass 126
        {1, true},  // pass 127
        {1, true},  // pass 128
        {1, false}, // pass 129
        {1, true},  // pass 130
        {1, true},  // pass 131
        {1, false}, // pass 132
        {1, true},  // pass 133
        {1, true},  // pass 134
        {1, false}, // pass 135
        {1, true},  // pass 136
        {1, true},  // pass 137
        {1, false}, // pass 138
        {1, true},  // pass 139
        {1, true},  // pass 140
        {1, false}, // pass 141
        {1, true},  // pass 142
        {1, true},  // pass 143
        {1, false}, // pass 144
        {1, true},  // pass 145
        {1, true},  // pass 146
        {1, false}, // pass 147
        {1, true},  // pass 148
        {1, true},  // pass 149
        {1, false}, // pass 150
        {1, true},  // pass 151
        {1, true},  // pass 152
        {1, false}, // pass 153
        {1, true},  // pass 154
        {1, true},  // pass 155
        {1, false}, // pass 156
        {1, true},  // pass 157
        {1, true},  // pass 158
        {1, false}, // pass 159
        {1, true},  // pass 160
        {1, true},  // pass 161
        {1, false}, // pass 162
        {1, true},  // pass 163
        {1, true},  // pass 164
        {1, false}, // pass 165
        {1, true},  // pass 166
        {1, true},  // pass 167
        {1, false}, // pass 168
        {1, true},  // pass 169
        {1, true},  // pass 170
        {1, false}, // pass 171
        {1, true},  // pass 172
        {1, true},  // pass 173
        {1, false}, // pass 174
        {1, true},  // pass 175
        {1, true},  // pass 176
        {1, false}, // pass 177
        {1, true},  // pass 178
        {1, true},  // pass 179
        {1, false}, // pass 180
        {1, true},  // pass 181
        {1, true},  // pass 182
        {1, false}, // pass 183
        {1, true},  // pass 184
        {1, true},  // pass 185
        {1, false}, // pass 186
        {1, true},  // pass 187
        {1, true},  // pass 188
        {1, false}, // pass 189
        {1, true},  // pass 190
        {1, true},  // pass 191
        {1, false}, // pass 192
        {1, true},  // pass 193
        {1, true},  // pass 194
        {1, false}, // pass 195
        {1, true},  // pass 196
        {1, true},  // pass 197
        {1, false}, // pass 198
        {1, true},  // pass 199
        {1, true},  // pass 200
        {1, false}, // pass 201
        {1, true},  // pass 202
        {1, true},  // pass 203
        {1, false}, // pass 204
        {1, true},  // pass 205
        {1, true},  // pass 206
        {1, false}, // pass 207
        {1, true},  // pass 208
        {1, true},  // pass 209
        {1, false}, // pass 210
        {1, true},  // pass 211
        {1, true},  // pass 212
        {1, false}, // pass 213
        {1, true},  // pass 214
        {1, true},  // pass 215
        {1, false}, // pass 216
        {1, true},  // pass 217
        {1, true},  // pass 218
        {1, false}, // pass 219
        {1, true},  // pass 220
        {1, true},  // pass 221
        {1, false}, // pass 222
        {1, true},  // pass 223
        {1, true},  // pass 224
        {1, false}, // pass 225
        {1, true},  // pass 226
        {1, true},  // pass 227
        {1, false}, // pass 228
        {1, true},  // pass 229
        {1, true},  // pass 230
        {1, false}, // pass 231
        {1, true},  // pass 232
        {1, true},  // pass 233
        {1, false}, // pass 234
        {1, true},  // pass 235
        {1, true},  // pass 236
        {1, false}, // pass 237
        {1, true},  // pass 238
        {1, true},  // pass 239
        {1, false}, // pass 240
        {1, true},  // pass 241
        {1, true},  // pass 242
        {1, false}, // pass 243
        {1, true},  // pass 244
        {1, true},  // pass 245
        {1, false}, // pass 246
        {1, true},  // pass 247
        {1, true},  // pass 248
        {1, false}, // pass 249
        {1, true},  // pass 250
        {1, true},  // pass 251
        {1, false}, // pass 252
        {1, true},  // pass 253
        {1, true},  // pass 254
        {1, false}, // pass 255
    };
    return info + passIdx;
}


} // end of namespace cuj2kd


