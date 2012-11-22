///
/// @file    j2kd_util.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Helper functions not directly related to encoding.
///


// prevent from multiple includes into the same file
#ifndef J2KD_UTIL_H
#define J2KD_UTIL_H


#include "j2kd_type.h"


namespace cuj2kd {


/// Integer division with rounding up.
template <typename T>
T divRndUp(const T & n, const T & d) { return (n + d - 1) / d; }


/// Round n up to multiple of d.
template <typename T>
inline T rndUp(const T & n, const T & d) { return divRndUp(n, d) * d; }


/// Minimum of two values.
template <typename T>
inline T min(const T & a, const T & b) { return (a < b) ? a : b; }


/// Component-wise minimum of two 2D points/vectors/sizes.
inline XY min(const XY & a, const XY & b) {
    return XY(min(a.x, b.x), min(a.y, b.y));
}


/// Maximum of two values.
template <typename T>
inline T max(const T & a, const T & b) { return (a > b) ? a : b; }



} // end of namespace cuj2kd

#endif // J2KD_UTIL_H
