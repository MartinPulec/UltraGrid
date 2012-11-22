///
/// @file    j2kd_seg_info.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Stuff for getting info about codestream segments in T2.
///


#ifndef J2KD_SEG_INFO_H
#define J2KD_SEG_INFO_H

#include "j2kd_type.h"


namespace cuj2kd {


/// Info about one codestream segment.
struct SegInfo {
    int maxPassCount; ///< maximal number of passes in the segment
    bool bypassAC;    ///< true if arithm. coding is not used for the segment
};


/// Gets segment info for segment sstarting with given pass,
/// when each pass termination is NOT used, nor is used selective AC bypass.
/// @param passIdx  index of first pass of the segment
/// @return pointer to immutable structure with info about the segment
const SegInfo * getSegInfoNormal(u8 passIdx);


/// Gets segment info for segment starting with given pass,
/// when selective AC bypass mode is used.
/// @param passIdx  index of first pass of the segment
/// @return pointer to immutable structure with info about the segment
const SegInfo * getSegInfoSelectiveBypass(u8 passIdx);


/// Gets segment info for segment sstarting with given pass,
/// when each pass termination is used.
/// @param passIdx  index of first pass of the segment
/// @return pointer to immutable structure with info about the segment
const SegInfo * getSegInfoTermAll(u8 passIdx);


/// Gets segment info for segment starting with given pass,
/// when each pass termination is used together with selective AC bypass mode.
/// @param passIdx  index of first pass of the segment
/// @return pointer to immutable structure with info about the segment
const SegInfo * getSegInfoTermAllSelectiveBypass(u8 passIdx);



} // end of namespace cuj2kd


#endif // J2KD_SEG_INFO_H

