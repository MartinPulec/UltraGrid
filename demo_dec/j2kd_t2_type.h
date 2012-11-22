///
/// @file    j2kd_t2_type.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   T2 specifis types for JPEG 2000 decoding.
///


#ifndef J2KD_T2_TYPE_H
#define J2KD_T2_TYPE_H

#include "j2kd_type.h"
#include "j2kd_tag_tree.h"
#include "j2kd_seg_info.h"


namespace cuj2kd {


/// Range of codeblocks with tag tree.
/// TODO: rewrite packet decoding process to use single tag tree
struct PrecBand {
    int cblkOffset;     ///< index of first codeblock
    TagTree incl;       ///< tag tree for inclusion information
    TagTree zbplns;     ///< tag tree for zero bitplanes info
    PrecBand * next;    ///< pointer to next codeblock range or null
};


/// Precinct
struct Prec {
    int resIdx;         ///< index of parent resolution (in resolution buffer)
    XY pixBegin;        ///< begin of the precinct area
    XY pixEnd;          ///< end of precinct area
    int cblkOffset;     ///< index of first codeblock of this precinct
    int cblkCount;      ///< number of all codeblocks of this precinct
    PrecBand * bands;   ///< pointer to first codeblock range or null
    Prec * nextPtr;     ///< pointer to next precinct in the chain or null
    
    /// pointer to static function for getting info about segments
    const SegInfo * (*segInfoGetter)(u8);
};


} // end of namespace cuj2kd


#endif // J2KD_T2_TYPE_H

