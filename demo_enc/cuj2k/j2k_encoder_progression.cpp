/* 
 * Copyright (c) 2011, Martin Jirman (martin.jirman@cesnet.cz)
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


#include <algorithm>
#include "j2k_encoder_progression.h"



/// Compares packets in CPRL order.
class precinct_cprl_compare {
private:
    /// encoder, whose progression should be ordered
    const j2k_encoder * const enc;
public:
    /// initializes comparator with encoder pointer
    precinct_cprl_compare(const j2k_encoder * const enc) : enc(enc) {}
    
    /// compares two precincts
    bool operator() (const int & prec_a_idx, const int & prec_b_idx) const {
        // pointers to both precincts
        const j2k_precinct * const prec_a = enc->precinct + prec_a_idx;
        const j2k_precinct * const prec_b = enc->precinct + prec_b_idx;
        
        // primary ordering by component
        const int comp_a = enc->resolution[prec_a->resolution_idx].component_index;
        const int comp_b = enc->resolution[prec_b->resolution_idx].component_index;
        if(comp_a != comp_b) {
            return comp_a < comp_b;
        }
        
        // otherwise by absolute positions (mapped into image space)
        if(prec_a->abs_position.y != prec_b->abs_position.y) {
            return prec_a->abs_position.y < prec_b->abs_position.y;
        }
        if(prec_a->abs_position.x != prec_b->abs_position.x) {
            return prec_a->abs_position.x < prec_b->abs_position.x;
        }
        
        // then by resolution
        const int res_a = enc->resolution[prec_a->resolution_idx].level;
        const int res_b = enc->resolution[prec_b->resolution_idx].level;
        if(res_a != res_b) {
            return res_a < res_b;
        }
        
        // finally by layer - this is not implemented, so for stability, 
        // decide using pointers
        return prec_a < prec_b;
    }
};  // end of precinct_cprl_compare



/// Compares packets in LRCP order.
class precinct_lrcp_compare {
private:
    /// encoder, whose progression should be ordered
    const j2k_encoder * const enc;
public:
    /// initializes comparator with encoder pointer
    precinct_lrcp_compare(const j2k_encoder * const enc) : enc(enc) {}
    
    /// compares two precincts
    bool operator() (const int & prec_a_idx, const int & prec_b_idx) const {
        // pointers to both precincts
        const j2k_precinct * const prec_a = enc->precinct + prec_a_idx;
        const j2k_precinct * const prec_b = enc->precinct + prec_b_idx;
        
        // primary ordering by layer - not used here, so go to resolution
        const int res_a = enc->resolution[prec_a->resolution_idx].level;
        const int res_b = enc->resolution[prec_b->resolution_idx].level;
        if(res_a != res_b) {
            return res_a < res_b;
        }
        
        // otherwise ordered by component
        const int comp_a = enc->resolution[prec_a->resolution_idx].component_index;
        const int comp_b = enc->resolution[prec_b->resolution_idx].component_index;
        if(comp_a != comp_b) {
            return comp_a < comp_b;
        }
        
        // finally by position
        if(prec_a->abs_position.y != prec_b->abs_position.y) {
            return prec_a->abs_position.y < prec_b->abs_position.y;
        }
        if(prec_a->abs_position.x != prec_b->abs_position.x) {
            return prec_a->abs_position.x < prec_b->abs_position.x;
        }
        
        // Huh, not decided? for stability, decide using pointers.
        return prec_a < prec_b;
    }
};  // end of precinct_lrcp_compare



/// Compares packets in PCRL order.
class precinct_pcrl_compare {
private:
    /// encoder, whose progression should be ordered
    const j2k_encoder * const enc;
public:
    /// initializes comparator with encoder pointer
    precinct_pcrl_compare(const j2k_encoder * const enc) : enc(enc) {}
    
    /// compares two precincts
    bool operator() (const int & prec_a_idx, const int & prec_b_idx) const {
        // pointers to both precincts
        const j2k_precinct * const prec_a = enc->precinct + prec_a_idx;
        const j2k_precinct * const prec_b = enc->precinct + prec_b_idx;
        
        // primary ordering by absolute positions (mapped into image space)
        if(prec_a->abs_position.y != prec_b->abs_position.y) {
            return prec_a->abs_position.y < prec_b->abs_position.y;
        }
        if(prec_a->abs_position.x != prec_b->abs_position.x) {
            return prec_a->abs_position.x < prec_b->abs_position.x;
        }
        
        // if not decided, compare by component
        const int comp_a = enc->resolution[prec_a->resolution_idx].component_index;
        const int comp_b = enc->resolution[prec_b->resolution_idx].component_index;
        if(comp_a != comp_b) {
            return comp_a < comp_b;
        }
        
        // then by resolution
        const int res_a = enc->resolution[prec_a->resolution_idx].level;
        const int res_b = enc->resolution[prec_b->resolution_idx].level;
        if(res_a != res_b) {
            return res_a < res_b;
        }
        
        // now, layer should decide, but not supported yet => use pointers
        return prec_a < prec_b;
    }
};  // end of precinct_pcrl_compare



/// Compares packets in RLCP order.
class precinct_rlcp_compare {
private:
    /// encoder, whose progression should be ordered
    const j2k_encoder * const enc;
public:
    /// initializes comparator with encoder pointer
    precinct_rlcp_compare(const j2k_encoder * const enc) : enc(enc) {}
    
    /// compares two precincts
    bool operator() (const int & prec_a_idx, const int & prec_b_idx) const {
        // pointers to both precincts
        const j2k_precinct * const prec_a = enc->precinct + prec_a_idx;
        const j2k_precinct * const prec_b = enc->precinct + prec_b_idx;
        
        // primary ordering by resolution
        const int res_a = enc->resolution[prec_a->resolution_idx].level;
        const int res_b = enc->resolution[prec_b->resolution_idx].level;
        if(res_a != res_b) {
            return res_a < res_b;
        }
        
        // otherwise ordered by layer (not used here, so go to component)
        const int comp_a = enc->resolution[prec_a->resolution_idx].component_index;
        const int comp_b = enc->resolution[prec_b->resolution_idx].component_index;
        if(comp_a != comp_b) {
            return comp_a < comp_b;
        }
        
        // finally by position
        if(prec_a->abs_position.y != prec_b->abs_position.y) {
            return prec_a->abs_position.y < prec_b->abs_position.y;
        }
        if(prec_a->abs_position.x != prec_b->abs_position.x) {
            return prec_a->abs_position.x < prec_b->abs_position.x;
        }
        
        // Not decided => for stability, decide using pointers.
        return prec_a < prec_b;
    }
};  // end of precinct_rlcp_compare



/// Compares packets in RPCL order.
class precinct_rpcl_compare {
private:
    /// encoder, whose progression should be ordered
    const j2k_encoder * const enc;
public:
    /// initializes comparator with encoder pointer
    precinct_rpcl_compare(const j2k_encoder * const enc) : enc(enc) {}
    
    /// compares two precincts
    bool operator() (const int & prec_a_idx, const int & prec_b_idx) const {
        // pointers to both precincts
        const j2k_precinct * const prec_a = enc->precinct + prec_a_idx;
        const j2k_precinct * const prec_b = enc->precinct + prec_b_idx;
        
        // primary ordering by resolution
        const int res_a = enc->resolution[prec_a->resolution_idx].level;
        const int res_b = enc->resolution[prec_b->resolution_idx].level;
        if(res_a != res_b) {
            return res_a < res_b;
        }
        
        // then by position
        if(prec_a->abs_position.y != prec_b->abs_position.y) {
            return prec_a->abs_position.y < prec_b->abs_position.y;
        }
        if(prec_a->abs_position.x != prec_b->abs_position.x) {
            return prec_a->abs_position.x < prec_b->abs_position.x;
        }
        
        // otherwise ordered by component
        const int comp_a = enc->resolution[prec_a->resolution_idx].component_index;
        const int comp_b = enc->resolution[prec_b->resolution_idx].component_index;
        if(comp_a != comp_b) {
            return comp_a < comp_b;
        }
        
        // Now, layers should be used, but not implemented yet.
        // Not decided => for stability, decide using pointers.
        return prec_a < prec_b;
    }
};  // end of precinct_rpcl_compare



/// Functor, which gets true for indices of all packets, which are needed
/// for 2K resolution and false for packets for 4K resolution.
class prec_is_for_2k {
private:
    /// encoder, whose precincts are subdivided
    const j2k_encoder * const enc;
    
    /// level of resolution, whose packets are not needed for 2K
    const int res_level_4k;
public:
    /// initializes comparator with encoder pointer
    prec_is_for_2k(const j2k_encoder * const enc)
            : enc(enc), res_level_4k(enc->params.resolution_count - 1) {}
    
    /// gets true if packet belongs to 2K
    bool operator() (const int & prec_idx) const {
        const int res_idx = enc->precinct[prec_idx].resolution_idx;
        return res_level_4k != enc->resolution[res_idx].level;
    }
}; // end of prec_is_for_2k predicate



/// Sorts range of packet indices according to default progression order.
/// @return true if OK, false if failed
static bool j2k_encoder_progression_sort(struct j2k_encoder * const enc,
                                         int * const range_begin,
                                         int * const range_end) {
    switch(enc->params.progression_order) {
        case PO_CPRL:
            std::sort(range_begin, range_end, precinct_cprl_compare(enc));
            break;
        case PO_LRCP:
            std::sort(range_begin, range_end, precinct_lrcp_compare(enc));
            break;
        case PO_PCRL:
            std::sort(range_begin, range_end, precinct_pcrl_compare(enc));
            break;
        case PO_RLCP:
            std::sort(range_begin, range_end, precinct_rlcp_compare(enc));
            break;
        case PO_RPCL:
            std::sort(range_begin, range_end, precinct_rpcl_compare(enc));
            break;
        default:
            return false;
    }
    return true;
}



/// Prepares packet ordering.
/// @param enc  encoder with initialied structure
/// @return zero if succeded, nonzero if failed
int j2k_encoder_progression_init(struct j2k_encoder * const enc) {
    // TODO: better progression order initialization needed to support layers
    
    // initialize the permutation
    for(int prec_idx = 0; prec_idx < enc->precinct_count; prec_idx++) {
        enc->c_precinct_permutation[prec_idx] = prec_idx;
    }
    
    // begin and end of the packet permutation indices
    int * const range_begin = enc->c_precinct_permutation;
    int * const range_end = range_begin + enc->precinct_count;
    
    // distribute packets into tile-parts
    if(J2K_CAP_DEFAULT == enc->params.capabilities) {
        // 1 tile-part expected
        if(enc->tilepart_count != 1) { return -2; } 
        
        // put all packets into the single tile-part
        enc->tilepart[0].precinct_index = 0;
        enc->tilepart[0].precinct_count = enc->precinct_count;
        
        // sort packets according to progression order
        j2k_encoder_progression_sort(enc, range_begin, range_end);
    } else if(J2K_CAP_DCI_2K_24 == enc->params.capabilities
           || J2K_CAP_DCI_2K_48 == enc->params.capabilities) {
        // 3 tile-parts expected
        if(enc->tilepart_count != 3) { return -3; }
        
        // number of packets per component
        const int comp_pkts = enc->precinct_count / 3;
        
        // packets will be divided into three tile-parts (all tile-components
        // share same coding settings, so all have same number of packets)
        for(int tpart_idx = 0; tpart_idx < 3; tpart_idx++) {
            enc->tilepart[tpart_idx].precinct_index = tpart_idx * comp_pkts;
            enc->tilepart[tpart_idx].precinct_count = comp_pkts;
        }
        
        // check correct progression order
        if(enc->params.progression_order != PO_CPRL) { return -5; }
        
        // sort packets according to progression_order (this will place 
        // component #1 packets at the begin, then component #2 packets 
        // and finally component #3 packets at the end - as expected)
        j2k_encoder_progression_sort(enc, range_begin, range_end);
    } else if(J2K_CAP_DCI_4K == enc->params.capabilities) {
        // 6 tile-parts expected
        if(enc->tilepart_count != 6) { return -4; }

        // check correct progression order
        if(enc->params.progression_order != PO_CPRL) { return -5; }

        // subdivide packets into two partitions: one for 2K packets and 
        // the other for additional 4K packets (highest resolution)
        int * const end2k = std::partition(range_begin, range_end,
                                           prec_is_for_2k(enc));
        
        // numbers of 2K precincts and number of 4K precincts
        const int prec_count_4k = range_end - end2k;
        const int prec_count_2k = end2k - range_begin;
        
        // numbers of precincts for each component in 2K and 4K
        const int comp_pkts_4k = prec_count_4k / 3;
        const int comp_pkts_2k = prec_count_2k / 3;
        
        // set offsets and numbers of precincts for each 2K tile-part
        for(int tpart_idx = 0; tpart_idx < 3; tpart_idx++) {
            enc->tilepart[tpart_idx].precinct_index = tpart_idx * comp_pkts_2k;
            enc->tilepart[tpart_idx].precinct_count = comp_pkts_2k;
        }
        
        // set offsets and numbers of precincts for each 4K tile-part
        for(int tpart_idx = 0; tpart_idx < 3; tpart_idx++) {
            enc->tilepart[3 + tpart_idx].precinct_index
                    = prec_count_2k + tpart_idx * comp_pkts_4k;
            enc->tilepart[3 + tpart_idx].precinct_count = comp_pkts_4k;
        }
        
        // sort packets for 2K and for 4K (both partitions will be ordered 
        // by component)
        j2k_encoder_progression_sort(enc, range_begin, end2k);
        j2k_encoder_progression_sort(enc, end2k, range_end);
    } else { // unknown codestream capabilities
        return -1;
    }
    
    // indicate success
    return 0;
}


