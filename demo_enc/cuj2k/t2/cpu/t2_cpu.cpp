///
/// @file    t2_cpu.cpp
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Implementation of interface of CPU T2 encoder for JPEG 2000.
///


#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstring>
#include <new>
#include "t2_cpu.h"
#include "t2_cpu_output.h"
#include "t2_cpu_tag_tree.h"
#include "t2_cpu_codewords.h"


/// Represents T2 CPU implementation's internal data.
struct t2_cpu_encoder {
    /// Common tag tree for encoding of all packets
    t2_cpu_tag_tree_t * tag_tree;
}; // end of struct t2_cpu_encoder



/// Initializes T2 CPU encoder for given JPEG 2000 encoder structure.
/// @param enc  initialized structure of JPEG 2000 encoder
/// @return  either pointer to newly created T2 encoder 
///          or 0 if anything goes wrong
t2_cpu_encoder * t2_cpu_create(const struct j2k_encoder * const enc) {
    t2_cpu_encoder * const t2_enc
            = (t2_cpu_encoder*) malloc(sizeof(t2_cpu_encoder));
    if(t2_enc) {
        // find greatest dimensions needed for the tag tree
        int max_cblks_x = 1;
        int max_cblks_y = 1;
        for(int prec_idx = enc->precinct_count; prec_idx--;) {
            const j2k_precinct * const prec = enc->precinct + prec_idx;
            for(int band_idx = 3; band_idx--;) {
                const j2k_size * const size = prec->cblk_counts + band_idx;
                if(max_cblks_x < size->width) {
                    max_cblks_x = size->width;
                }
                if(max_cblks_y < size->height) {
                    max_cblks_y = size->height;
                }
            }
        }
        
        // try to allocate the tag tree
        try {
            t2_enc->tag_tree = new t2_cpu_tag_tree_t(max_cblks_x, max_cblks_y);
        } catch (std::bad_alloc&) {
            t2_enc->tag_tree = 0;
        }
        
        // make sure that the tree is allocated
        if(!t2_enc->tag_tree) {
            t2_cpu_destroy(t2_enc);
            return 0;
        }
    }
    
    return t2_enc;
}



/// Releases resources associated to given T2 CPU encoder.
/// @param t2_enc  pointer to t2 encoder
/// @return 0 if succeded, negative if failed
int t2_cpu_destroy(t2_cpu_encoder * const t2_enc) {
    if(t2_enc) {
        if(t2_enc->tag_tree) delete t2_enc->tag_tree;
        free(t2_enc);
    }
    return 0;
}


/// integer logarithm of 2 (for codeblock and precinct size signaling)
inline int ilog2(int n) {
    int l = 0;
    while(n >>= 1) l++;
    return l;
}


/// Encodes main header of JPEG2000 codestream into given output.
/// @param enc  encoder structure of the image
/// @param out  output bytes writer
/// @param tlm_out  output writer for TLM marker records - to be initialized
/// @return true if OK, false otherwise
bool t2_cpu_main_header_encode(const struct j2k_encoder * const enc,
                               t2_cpu_output_t * const out,
                               t2_cpu_output_t * const tlm_out) {
    // pointer directly to parameters (used frequently)
    const j2k_encoder_params * const params = &enc->params;
    
    // some interesting parameters
    const int comp_count = params->comp_count;
    const int layer_count = 1;  // TODO: layer count
    const int res_count = params->resolution_count;
    const int band_count = res_count * 3 - 2; // for each tile-component
    
    // sizes of various markers
    const int siz_bytes = 38 + 3 * comp_count;
    const int cod_bytes = 12 + res_count;
    const int qcd_bytes = enc->quantization_mode == QM_EXPLICIT
            ? (3 + 2 * band_count)
            : (enc->quantization_mode == QM_IMPLICIT ? 5 : (3 + band_count));
    const int tlm_bytes = 4 + 6 * enc->tilepart_count;
            
    // have enough space for SOC, SIZ, COD and QCD?
    if(!out->has_space(10 + siz_bytes + cod_bytes + qcd_bytes + tlm_bytes)) {
        return false;
    }
    
    // SOC marker
    out->put_2bytes(0xFF4F);
    
    // SIZ marker (p. 26, JPEG2000 Part 1 final draft)
    out->put_2bytes(0xFF51);
    out->put_2bytes(siz_bytes); // SIZ length
    switch(params->capabilities) {
        case J2K_CAP_DEFAULT:
            out->put_2bytes(0);
            break;
        case J2K_CAP_DCI_2K_24:
        case J2K_CAP_DCI_2K_48:
            out->put_2bytes(3);
            break;
        case J2K_CAP_DCI_4K:
            out->put_2bytes(4);
            break;
        default:
            return false; // unknown codestream capabilities
    }
    out->put_4bytes(params->size.width); // image width
    out->put_4bytes(params->size.height); // & height
    out->put_4bytes(0);  //  image origin x
    out->put_4bytes(0);  //  image origin y
    out->put_4bytes(params->size.width); // tile width
    out->put_4bytes(params->size.height); // & height
    out->put_4bytes(0);  //  tile origin x
    out->put_4bytes(0);  //  tile origin y
    out->put_2bytes(comp_count);
    const int bit_depth = params->out_bit_depth == -1
                        ? params->bit_depth : params->out_bit_depth;
    for(int comp_idx = 0; comp_idx < comp_count; comp_idx++) {
        // for each component: (bit depth indexed from 1)
        out->put_byte((params->is_signed ? 0x80 : 0x00) | (bit_depth - 1));
        out->put_byte(1);  // component's x-subsampling
        out->put_byte(1);  // component's y-subsampling
    }
    
    // COD marker (p. 29, JPEG2000 Part 1 final draft)
    out->put_2bytes(0xFF52);
    out->put_2bytes(cod_bytes);
    unsigned char coding_style = 0x01;  // 1 = always using precincts
    if(params->use_sop) { coding_style |= 0x02; }
    if(params->use_eph) { coding_style |= 0x04; }
    out->put_byte(coding_style);
    out->put_byte((int)params->progression_order);
    out->put_2bytes(layer_count);
    out->put_byte(params->mct);
    out->put_byte(res_count - 1); // DWT
    out->put_byte(ilog2(params->cblk_size.width) - 2);
    out->put_byte(ilog2(params->cblk_size.height) - 2);
    out->put_byte(0/*params->cblk_style*/);
    out->put_byte(params->compression != CM_LOSSLESS ? 0 : 1);
    for(int res_idx = 0; res_idx < res_count; res_idx++) {
        // precinct size for each resolution
        const int prec_exp_x = ilog2(params->precinct_size[res_idx].width);
        const int prec_exp_y = ilog2(params->precinct_size[res_idx].height);
        out->put_byte((prec_exp_y << 4) | prec_exp_x);
    }
    
    // QCD marker (p. 37, JPEG2000 Part 1 final draft) 
    // use it to signalize stepsizes for first component
    out->put_2bytes(0xFF5C);
    out->put_2bytes(qcd_bytes);
    if(enc->quantization_mode == QM_EXPLICIT) {
        // quantization stuff signalled for each subband
        out->put_byte((params->guard_bits << 5) | 2);
        for(int band_idx = 0; band_idx < band_count; band_idx++) {
            const int expnt = enc->band[band_idx].stepsize_exponent;
            const int mant = enc->band[band_idx].stepsize_mantisa;
            out->put_2bytes(mant | (expnt << 11));
        }
    } else if (enc->quantization_mode == QM_IMPLICIT) {
        // quantization stuff signaled for LL only
        out->put_byte((params->guard_bits << 5) | 1);
        out->put_2bytes((enc->band[0].stepsize_exponent << 11)
                           | enc->band[0].stepsize_mantisa);
    } else if (enc->quantization_mode == QM_NONE) { // no quantization
        // no quantization at all => signalize dynamic range
        out->put_byte((params->guard_bits << 5) | 0);
        const int g_bits = params->guard_bits;
        for(int band_idx = 0; band_idx < band_count; band_idx++) {
            out->put_byte((enc->band[band_idx].bit_depth + 1 - g_bits) << 3);
        }
    } else {
        return false; // unknown quantization type => FAIL
    }
    
    // TODO: add QCC for each component except of the first one
    
    // TML marker
    out->put_2bytes(0xFF55);    // TLM 
    out->put_2bytes(tlm_bytes); // size of TLM marker
    out->put_byte(0);           // = ID of the one and only TLM marker
    out->put_byte(0x60);        // = using 16bit Ttlm and 32bit Ptlm
    
    // initialize tilepart record writer
    const int tilepart_info_bytes = 6 * enc->tilepart_count;
    tlm_out->init(out->get_end(), out->get_end() + tilepart_info_bytes);
    
    // initialize tilepart info records with zeros
    for(int i = 0; i < tilepart_info_bytes; i++) {
        out->put_byte(0); 
    }
    
    // CME marker :)
    const char * const comment = "Encoded by sonny.";
    const int comment_len = strlen(comment) + 1; // plus trailing 0
    
    // is there enough space for CME marker?
    if(!out->has_space(6 + comment_len)) {
        return false;
    }
    
    // enough space => write it
    out->put_2bytes(0xFF64); // CME
    out->put_2bytes(4 + comment_len); // size of CME marker
    out->put_2bytes(1); // ISO 8859-1 encoded
    out->put_bytes((const unsigned char*)comment, comment_len);
    
    // DCI 4K needs hardcoded progression order change
    if(params->capabilities == J2K_CAP_DCI_4K) {
        // have enough space for the marker segment?
        if(!out->has_space(2 + 2 + 7 * 2)) {
            return false;
        }
        
        out->put_2bytes(0xFF5F); // POC marker
        out->put_2bytes(2 + 7 * 2); // lenght of POC marker segment
        
        // Progression #1 (2K only):
        out->put_byte(0); // starting from resolution #0
        out->put_byte(0); // starting from component #0
        out->put_2bytes(1); // up to layer 1 (the one and only quality layer)
        out->put_byte(params->resolution_count - 1); // except last resolution
        out->put_byte(3); // all components
        out->put_byte(4); // CRPL progression
        
        // Progression #2 (4K additional data):
        out->put_byte(params->resolution_count - 1); // last resolution only
        out->put_byte(0); // starting from component #0
        out->put_2bytes(1); // up to layer 1 (the one and only quality layer)
        out->put_byte(params->resolution_count); // last resolution only
        out->put_byte(3); // all components
        out->put_byte(4); // CRPL progression
    }
    
    // indicate success
    return true;
}



bool t2_cpu_prec_encode(const struct j2k_encoder * const enc,
                        const int prec_idx,
                        t2_cpu_tag_tree_t * const tree,
                        t2_cpu_output_t * const out,
                        unsigned int * tile_prec_idx) {
    // possibly start the packet with start of packet marker
    if (enc->params.use_sop) {
        out->put_2bytes(0xFF91);                       // SOP marker
        out->put_2bytes(4);                            // SOP body size
        out->put_2bytes(0xFFFF & (*tile_prec_idx)++);  // packet index
    }
    
    // pointer to the precinct
    const j2k_precinct * const prec = enc->precinct + prec_idx;
    
    // always signalize nonempty packet
    out->put_one();
    
    // count codeblock data bytes
    int data_bytes = 0;
    
    // first add headers for all codeblocks from (up to) 3 bands
    const j2k_cblk * cblk = enc->cblk + prec->cblk_index;
    for(int band_idx = 0; band_idx < 3; band_idx++) {
        // get number of codeblocks included in this precinct for the band
        const int cblk_count_x = prec->cblk_counts[band_idx].width;
        const int cblk_count_y = prec->cblk_counts[band_idx].height;
        
        // remember pointer to first precinct's codeblock of the band
        const j2k_cblk * const prec_band_first_cblk = cblk;
        
        // is there enough space for headers of the band?
        if(!out->has_space(cblk_count_x * cblk_count_y * 10)) {
            return false;
        }
        
        // initialize tag tree for the band
        for(int cblk_y = 0; cblk_y < cblk_count_y; cblk_y++) {
            for(int cblk_x = 0; cblk_x < cblk_count_x; cblk_x++) {
                // inclusion layer  // TODO: support more layers
                const int ilayer = cblk->pass_count ? 0 : 1; 
                
                // number of zero bitplanes
                const int zbplns = enc->band[cblk->band_index].bit_depth
                                 - cblk->bitplane_count;
                
                // initialize the tag tree
                tree->set(cblk_x, cblk_y, ilayer, zbplns);
                
                // add size of codeblock's data
                if(ilayer == 0) { // TODO: support more layers
                    data_bytes += cblk->byte_count;
                }
                
//                 // verbose
//                 printf("T2: encoding codeblock: %d encoded bitplanes, "
//                        "%d bytes\n", cblk->bitplane_count, cblk->byte_count);
                
                // advance to next codeblock
                cblk++;
            }
        }
        tree->finalize(cblk_count_x, cblk_count_y);
        
        // use the initialized tag tree to encode precinct's codeblocks 
        // in the band
        cblk = prec_band_first_cblk;
        for(int cblk_y = 0; cblk_y < cblk_count_y; cblk_y++) {
            for(int cblk_x = 0; cblk_x < cblk_count_x; cblk_x++) {
                // inclusion layer of the codeblock
                const int ilayer = tree->get_ilayer(cblk_x, cblk_y);
                
                // always encode the inclusion info - TODO: support more layers 
                tree->encode_ilayer(cblk_x, cblk_y, 0, out);
                
                // possibly encode other info TODO: support more layers 
                if(ilayer == 0) {
                    tree->encode_zbplns(cblk_x, cblk_y, out);
                    t2_cpu_codeword_encode(cblk->pass_count, cblk->byte_count, out);
                }
                
                // advance to next codeblock
                cblk++;
            }
        }
        
        // 'cblk' pointer should now point to first codeblock of next 
        // precinct's band (or to first codeblock of next precinct)
    }
    
    // align precinct's header to byte boundary
    out->flush_bits();
    
    // possibly terminate packet header with EPH marker
    if (enc->params.use_eph) {
        out->put_2bytes(0xFF92); // EPH marker
    }
    
    // pointer past the last precinct's codeblock
    const j2k_cblk * const cblk_end = cblk;
    
    // is there enough space for all the data?
    if(!out->has_space(data_bytes)) {
        return false;
    }
    
    // add compressed bytes for all codeblocks from (up to) 3 bands
    cblk = enc->cblk + prec->cblk_index;
    while(cblk != cblk_end) {
        // skip all codeblocks with no included passes
        if(cblk->pass_count) {  // TODO: support more layers
            // add bytes
            out->put_bytes(enc->c_byte_compact + cblk->byte_index_compact,
                           cblk->byte_count);
        }
            
        // advance to next codeblock
        cblk++;
    }
    
    // indicate success
    return true;
}



/// @return true if OK, false otherwise
bool t2_cpu_tilepart_encode(const struct j2k_encoder * const enc,
                            const int tpart_idx,
                            t2_cpu_tag_tree_t * const tree,
                            t2_cpu_output_t * const out,
                            t2_cpu_output_t * const tlm_out,
                            unsigned int  * tile_prec_idx) {
    // pointer to the tile part
    const struct j2k_tilepart * const tpart = enc->tilepart + tpart_idx;
    
    // remember pointer to begin of the tile (to be able to get its size)
    unsigned char * const tile_begin_ptr = out->get_end();
    
    // is there enough space for the tile-part header?
    if(!out->has_space(14)) {
        return false;
    }
    
    // some attributes of SOT marker
    const unsigned short tile_idx = 0; // TODO: add support for more tiles
    
    // write tile index into TLM marker body
    tlm_out->put_2bytes(tile_idx);
    
    // SOT marker
    out->put_2bytes(0xFF90); // SOT
    out->put_2bytes(10); // length of SOT marker (fixed size)
    out->put_2bytes(tile_idx);
    out->put_4bytes(0);  // tile part size - overwritten later
    out->put_byte(tpart_idx);  // tile's part index - TODO: support more tiles
    out->put_byte(enc->tilepart_count);  // tile's parts count
    
    // SOD marker
    out->put_2bytes(0xFF93); // SOD
    
    // encode all packets
    const int prec_idx_begin = tpart->precinct_index;
    const int prec_idx_end = prec_idx_begin + tpart->precinct_count;
    for(int prec_idx = prec_idx_begin; prec_idx < prec_idx_end; prec_idx++) {
        t2_cpu_prec_encode(enc, enc->c_precinct_permutation[prec_idx], tree,
                           out, tile_prec_idx);
    }
    
    // finally, write correct size of the tile:
    const unsigned int tile_bytes = out->get_end() - tile_begin_ptr;
    
    // ... both to SOT marker ...
    t2_cpu_output_t tile_size_output;
    tile_size_output.init(tile_begin_ptr + 6, tile_begin_ptr + 10);
    tile_size_output.put_4bytes(tile_bytes);
    
    // ... and into TLM marker
    tlm_out->put_4bytes(tile_bytes);
    
    // indicate success
    return true;
}



/// Encodes packets in given JPEG2000 encoder structure.
/// @param j2k_enc    pointer to J2K encoder with initialized T2 pointer
/// @param t2_enc     pointer to T2 CPU encoder for given J2K encoder
/// @param out_begin  pointer to output buffer in main memory, where the output 
///                   should be written
/// @param out_size   size of output buffer
/// @return  either size of output stream (in bytes) if encoded OK,
///          or negative error code if failed
int t2_cpu_encode(const struct j2k_encoder * const j2k_enc,
                  t2_cpu_encoder * const t2_enc,
                  unsigned char * const out_begin,
                  const int out_size) {
    // prepare bit/byte writer
    t2_cpu_output_t output, tlm_output;
    output.init(out_begin, out_begin + out_size);
    
    // put main header into the output buffer
    if(!t2_cpu_main_header_encode(j2k_enc, &output, &tlm_output)) {
        return -4;
    }
    
    // currently only one tile supported - this contains index of next precinct
    // in the tile
    unsigned int tile_prec_idx = 0;
    
    // encode all tileparts
    for(int tpart_idx = 0; tpart_idx < j2k_enc->tilepart_count; tpart_idx++) {
        // encode header of current tilepart
        if(!t2_cpu_tilepart_encode(j2k_enc,
                                   tpart_idx,
                                   t2_enc->tag_tree,
                                   &output,
                                   &tlm_output,
                                   &tile_prec_idx)) {
            return -6;
        }
    }
    
    // has enough space for EOC marker?
    if(output.has_space(2)) {
        // terminate the codestream with EOC marker
        output.put_2bytes(0xFFD9);
        
        // return total number of bytes
        return output.get_end() - out_begin;
    } else {
        // not enough space => indicate failure
        return -7;
    }
}


