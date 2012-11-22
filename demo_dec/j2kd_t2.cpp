///
/// @file    j2kd_t2.cpp
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Implementation of T2 decoder for JPEG 2000 codestream.
///


#include <cmath>
#include <algorithm>

#include "j2kd_t2.h"
#include "j2kd_t2_reader.h"
#include "j2kd_stepsize_reader.h"
#include "j2kd_seg_info.h"
#include "j2kd_timer.h"
#include "j2kd_ebcot.h"

namespace cuj2kd {

    

/// Reads variable-sized component index using given reader.
inline int readCompIdx(T2Reader & data, const Image * img) {
    return img->comps.count() < 257 ? (int)data.readU8() : (int)data.readU16();
}



/// Gets lowpass band coordinates scaled down to some resolution.
/// @param d              position of the begin or end of the band
/// @param dwtLevelCount  number of DWT levels to get the band
/// @return the position scaled down to the resolution
inline int loBandCoord(const unsigned int d, const int dwtLevelCount) {
    // (B-15), page 48, ITU-T Rec. T.800 (08/2002 E)
    return (d + (1 << dwtLevelCount) - 1) >> dwtLevelCount;
}



/// Gets highpass band coordinates scaled down to some resolution.
/// @param d              position of the begin or end of the band
/// @param dwtLevelCount  number of DWT levels to get the band
/// @return the position scaled down to the resolution
inline int hiBandCoord(const unsigned int d, const int dwtLevelCount) {
    // (B-15), page 48, ITU-T Rec. T.800 (08/2002 E)
    const unsigned int s = 1 << dwtLevelCount;
    return (d + s - 1 - (s >> 1)) >> dwtLevelCount;
}



inline XY loBandCoord(const XY & p, const int dwtLevelCount) {
    // TODO: optimize
    return XY(loBandCoord(p.x, dwtLevelCount), loBandCoord(p.y, dwtLevelCount));
}



inline XY hiBandCoord(const XY & p, const int dwtLevelCount) {
    // TODO: optimize
    return XY(hiBandCoord(p.x, dwtLevelCount), hiBandCoord(p.y, dwtLevelCount));
}



/// Gets the precinct size for given resolution.
/// (Includes halving for other resolutions than resolution #0.)
/// @param coding      pointer to structure with related coding info
/// @param resolution  number of the resolution (e.g. 0 for LL band resolution)
/// @return size of the precinct for the resolution (including halving)
inline XY getResPrecSize(const TCompCoding * coding, const int resolution) {
    // if not explicitly listed, use the default one
    if(resolution >= coding->precSizesCount) {
        return resolution ? XY(1 << 14, 1 << 14) : XY(1 << 15, 1 << 15);
    }
    
    // get the size from the codestream otherwise
    const u8 packedSize = coding->precSizesPtr[resolution];
    return resolution
            ? XY(1 << ((packedSize & 0xF) - 1), 1 << ((packedSize >> 4) - 1))
            : XY(1 << (packedSize & 0xF), 1 << (packedSize >> 4));
}



/// Gets number of partitions in given range.
static int precPartition(const int begin, const int end, const int size) {
    // empty intersection is a special case
    if(begin >= end) {
        return 0;
    }
    
    // versions of begin and end divided by unit size with remainders
    const int dBegin = (begin + size - 1) / size;
    const int rBegin = begin % size;
    const int dEnd = end / size;
    const int rEnd = end % size;
    
    // return number of full intervals + up to 2 boundary intervals
    return (dEnd - dBegin) + (rEnd ? 1 : 0) + (rBegin ? 1 : 0);
}



/// Gets number of codeblocks of the precinct.
static XY precPartition(const XY & precBegin, const XY & precEnd, 
                        const XY & bandBegin, const XY & bandEnd,
                        const XY & cblkSize) {
    // intersection of precinct and band range
    const XY begin = XY(max(precBegin.x, bandBegin.x),
                        max(precBegin.y, bandBegin.y));
    const XY end = XY(min(precEnd.x, bandEnd.x), min(precEnd.y, bandEnd.y));

    // get number of partitions along both axes
    return XY(precPartition(begin.x, end.x, cblkSize.x),
              precPartition(begin.y, end.y, cblkSize.y));
}



/// Reads the variable symbol representing number of passes.
static int readNumPasses(T2Reader & reader) {
    if(0 == reader.readBit()) {
        return 1;
    }
    if(0 == reader.readBit()) {
        return 2;
    }
    const u32 bits2 = reader.readBits(2);
    if(0x3 != bits2) {  // if either bit is zero
        return 3 + bits2;
    }
    const u16 bits5 = reader.readBits(5);
    if(0x1F != bits5) { // if any of those 5 bits is zero
        return 6 + bits5;
    }
    return 37 + reader.readBits(7);
}



/// integer logarithm (for codeblock and precinct size signaling)
inline int ilog2(int n) {
    int l = 0;
    while(n >>= 1) l++;
    return l;
}



/// Reads progression orders and adds them to given packet iterator.
/// @param iter  pointer to iterator to be extended with progression orders
/// @param reader  reader of POC marker segment
void Tier2::readPoc(PacketIterator * const iter, T2Reader & reader) {
    // remaining byte count with packed progression info
    const int byteCount = reader.bytesRemaining();
    
    // 9 or 7 bytes per one progression? (Depends on number of components.)
    if(img->comps.count() < 257) {
        if(byteCount < 7) {
            log->warning("Empty POC (only %d bytes of payload).", byteCount);
        }
        for(int remainingProgCount = byteCount / 7; remainingProgCount--;) {
            // read progression volume info
            const int resBeginIdx = reader.readU8();
            const int compBeginIdx = reader.readU8();
            const int layerEndIdx = reader.readU16();
            const int resEndIdx = reader.readU8();
            const int compEndIdx = reader.readU8();
            const int progType = reader.readU8();
            
            // add this progression volume to packet iterator
            iter->addVolume(resBeginIdx, resEndIdx, compBeginIdx, compEndIdx,
                            layerEndIdx, (ProgOrder)progType);
        }
    } else {
        if(byteCount < 9) {
            log->warning("Empty POC (only %d bytes of payload).", byteCount);
        }
        for(int remainingProgCount = byteCount / 9; remainingProgCount--;) {
            // read progression volume info
            const int resBeginIdx = reader.readU8();
            const int compBeginIdx = reader.readU16();
            const int layerEndIdx = reader.readU16();
            const int resEndIdx = reader.readU8();
            const int compEndIdx = reader.readU16();
            const int progType = reader.readU8();
            
            // add this progression volume to packet iterator
            iter->addVolume(resBeginIdx, resEndIdx, compBeginIdx, compEndIdx,
                            layerEndIdx, (ProgOrder)progType);
        }
    }
}



/// Analyzes SOC and SIZ marker.
/// @param cStream  codestream reader
/// @return number of bytes to skip to get to next marker.
size_t Tier2::readSiz(T2Reader cStream) {
    // is there at least complete SOC and partial SIZ marker?
    if(!cStream.hasBytes(42)) {
        throwBadCStream("Codestream too short (%d B) for SIZ marker.",
                        (int)cStream.bytesRemaining());
    }
    
    // make sure that SOC marker is present (if not, this is not valid 
    // JPEG 2000 codestream)
    const u16 socMarker = cStream.readU16();
    if(0xFF4F != socMarker) {
        throwBadCStream("Found %x instead of SOC.", (int)socMarker);
    }
    
    // SIZ marker must follow in codestream
    const u16 sizMarker = cStream.readU16();
    if(0xFF51 != sizMarker) {
        throwBadCStream("Found %x instead of SIZ.", (int)sizMarker);
    }
    
    // fields of SIZ marker
    const u16 sizLen = cStream.readU16();   // length of the SIZ marker
    img->capabilities = cStream.readU16();  // stream type/capabilities
    img->imgEnd.x = cStream.readU32();      // image end x
    img->imgEnd.y = cStream.readU32();      // image end y
    img->imgBegin.x = cStream.readU32();    // image begin x
    img->imgBegin.y = cStream.readU32();    // image begin y
    img->tSize.x = cStream.readU32();    // tile width
    img->tSize.y = cStream.readU32();    // tile height
    img->tOrigin.x = cStream.readU32();  // tile origin x
    img->tOrigin.y = cStream.readU32();  // tile origin y
    
    // check sizes and offsets
    if(img->tOrigin.x > img->imgBegin.x) {
        throwBadCStream("XTOsiz = %d > XOsiz = %d.",
                        img->tOrigin.x, img->imgBegin.x);
    }
    if(img->tOrigin.y > img->imgBegin.y) {
        throwBadCStream("YTOsiz = %d > YOsiz = %d.",
                        img->tOrigin.y, img->imgBegin.y);
    }
    if(img->tSize.x + img->tOrigin.x <= img->imgBegin.x) {
        throwBadCStream("XTsiz + XTOsiz = %d > XOsiz = %d.",
                        img->tSize.x + img->tOrigin.x, img->imgBegin.x);
    }
    if(img->tSize.y + img->tOrigin.y <= img->imgBegin.y) {
        throwBadCStream("YTsiz + YTOsiz = %d > YOsiz = %d.",
                        img->tSize.y + img->tOrigin.y, img->imgBegin.y);
    }
    if(img->imgEnd.x <= img->imgBegin.x) {
        throwBadCStream("Xsiz %d <= XOsiz %d.", img->imgEnd.x, img->imgBegin.x);
    }
    if(img->imgEnd.y <= img->imgBegin.y) {
        throwBadCStream("Ysiz %d <= YOsiz %d.", img->imgEnd.y, img->imgBegin.y);
    }
    
    // tile count along both axes
    img->tCount = (img->imgEnd - img->tOrigin + img->tSize - 1) / img->tSize;
    
    // get count of components, check it and extract component info
    const int compCount = cStream.readU16();
    if(sizLen < 38 + 3 * compCount) {
        throwBadCStream("SIZ too short: %d bytes.", (int)sizLen);
    }
    for(int c = 0; c < compCount; c++) {
        // get component attributes
        const u8 info = cStream.readU8();
        const int resX = cStream.readU8();
        const int resY = cStream.readU8();
        
        // check the attributes
        if(resX == 0) { throwBadCStream("Comp #%d XRsiz = 0", c); }
        if(resY == 0) { throwBadCStream("Comp #%d YRsiz = 0", c); }
        if(resX > 1) { throwUnsupported("XRsiz %d (comp #%d).", resX, c); }
        if(resY > 1) { throwUnsupported("YRsiz %d (comp #%d).", resY, c); }
        
        // put attributes into output structure
        Comp * const comp = img->comps.reservePtr();
        comp->bitDepth = (0x7F & info) + 1;
        if(comp->bitDepth > 38) {
            throwBadCStream("%d bpp (comp #%d).", comp->bitDepth, c);
        }
        comp->isSigned = (bool)(0x80 & info);
        
        // set default coding style and quantization
        comp->defQuantIdx = 0;
        comp->defCStyleIdx = 0;
    }
    
    // initialize tile info, adding tile info structures in raster order
    for(int ty = img->tOrigin.y; ty < img->imgEnd.y; ty += img->tSize.y) {
        for(int tx = img->tOrigin.x; tx < img->imgEnd.x; tx += img->tSize.x) {
            // reserve output structure for the tile and initialize it
            const size_t tileIdx = img->tiles.reserveIdx();
            Tile * const tile = &img->tiles[tileIdx];
            tile->pixBegin.x = max(tx, img->imgBegin.x);
            tile->pixBegin.y = max(ty, img->imgBegin.y);
            tile->pixEnd.x = min(tx + img->tSize.x, img->imgEnd.x);
            tile->pixEnd.y = min(ty + img->tSize.y, img->imgEnd.y);
            tile->tileIdx = tileIdx;
            tile->tCompIdx = -1; // means "NOT INITIALIZED", will be set later
            tile->nextTPartIdx = 0;
            tile->tileCodingIdx = 0; // default
        }
    }
    
    // return size of two analyzed markers
    return sizLen + 4;
}



/// Reads single-component related parts of COD anc COC marker segments.
/// @param stream  reader to read info from
/// @param tCompCodingIdx  index of output structure
/// @param explicitPrecSizes  true if precinct sizes should be extracted
void Tier2::readTCompCoding(T2Reader & stream, const int tCompCodingIdx,
                            const bool explicitPrecSizes) {
    // pointer to output structure
    TCompCoding * const coding = &img->tCompCoding[tCompCodingIdx];
    
    // number of DWT levels
    coding->dwtLevelCount = stream.readU8();
    if(coding->dwtLevelCount > 32) {
        throwBadCStream("Bad DWT levels: %d.", coding->dwtLevelCount);
    }
    
    // codeblock size
    const u8 cblkExpX = stream.readU8() & 0xF;
    const u8 cblkExpY = stream.readU8() & 0xF;
    if(cblkExpX > 8) {
        throwBadCStream("Bad cblk width exp code: %d.", (int)cblkExpX);
    }
    if(cblkExpY > 8) {
        throwBadCStream("Bad cblk height exp code: %d.", (int)cblkExpY);
    }
    coding->cblkSize.x = 1 << (2 + cblkExpX);
    coding->cblkSize.y = 1 << (2 + cblkExpY);
    if(coding->cblkSize.x * coding->cblkSize.y > 4096) {
        throwBadCStream("Codeblock size %d x %d exceeds 4096 pixels.",
                        coding->cblkSize.x, coding->cblkSize.y);
    }
    
    // Codeblock coding style
    const u8 codingStyle = stream.readU8();
    coding->bypassAC = codingStyle & 0x01;
    coding->resetProb = codingStyle & 0x02;
    coding->termAll = codingStyle & 0x04;
    coding->vericalCausal = codingStyle & 0x08;
    coding->predictTerm = codingStyle & 0x10;
    coding->segSymbols = codingStyle & 0x20;
    
    // Type fo DWT
    const u8 dwtTypeCode = stream.readU8();
    switch(dwtTypeCode) {
        case 0: coding->reversible = false; break;
        case 1: coding->reversible = true; break;
        default: throwBadCStream("Bad DWT type: %x.", (int)dwtTypeCode);
    }
    
    // add pointer to precinct sizes
    coding->precSizesPtr = stream.pos();
    coding->precSizesCount = explicitPrecSizes ? stream.bytesRemaining() : 0;
}



/// Processes COD marker, saving data into places specified by parameters.
/// @param stream          reader of codestream bytes
/// @param tileCodingIdx   index of output structure for general tile info
/// @param tCompCodingIdx  index of output structure for component info
/// @param iter            packet iterator to be initialized
void Tier2::readCODMarker(T2Reader & stream, const int tileCodingIdx,
                          const int tCompCodingIdx, PacketIterator & iter) {
    // pointer to output structure
    TileCoding * const coding = &img->tileCoding[tileCodingIdx];
    
    // coding style
    const u8 codingStyle = stream.readU8();
    coding->useSop = codingStyle & 2;  // bit #2
    coding->useEph = codingStyle & 4;  // bit #3
    
    // Progression order
    const u8 ordCode = stream.readU8();
    switch(ordCode) {
        case 0: iter.setDefault(PO_LRCP); break;
        case 1: iter.setDefault(PO_RLCP); break;
        case 2: iter.setDefault(PO_RPCL); break;
        case 3: iter.setDefault(PO_PCRL); break;
        case 4: iter.setDefault(PO_CPRL); break;
        default: throwBadCStream("Bad progression code: %x.", (int)ordCode);
    }
    
    // Layer count
    coding->layerCount = stream.readU16();
    if(0 == coding->layerCount) {
        throwBadCStream("Zero layers signalized.");
    }
    
    // MCT
    const u8 mct_code = stream.readU8();
    switch(mct_code) {
        case 0: coding->useMct = false; break;
        case 1: coding->useMct = true;  break;
        default: throwBadCStream("Bad MCT code: %x.", (int)mct_code);
    }
    
    // Process single-component-related stuff.
    readTCompCoding(stream, tCompCodingIdx, codingStyle & 1);
}



/// Processes coding style-component.
/// @param mStream         reader of marker segment codestream
/// @param tCompCodingIdx  index of output structure
/// @return index of affected component
int Tier2::readCOCMarker(T2Reader & mStream, const int tCompCodingIdx) {
    // read component index
    const int compIdx = readCompIdx(mStream, img);
    
    // read coding style
    const u8 cStyleCode = mStream.readU8();
    if(cStyleCode & ~1) { // other bits set than first one
        throwBadCStream("Bad component coding style: %x.", (int)cStyleCode);
    }
    
    // use generic function for parsing component-related coding stuff
    readTCompCoding(mStream, tCompCodingIdx, cStyleCode);
    
    // return component index
    return compIdx;
}



/// Processes QCD and QCC markers.
/// @param reader  codestream reader pointing to quantization marker body
/// @param tCompQuantIdx  index of the output structure
void Tier2::readQuantMarker(T2Reader & reader, const int tCompQuantIdx) {
    // pointer to the structure to place quantization settings into
    TCompQuant * const quant = &tCompQuants[tCompQuantIdx];
    
    // read quantization style packed with number of guard bits
    const u8 packedStyle = reader.readU8();
    
    // unpack quantization mode
    switch(packedStyle & 0x1F) {
        case 0: quant->mode = QM_NONE; break;
        case 1: quant->mode = QM_IMPLICIT; break;
        case 2: quant->mode = QM_EXPLICIT; break;
        default: throwBadCStream("Bad quantization mode code: %x.",
                                 (int)(packedStyle & 0x1F));
    }
    
    // unpack number of guard bits
    quant->guardBitCount = packedStyle >> 5;
    
    // put the pointer and number of items to the structure
    quant->stepsizePtr = reader.pos();
    quant->stepsizeBytes = reader.bytesRemaining();
}



/// Adds codeblocks from some band into the partially initialized precinct.
/// @param prec     pointer to the partially initialized precinct
/// @param bandIdx  index of the band structure
/// @param cblkSize   size of codeblocks in the band
void Tier2::precAddBand(Prec * const prec, const int bandIdx,
                        const XY & cblkSize) {
    // add new precinct-band structure to the precinct
    PrecBand * precBand = precBands.get();
    precBand->next = prec->bands;
    prec->bands = precBand;
    
    // get pointer to the band
    const Band * const band = &img->bands[bandIdx];
    
    // numbers of included codeblocks along both axes
    const XY cblkCount = precPartition(prec->pixBegin, prec->pixEnd,
                                       band->pixBegin, band->pixEnd,
                                       cblkSize);
    
    // reserve tag tree instance
    precBand->zbplns.reset(cblkCount);
    precBand->incl.reset(cblkCount);
    precBand->cblkOffset = img->cblks.reserveMore(cblkCount.x * cblkCount.y);
    Cblk * cblk = &img->cblks[precBand->cblkOffset];
    
    // iterate through the band with codeblock-sized steps to get all 
    // codeblocks
    for(int y = prec->pixBegin.y; y < prec->pixEnd.y; y += cblkSize.y) {
        // skip this codeblock-row if there is no intersection with the band
        const int beginY = max(band->pixBegin.y, y);
        const int sizeY = min(band->pixEnd.y, y + cblkSize.y) - beginY;
        if(sizeY > 0) {
            // add all codeblocks of the codeblock row
            for(int x = prec->pixBegin.x; x < prec->pixEnd.x; x += cblkSize.x) {
                // skip the codeblock if there is no intersection with the band
                const int beginX = max(band->pixBegin.x, x);
                const int sizeX = min(band->pixEnd.x, x + cblkSize.x) - beginX;
                if(sizeX > 0) {
                    // initialize unused codeblock structure
                    cblk->bandIdx = bandIdx;
                    cblk->firstSegIdx = -1;
                    cblk->lastSegIdx = -1;
                    cblk->pos = XY(beginX, beginY);
                    cblk->size = XY(sizeX, sizeY);
                    cblk->stdSize = cblkSize;
                    cblk->bplnCount = 0;
                    cblk->segLenBits = 0;  // 0 == not included yet
                    cblk->passCount = 0;
                    cblk->totalBytes = 0;
                    
                    // reserve EBCOT temporary storage for the codeblock
                    // and update total buffer size for EBCOT
                    cblk->ebcotTemp = img->ebcotTempSize;
                    img->ebcotTempSize
                            += Ebcot::cblkTempSize(cblk->size, cblkSize);
                    
                    // debug
                    log->debug("      Adding codeblock (%d-%d)x(%d-%d)",
                               beginX, beginX + sizeX, beginY, beginY + sizeY);
                    
                    // advance to next unused codeblock structure
                    cblk++;
                }
            }
        }
    }
    
    // check the count of actually added codeblocks
    const int realCblkCount = cblk - &img->cblks[precBand->cblkOffset];
    if(realCblkCount != cblkCount.x * cblkCount.y) {
        throw Error(J2KD_ERROR_UNKNOWN, "Runtime error - bad codeblock count "
                    "(found %d, but expected %d).",
                    realCblkCount, cblkCount.x * cblkCount.y);
    }
}



/// Prepares structure of specified partially initialized tile-component.
/// @param tCompIdx  index of the tile-component structure
void Tier2::tileCompInit(const int tCompIdx) {
    // get pointers to the tile component and its coding and quantization
    TComp * const tComp = &img->tComps[tCompIdx];
    const TCompCoding * const cod = &img->tCompCoding[tComp->codingIdx];
    const TCompQuant * const quant = &tCompQuants[tComp->quantIdx];
    
    // pointer to related component info
    const Comp * const comp = &img->comps[tComp->compIdx];
    
    // pointer to related tile and its coding
    const Tile * const tile = &img->tiles[tComp->tileIdx];
    const TileCoding * const tCod = &img->tileCoding[tile->tileCodingIdx];
    
    // layer count and component index
    const int layers = tCod->layerCount;
    const int compIdx = tComp->compIdx;
    
    // reserve resolution structures
    tComp->resCount = cod->dwtLevelCount + 1;
    tComp->resIdx = img->res.reserveMore(tComp->resCount);
    
    // reader of stepsizes for the resolution
    StepsizeReader stepsizeReader(quant);
    
    log->debug("Adding component #%d (%d DWT levels):",
               compIdx, cod->dwtLevelCount);
    
    // decompose the tile-component into resolutions (beginning with res #0)
    for(int resIdx = 0; resIdx < tComp->resCount; resIdx++) {
        // initialize the resoution
        Res * const res = &img->res[tComp->resIdx + resIdx];
        res->resolution = resIdx;
        res->dwtCount = min(tComp->resCount - resIdx, cod->dwtLevelCount);
        res->tCompIdx = tCompIdx;
        
        // reserve band structures
        res->bandCount = resIdx ? 3 : 1;
        res->bandOffset = img->bands.reserveMore(res->bandCount);
        Band * const resBandsPtr = &img->bands[res->bandOffset];
        
        // begins and ends of lowpass band (along both axes)
        const XY loPixBegin = loBandCoord(tile->pixBegin, res->dwtCount);
        const XY loPixEnd = loBandCoord(tile->pixEnd, res->dwtCount);
        
        // lowest resolution?
        if(resIdx) {
            // ends of highpass bands
            const XY hiPixBegin = hiBandCoord(tile->pixBegin, res->dwtCount);
            const XY hiPixEnd = hiBandCoord(tile->pixEnd, res->dwtCount);
            
            // initialize 3 band structures
            resBandsPtr[0].orient = ORI_HL;
            resBandsPtr[0].pixBegin = XY(hiPixBegin.x, loPixBegin.y);
            resBandsPtr[0].pixEnd = XY(hiPixEnd.x, loPixEnd.y);
            resBandsPtr[0].stepsize = comp->bitDepth + 1;
            resBandsPtr[1].orient = ORI_LH;
            resBandsPtr[1].pixBegin = XY(loPixBegin.x, hiPixBegin.y);
            resBandsPtr[1].pixEnd = XY(loPixEnd.x, hiPixEnd.y);
            resBandsPtr[1].stepsize = comp->bitDepth + 1;
            resBandsPtr[2].orient = ORI_HH;
            resBandsPtr[2].pixBegin = hiPixBegin;
            resBandsPtr[2].pixEnd = hiPixEnd;
            resBandsPtr[2].stepsize = comp->bitDepth + 2;
        } else {
            // initialize the single LL band
            resBandsPtr[0].orient = ORI_LL;
            resBandsPtr[0].pixBegin = loPixBegin;
            resBandsPtr[0].pixEnd = loPixEnd;
            resBandsPtr[0].stepsize = comp->bitDepth;
        }
        
        // precinct and codeblock sizes in this resolution
        const XY precSize = getResPrecSize(cod, resIdx);
        const XY cblkSize = min(cod->cblkSize, precSize);
        
        // debug
        log->debug("  Adding resolution #%d, dwt %d, prec %dx%d, "
                   "cblk %dx%d", res->resolution, res->dwtCount, precSize.x,
                   precSize.y, cblkSize.x, cblkSize.y);
        
        // reserve space for precincts
        const XY bandBegin = loBandCoord(tile->pixBegin, res->dwtCount);
        const XY bandEnd = hiBandCoord(tile->pixEnd, res->dwtCount);
        const XY precMask = precSize - 1;
        const XY precBegin = bandBegin & ~precMask;           // round down
        const XY precEnd = (bandEnd + precMask) & ~precMask;  // round up
        res->precCount = (precBegin - precEnd) / precSize;
        
        // Finish band initializaition.
        for(int bandIdx = 0; bandIdx < res->bandCount; bandIdx++) {
            // pointer to the band
            Band * const band = resBandsPtr + bandIdx;
            
            // read band's stepsize (posssibly only exponent)
            const Stepsize stepsize = stepsizeReader.next();
            
            // reserve output space for the band
            const XY bandSize = band->pixEnd - band->pixBegin;
            band->outPixStride = rndUp(bandSize.x, 8);  // align each row
            band->outPixOffset = img->bandsPixelCount;
            img->bandsPixelCount += band->outPixStride * bandSize.y;
            
            // initialize band's stepsize
            // (now, it's value is set to dynamic range of the band)
            band->stepsize = std::pow(2.0, band->stepsize - stepsize.exponent)
                           * (1.0 + stepsize.mantisa / 2048.0);
            
            // finalize initialization of the band
            band->resIdx = tComp->resIdx + resIdx;
            band->bitDepth = stepsize.exponent + quant->guardBitCount - 1;
            band->reversible = cod->reversible;
            
            // debug output
            const static char * orientationNames[] = {"LL", "HL", "LH", "HH"};
            log->debug("    Adding band %s (%d-%d)x(%d-%d), "
                       "stride %d, stepsize (m%d, e%d), %d bpp",
                       orientationNames[(int)band->orient], band->pixBegin.x,
                       band->pixEnd.x, band->pixBegin.y, band->pixEnd.y,
                       band->outPixStride, stepsize.mantisa,
                       stepsize.exponent, band->bitDepth);
        }
        
        // reserve space for resolution's output pixels
        if(resIdx) {
            // size of resolution:
            res->begin = loBandCoord(tile->pixBegin, res->dwtCount - 1);
            res->end = loBandCoord(tile->pixEnd, res->dwtCount - 1);
            const XY resSize = res->end - res->begin;
            res->outPixOffset = img->bandsPixelCount;
            res->outPixStride = rndUp(resSize.x, 8);  // align each row
            img->bandsPixelCount += res->outPixStride * resSize.y;
        } else {
            // resolution #0 => output is same as input (no DWT to be done)
            res->outPixStride = resBandsPtr->outPixStride;
            res->outPixOffset = resBandsPtr->outPixOffset;
            res->begin = resBandsPtr->pixBegin;
            res->end = resBandsPtr->pixEnd;
        }
        
        // last resolution's output is also a tile-component's output
        // (following indices are overwritten in each resolution, keeping 
        // only those indices from last resolution)
        tComp->outPixOffset = res->outPixOffset;
        tComp->outPixStride = res->outPixStride;
        
        // select right segment info getter for all added precincts
        const SegInfo * (*segInfoGetter)(u8) = getSegInfoNormal;
        if(cod->bypassAC && cod->termAll) {
            segInfoGetter = getSegInfoTermAllSelectiveBypass;
        } else if(cod->bypassAC) {
            segInfoGetter = getSegInfoSelectiveBypass;
        } else if(cod->termAll) {
            segInfoGetter = getSegInfoTermAll;
        }
        
        // reserve precincts and add codeblocks to them
        XY pos;
        for(pos.y = precBegin.y; pos.y < precEnd.y; pos.y += precSize.y) {
            for(pos.x = precBegin.x; pos.x < precEnd.x; pos.x += precSize.x) { 
                Prec * const prec = precincts.get();
                prec->cblkCount = 0;
                prec->cblkOffset = img->cblks.count();
                prec->bands = 0;
                prec->pixBegin = pos;
                prec->pixEnd = pos + precSize;
                prec->resIdx = tComp->resIdx + resIdx;
                prec->segInfoGetter = segInfoGetter;
                
                // put the precinct into the packet iterator
                const XY refPos(pos.x << res->dwtCount, pos.y << res->dwtCount);
                tile->iter->addPrecinct(prec, compIdx, resIdx, refPos, layers);
                
                // debug
                log->debug("    Adding precinct (%d-%d)x(%d-%d)",
                           prec->pixBegin.x, prec->pixEnd.x, prec->pixBegin.y, 
                           prec->pixEnd.y);
                
                // add codeblocks from all bands
                for(int bandIdx = res->bandCount; bandIdx--;) {
                    precAddBand(prec, res->bandOffset + bandIdx, cblkSize);
                }
                
                // update codeblock count of the precinct
                prec->cblkCount = img->cblks.count() - prec->cblkOffset;
            }
        }
    }
}



/// Decodes header of first part of the tile, preparing the tile structure 
/// accordingly.
/// @param tStream  codestream reader for tile-header markers
/// @param tileIdx  index of the tile structure in buffer of all tiles
void Tier2::decodeTileFirstHeader(T2Reader & tStream, const int tileIdx) {
    // the first tile-part of the tile - make sure that COD and QCD are set
    if(!qcdFound) { throwBadCStream("No QCD before SOT."); }
    if(!codFound) { throwBadCStream("No COD before SOT."); }
    
    // pointer to the tile
    Tile * const tile = &img->tiles[tileIdx];
    
    // partially initialize tile-components
    const int compCount = img->comps.count();
    const size_t tCompOffset = img->tComps.reserveMore(compCount);
    tile->tCompIdx = tCompOffset;
    for(size_t compIdx = 0; compIdx < img->comps.count(); compIdx++) {
        const size_t tCompIdx = tCompOffset + compIdx;
        TComp * const tComp = &img->tComps[tCompIdx];
        tComp->quantIdx = img->comps[compIdx].defQuantIdx;
        tComp->codingIdx = img->comps[compIdx].defCStyleIdx;
        tComp->compIdx = compIdx;
        tComp->tileIdx = tileIdx;
    }
    
    // remember number of default coding styles and quantization styles
    // for tile-components
    const int numDefaultQStyles = tCompQuants.count();
    const int numDefaultCodings = img->tCompCoding.count();
    
    // initialize tile's packet iterator
    tile->iter = iters.get();
    tile->iter->init(defaultProgression);
    // process all marker segments until SOD (start of data)
    while(true) {
        // read marker code
        const u16 markerCode = tStream.readU16();
        
        // stop marker processing if SOD marker found
        if(0xFF93 == markerCode) { break; }
            
        // read marker segment length otherwise, create reader for processing 
        // the marker and skip the marker with main tile reader
        const int len = tStream.readU16();
        if(!tStream.hasBytes(len - 2)) { 
            log->warning("T2: Incomplete segment %x.", (int)markerCode);
            tStream.skip(tStream.bytesRemaining());  // skip remaining bytes
            break;
        }
        T2Reader mStream(tStream.pos(), max(0, len - 2));
        tStream.skip(len - 2);
        
        // decide what to do according to marker
        if(0xFF52 == markerCode) {  // COD
            // reserve new structures for the info
            const int tileCodIdx = img->tileCoding.reserveIdx();
            const int tCompCodIdx = img->tCompCoding.reserveIdx();

            // read the coding 
            readCODMarker(mStream, tileCodIdx, tCompCodIdx, *tile->iter);
            
            // set new default coding for this tile
            tile->tileCodingIdx = tileCodIdx;
            
            // possibly set new coding styles to all tile-components
            for(int compIdx = img->comps.count(); compIdx--;) {
                // pointer to tile-component
                TComp * const tComp = &img->tComps[compIdx + tile->tCompIdx];
                
                // replace style only if not already explicitly set by tile COC
                if(tComp->codingIdx < numDefaultCodings) {
                    tComp->codingIdx = tCompCodIdx;
                }
            }
        } else if (0xFF53 == markerCode) { // COC
            // reserve new tilecomp coding structure and remember its index
            const int codIdx = img->tCompCoding.reserveIdx();
            
            // process the marker, getting the component index
            const int compIdx = readCOCMarker(mStream, codIdx);
            
            // assign newly created coding style to the right tile-component
            img->tComps[tile->tCompIdx + compIdx].codingIdx = codIdx;
        } else if (0xFF5C == markerCode) { // QCD
            // reserve new structure for the info
            const int quantIdx = tCompQuants.reserveIdx();
            
            // parse quantiation marker, putting it into newly reserved struct
            readQuantMarker(mStream, quantIdx);
            
            // set new default quantization index to all components,
            // which still use some of default quantization settings
            for(int compIdx = img->comps.count(); compIdx--;) {
                // pointer to currently processed tile-component
                TComp * const tComp = &img->tComps[tile->tCompIdx + compIdx];
                
                // possibly change component quantization (only if not already 
                // explicitly specified using QCC marker)
                if(tComp->quantIdx < numDefaultQStyles) {
                    tComp->quantIdx = quantIdx;
                }
            }
        } else if (0xFF5D == markerCode) { // QCC
            // reserve new quantization structure for contents of this marker
            const int quantIdx = tCompQuants.reserveIdx();
            
            // read the index of the affected component
            const int compIdx = readCompIdx(mStream, img);
            
            // extract contents of the QCC marker into the structure
            readQuantMarker(mStream, quantIdx);
            
            // assign new quantization settings to affected tile-component
            img->tComps[tile->tCompIdx + compIdx].quantIdx = quantIdx;
        } else if (0xFF5E == markerCode) { // RGN
            throwUnsupported("T2: RGN marker not supported.");
        } else if (0xFF5F == markerCode) { // POC
            readPoc(tile->iter, mStream);
        } else if (0xFF58 == markerCode) { // PLT
            log->info("T2: skipping unsupported PLT in tile header.");
        } else if (0xFF61 == markerCode) { // PPT
            // TODO: implement
            throwUnsupported("T2: PPT marker not implemented yet.");
        } else if (0xFF64 == markerCode) { // COM
            log->info("T2: skipping comment in tile header.");
        } else { // unknown marker
            log->info("T2: unknown marker in tile header: %x.",
                      (int)markerCode);
        }
    }
    
    // finish initialization of tile-components now
    for(int compIdx = 0; compIdx < compCount; compIdx++) {
        tileCompInit(tCompOffset + compIdx);
    }
}



/// Decodes header of other-than-first part of the tile,
/// updating the tile structure accordingly.
/// @param tStream  codestream reader for the tilepart
/// @param tileIdx  index of the tile
void Tier2::decodeTileOtherHeader(T2Reader & tStream, const int tileIdx) {
    // make sure that COD and QCD are set
    if(!qcdFound) { throwBadCStream("No QCD before SOT."); }
    if(!codFound) { throwBadCStream("No COD before SOT."); }
    
    // pointer to the tile
    Tile * const tile = &img->tiles[tileIdx];
    
    // process all marker segments until SOD (start of data)
    while(true) {
        // read marker code
        const u16 markerCode = tStream.readU16();
        
        // stop marker processing if SOD marker found
        if(0xFF93 == markerCode) { break; }
            
        // read marker segment length otherwise, create reader for processing 
        // the marker and skip the marker with main tile reader
        const int len = tStream.readU16();
        if(!tStream.hasBytes(len - 2)) { 
            log->warning("T2: Incomplete segment %x.", (int)markerCode);
            tStream.skip(tStream.bytesRemaining());  // skip remaining bytes
            break;
        }
        T2Reader mStream(tStream.pos(), max(0, len - 2));
        tStream.skip(len - 2);
        
        if (0xFF5F == markerCode) { // POC
            readPoc(tile->iter, mStream);
        } else if (0xFF58 == markerCode) { // PLT
            log->info("T2: skipping unsupported PLT in tile header.");
        } else if (0xFF61 == markerCode) { // PPT
            // TODO: implement
            throwUnsupported("T2: PPT marker not implemented yet.");
        } else if (0xFF64 == markerCode) { // COM
            log->info("T2: skipping comment in tile-part header.");
        } else { // unknown marker
            log->info("T2: unknown marker in tile-part header: %x.",
                      (int)markerCode);
        }
    }
}



/// Reads info about single codeblock from packet header, 
/// updating decoder's image structure accordingly.
/// @param cblk        pointer to codeblock structure
/// @param reader      packet header bits reader
/// @param inclTree    inclusion tag tree
/// @param zbplnsTree  tag tree with zero-bitplanes information
/// @param layer       zero based layer index of hte precinct
/// @param coords      coordinates of the codeblock in precinct
/// @param getSegInfo  pointer to getter of info about code segments
void Tier2::readCblk(Cblk * const cblk, T2Reader & reader, TagTree & inclTree,
                     TagTree & zbplnsTree, const int layer, const XY & coords,
                     const SegInfo* (*getSegInfo)(u8)) {
    // skip if not included
    if(cblk->segLenBits) {
        if(0 == reader.readBit()) {
            return;  // decide according to single bit if included before
        }
    } else {
        if (layer == inclTree.value(coords, layer + 1, reader)) {
            // included for the first time => read zero bitplane count 
            // and set number of encoded bitplanes
            cblk->bplnCount = img->bands[cblk->bandIdx].bitDepth
                            - zbplnsTree.value(coords, reader);
            cblk->segLenBits = 3;
        } else {
            return; // still not included
        }
    }
    
    // get number of new coding passes in current packet
    const int initialPassCount = cblk->passCount;
    cblk->passCount += readNumPasses(reader);
    
    // possibly increase size of variable-length codeword segment
    cblk->segLenBits += reader.readOneBits();
    
    // distribute passes into segments (segments are delimited by arithmetic
    // coder terminations, which are determined by coding style)
    for(int passIdx = initialPassCount; passIdx < cblk->passCount;) {
        // allocate segment and possibly chain it together with other segments
        const int segIdx = img->segs.reserveIdx();
        if(-1 == cblk->lastSegIdx) { // first segment of the codeblock
            cblk->firstSegIdx = segIdx;
        } else {
            img->segs[cblk->lastSegIdx].nextSegIdx = segIdx;
        }
        cblk->lastSegIdx = segIdx;
        
        // get pointer to the segment and initialize it
        Seg * const seg = &img->segs[segIdx];
        seg->cblkIdx = (cblk - &img->cblks[0]);
        seg->nextSegIdx = -1;
        
        // get info about current codestream segment
        const SegInfo * const segInfo = getSegInfo(cblk->passCount);
        seg->passCount = min(cblk->passCount - passIdx, segInfo->maxPassCount);
        seg->bypassAC = segInfo->bypassAC;
        
        // read byte count
        const int byteCountBits = cblk->segLenBits + ilog2(seg->passCount);
        seg->codeByteCount = reader.readBits(byteCountBits);
        cblk->totalBytes += seg->codeByteCount;
        
        // advance to next segment
        passIdx += seg->passCount;
    }
}



/// Reads one packet from given stream (updating the stream state).
/// @param bodies     packet body stream
/// @param headers    packet header stream
/// @param detectSOP  detect possible SOP markers
/// @param detectEPH  detect possible EPH markers
/// @param iter       pointer to packet iterator for the tile
/// @return true = packet read OK, false otherwise (e.g. end of codestream)
bool Tier2::readPacket(T2Reader & bodies, T2Reader & headers, 
                       const bool detectSOP, const bool detectEPH,
                       PacketIterator * const iter) {
    // Have enough bytes for packet bodies and headers? (Wait for next 
    // tile-part before selecting next packet if not having bytes.)
    log->debug("Header bytes remaining: %d.", (int)headers.bytesRemaining());
    if(headers.bytesRemaining() < 1) {
        return false;
    }
    
    // index of first added codestream segment
    const int firstSegmentIdx = img->segs.count();
    
    // select next precinct to decode
    int layerIdx;
    Prec * const prec = iter->next(layerIdx);
    if(0 == prec) {
        return false;
    }
    
    // possibly detect and skip 6 bytes of SOP marker
    if(detectSOP && 0xFF91 == bodies.getU16()) {
        bodies.skip(6);
    }
        
    // have any bytes for the precinct in this packet?
    if(headers.readBit()) {  // 0 == empty
        // debug
        log->debug("Reading packet (comp %d, res %d, x %d, y %d, layer %d):",
                   img->tComps[img->res[prec->resIdx].tCompIdx].compIdx,
                   img->res[prec->resIdx].resolution,
                   prec->pixBegin.x,
                   prec->pixBegin.y,
                   layerIdx);
        
        // nonempty packet => read info for codeblocks of all precinct's bands
        for(PrecBand * cblks = prec->bands; cblks; cblks = cblks->next) {
            // pointer to first codeblock of this precinct-band
            Cblk * cblk = &img->cblks[cblks->cblkOffset];
            
            // for each codeblock in the precinct's band portion
            XY pos;
            for(pos.y = 0; pos.y < cblks->incl.size().y; pos.y++) {
                for(pos.x = 0; pos.x < cblks->incl.size().x; pos.x++) {
                    // read codeblock info, advancing to next codeblock
                    readCblk(cblk++, headers, cblks->incl, cblks->zbplns,
                             layerIdx, pos, prec->segInfoGetter);
                }
            }
        }
    } else {
        log->debug("Skipping empty packet.");
    }
    
    // discard remaining bits to align packet header end to byte boundary
    headers.align();
    
    // possibly skip EPH header terminating the packet header
    if(detectEPH && 0xFF92 == headers.getU16()) {
        headers.skip(2);
    }
    
    // add begins of all codestream segments
    const int segmentsOffset = bodies.pos() - cStreamBeginPtr;
    int segmentBytesTotal = 0;
    for(u32 segIdx = firstSegmentIdx; segIdx < img->segs.count(); segIdx++) {
        Seg * const seg = &img->segs[segIdx];
        seg->codeByteOffset = segmentsOffset + segmentBytesTotal;
        segmentBytesTotal += seg->codeByteCount;
        
        // debug
        log->debug("  Adding segment: %s, cblk: %d, %dB (%d - %d), %d passes.",
                   seg->bypassAC ? "bypass" : "AC",
                   seg->cblkIdx,
                   seg->codeByteCount, 
                   seg->codeByteOffset,
                   seg->codeByteOffset + seg->codeByteCount,
                   seg->passCount);
    }
    
    // skip all segment bytes at once
    bodies.skip(segmentBytesTotal);
    
    // indicate success
    return true;
}



/// Processes tile-part marker semgent and following data.
/// @param tStream  marker segment stream, starting after len field
/// @param len      length of the marker segment
/// @return Total size of the tilepart (SOT marker segment + SOD + data)
size_t Tier2::decodeTilepart(T2Reader & tStream, const u16 len) {
    // check marker-segment length
    log->debug("Tile-part begin.");
    if(len != 10) { throwBadCStream("Lsot = %d.", (int)len); }
    
    // extract parameters
    const u16 tileIdx = tStream.readU16();
    const u32 tPartSize = tStream.readU32();
    const u8 tPartIdx = tStream.readU8();
    tStream.readU8();  // ignored tile-part count
    
    // check tile-part index
    Tile * const tile = &img->tiles[tileIdx];
    if(tPartIdx == tile->nextTPartIdx) {
        tile->nextTPartIdx++; // OK
    } else {
        throwBadCStream("Expected part #%d of tile #%d, but found part #%d.",
                        tile->nextTPartIdx, (int)tileIdx, (int)tPartIdx);
    }
    
    // get stream for tile-part payload
    const size_t remainingTPartByteCount = tPartSize
            ? tPartSize - 12  // 12 bytes already read
            : cStreamEndPtr - tStream.pos();
    T2Reader bStream(tStream.pos(), remainingTPartByteCount);
    
    // parse header of tile
    if(tPartIdx == 0) {
        decodeTileFirstHeader(bStream, tileIdx);
    } else {
        decodeTileOtherHeader(bStream, tileIdx);
    }
    
    // process all packets
    // TODO: possibly separate stream for headers from the stream for packet 
    // bodies to implement packed headers
    while(readPacket(bStream,
                     bStream,
                     img->tileCoding[tile->tileCodingIdx].useSop,
                     img->tileCoding[tile->tileCodingIdx].useEph,
                     tile->iter)
         );
    
    // finally return the size of whole tilepart
    log->debug("Tile-part end.");
    return tPartSize;
}




/// Processes one marker segment, advancing the codestream reader 
/// to next one.
/// @param cStream  codestream reader for processed marker segment
/// @return true if some marker segment may follow, false if end reached
bool Tier2::processMainHeaderMarkerSegment(T2Reader & cStream) {
    // read marker code and length
    if(!cStream.hasBytes(4)) {
        if(!cStream.hasBytes(2) || 0xFFd9 != cStream.readU16()) {
            log->warning("T2: Unexpected end of codestream.");
        }
        return false;
    }
    const u16 markerCode = cStream.readU16();
    const u16 len = cStream.readU16();
    T2Reader mStream(cStream.pos(), max(0, len - 2));
    
    // Process the marker itself, if have enough bytes
    if(!cStream.hasBytes(len - 2)) { 
        log->warning("T2: Incomplete segment %x.", (int)markerCode);
        return false;
    } else {
        cStream.skip(len - 2);
    }
    switch(markerCode) {
        case 0xFF90: // Start of tile-part (skip more bytes)
            cStream.skip(decodeTilepart(mStream, len) - 12);
            break;
        case 0xFF52: // main header coding style default
            readCODMarker(mStream, 0, 0, defaultProgression);
            codFound = true;
            break;
        case 0xFF53: // coding style component
            {
                const int idx = img->tCompCoding.reserveIdx();
                const int compIdx = readCOCMarker(mStream, idx);
                img->comps[compIdx].defCStyleIdx = idx;
            }
            break;
        case 0xFF5C: // quantization default
            qcdFound = true;
            readQuantMarker(mStream, 0);
            break;
        case 0xFF5D: // quantization component
            {
                const int idx = tCompQuants.reserveIdx();
                const int compIdx = readCompIdx(mStream, img);
                readQuantMarker(mStream, idx);
                img->comps[compIdx].defQuantIdx = idx;
            }
            break;
        case 0xFF5E: // region of interest => not supported
            throwUnsupported("T2: Region of interest not supported.");
            break;
        case 0xFF5F: // progression order change
            readPoc(&defaultProgression, mStream);
            break;
        case 0xFF63: // component registration => ignored
            log->warning("T2: Ignoring unsupported CRG marker segment.");
            break;
        case 0xFF64: // comment => ignored
            log->info("T2: Skipping comment marker segment (COM).");
            break;
        case 0xFF55: // TLM => ignored
            log->info("T2: Ignoring TLM marker segment.");
            break;
        case 0xFF57: // PLM => ignored
            log->info("T2: Ignoring PLM marker segment.");
            break;
        case 0xFF60: // Packed packet headers: main header
            throwUnsupported("T2: Packed packet headers not implemented yet.");
            // TODO: NOT IMPLEMENTED
            break;
        default: 
            log->warning("T2: Skipped unknown main header marker segment"
                         " (marker: %x, len: %d).", (int)markerCode, (int)len);
    }
    
    // signalize that this was not the end
    return true;
}



/// Codeblock comparator for decoding-optimal codeblock permutation.
class CblkComp {
private:
    const Cblk * const cblks;  // codeblocks info
public:
    CblkComp(const Cblk * const cblks) : cblks(cblks) {}
    bool operator()(const u32 & aIdx, const u32 & bIdx) const {
        const Cblk & a = cblks[aIdx];
        const Cblk & b = cblks[bIdx];
        
        // compare by standard codeblock size, then by number of passes,
        // then by number of bytes and finally by index
        if(a.stdSize != b.stdSize) return a.stdSize > b.stdSize;
        if(a.passCount != b.passCount) return a.passCount > b.passCount;
//         if(a.totalBytes != b.totalBytes) return a.totalBytes > b.totalBytes;
        return false;
    }
};



/// Composes codeblock permutation.
/// (Groups similar codeblocks together for coalescent decoding.)
/// @param image  initialized image structure
static void composeCblkPermutation(Image * const image){
    // pointer to resized permutation buffer
    const int cblkCount = image->cblks.count();
    image->cblkPerm.reserveMore(cblkCount);
    u32 * const perm = &image->cblkPerm[0];
    
    // initialize by "no permutation" and then reorder
    for(u32 i = cblkCount; i--;) {
        perm[i] = i;
    }
    std::sort(perm, perm + cblkCount, CblkComp(&image->cblks[0]));
}



/// Decodes given codestream on CPU.
/// @param image  pointer to output image structure
/// @param codestreamPtr  pointer to input codestream
/// @param codestreamSize  input codestream size in bytes
/// @param log  logger for tracing decoding process
void Tier2::analyze(Image * const image,
                    const u8 * const codestreamPtr,
                    const size_t codestreamSize,
                    Logger * const log) {
    // set common attributes for current codestream decoding
    this->img = image;
    this->log = log;
    this->cStreamBeginPtr = codestreamPtr;
    this->cStreamEndPtr = codestreamPtr + codestreamSize;
    
    // clear buffer usages before analysis
    tCompQuants.clear();
    qcdFound = false;
    codFound = false;
    defaultProgression.reset();
    precincts.reuse();
    precBands.reuse();
    iters.reuse();
    img->ebcotTempSize = 0;
    
    // reserve first COD and QCD for default configurations
    image->tileCoding.reserveIdx();
    image->tCompCoding.reserveIdx();
    tCompQuants.reserveIdx();
    
    // initialize codestream reader
    T2Reader cStream((u8*)codestreamPtr, codestreamSize);
    
    // analyze SIZ marker (codestream header)
    cStream.skip(readSiz(T2Reader(cStream)));
    
    // analyze rest of the stream (all other markers), composing the 
    // image structure and decoding packet headers
    while(processMainHeaderMarkerSegment(cStream));
    
    // compose codeblock permutation for faster decoding
    composeCblkPermutation(image);
}



/// Extracts image info directly from the codestream.
/// @param cStreamPtr   pointer to codestream
/// @param cStreamSize  size of codestream in bytes
/// @param outInfoPtr   pointer to output structure
void Tier2::getImageInfo(const u8 * const cStreamPtr,
                         const size_t cStreamSize,
                         ImageInfo * const outInfoPtr) {
    // reader for the codestream
    T2Reader reader(cStreamPtr, cStreamSize);
    if(!reader.hasBytes(42)) {
        throwBadCStream("Not enough codestream bytes to get image info.");
    }
    
    // extract the info
    reader.skip(6);
    outInfoPtr->capabilities = reader.readU16();
    outInfoPtr->image_end_x = reader.readU32();
    outInfoPtr->image_end_y = reader.readU32();
    outInfoPtr->image_begin_x = reader.readU32();
    outInfoPtr->image_begin_y = reader.readU32();
    outInfoPtr->tile_size_x = reader.readU32();
    outInfoPtr->tile_size_y = reader.readU32();
    outInfoPtr->tile_origin_x = reader.readU32();
    outInfoPtr->tile_origin_y = reader.readU32();
    outInfoPtr->comp_count = reader.readU16();
}



/// Extract info about one component directly from codestream.
/// @param cStreamPtr   pointer to codestream
/// @param cStreamSize  size of codestream in bytes
/// @param compIdx      index of the component
/// @param outInfoPtr   output structure for component info
void Tier2::getCompInfo(const u8 * const cStreamPtr,
                        const size_t cStreamSize,
                        const int compIdx,
                        ComponentInfo * const outInfoPtr) {
    // prepare reader for the component
    T2Reader reader(cStreamPtr + 40, cStreamSize - 40);
    
    // how many components are there?
    if(!reader.hasBytes(2)) {
        throwBadCStream("Not enough codestream bytes to get component count.");
    }
    const int compCount = reader.readU16();
    if(compIdx >= compCount) {
        throw Error(J2KD_ERROR_ARGUMENT_OUT_OF_RANGE,
                    "Component index #%d out of range (% components found).",
                    compIdx, compCount);
    }
    
    // skip enough bytes to get to the right component info
    reader.skip(compIdx * 3);
    if(!reader.hasBytes(3)) {
        throwBadCStream("Not enough codestream bytes to get "
                        "info about component #%d.", compIdx);
    }
    
    // read the component info
    const u8 compInfo = reader.readU8();
    outInfoPtr->is_signed = (compInfo & 0x80) ? 1 : 0;
    outInfoPtr->bit_depth = (compInfo & 0x7F) + 1;
    outInfoPtr->index = compIdx;
}



} // end of namespace cuj2kd

