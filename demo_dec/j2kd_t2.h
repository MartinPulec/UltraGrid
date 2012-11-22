///
/// @file    j2kd_t2.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Interface of T2 decoder for JPEG 2000 codestream.
///


#ifndef J2KD_T2_H
#define J2KD_T2_H

#include "j2kd_type.h"
#include "j2kd_image.h"
#include "j2kd_buffer.h"
#include "j2kd_logger.h"
#include "j2kd_tag_tree.h"
#include "j2kd_t2_type.h"
#include "j2kd_packet_iter.h"


namespace cuj2kd {



/// Tier II decoder, running on CPU.
class Tier2 {
private:
    /// components' quantization settings
    BufferPair<TCompQuant> tCompQuants;
    
    /// True if main header QCD already processed,
    bool qcdFound;
    
    /// True if main header COD already processed.
    bool codFound;
    
    /// default progression order
    PacketIterator defaultProgression;
    
    /// pool of precinct structures
    Pool<Prec> precincts;
    
    /// pool of codeblock ranges
    Pool<PrecBand> precBands;
    
    /// pool of packet iterators for tiles
    Pool<PacketIterator> iters;
    
    /// Currently composed output image structure.
    Image * img;
    
    /// Current logger.
    Logger * log;
    
    /// Pointer to begin of current codestream.
    const u8 * cStreamBeginPtr;
    
    /// Pointer to end of current codestream.
    const u8 * cStreamEndPtr;
    
    /// Reads progression orders and adds them to given packet iterator.
    /// @param iter  pointer to iterator to be extended with progression orders
    /// @param reader  reader of POC marker segment
    void readPoc(PacketIterator * const iter, T2Reader & reader);
    
    /// Analyzes SOC and SIZ marker.
    /// @param cStream  codestream reader
    /// @return number of bytes to skip to get to next marker.
    size_t readSiz(T2Reader cStream);

    /// Reads single-component related parts of COD anc COC marker segments.
    /// @param stream  reader to read info from
    /// @param tCompCodingIdx  index of output structure
    /// @param explicitPrecSizes  true if precinct sizes should be extracted
    void readTCompCoding(T2Reader & stream, const int tCompCodingIdx,
                         const bool explicitPrecSizes);

    /// Processes COD marker, saving data into places specified by parameters.
    /// @param stream          reader of codestream bytes
    /// @param tileCodingIdx   index of output structure for general tile info
    /// @param tCompCodingIdx  index of output structure for component info
    /// @param iter            packet iterator to be initialized
    void readCODMarker(T2Reader & stream, const int tileCodingIdx,
                       const int tCompCodingIdx, PacketIterator & iter);

    /// Processes coding style-component.
    /// @param mStream         reader of marker segment codestream
    /// @param tCompCodingIdx  index of output structure
    /// @return index of affected component
    int readCOCMarker(T2Reader & mStream, const int tCompCodingIdx);

    /// Processes QCD and QCC markers.
    /// @param reader  codestream reader pointing to quantization marker body
    /// @param tCompQuantIdx  index of the output structure
    void readQuantMarker(T2Reader & reader, const int tCompQuantIdx);

    /// Adds codeblocks from some band into the partially initialized precinct.
    /// @param prec     pointer to the partially initialized precinct
    /// @param bandIdx  index of the band structure
    /// @param cbSize   size of codeblocks in the band
    void precAddBand(Prec * const prec, const int bandIdx, const XY & cbSize);

    /// Prepares structure of specified partially initialized tile-component.
    /// @param tCompIdx  index of the tile-component structure
    void tileCompInit(const int tCompIdx);

    /// Decodes header of first part of the tile, preparing the tile structure 
    /// accordingly.
    /// @param tStream  codestream reader for tile-header markers
    /// @param tileIdx  index of the tile structure in buffer of all tiles
    void decodeTileFirstHeader(T2Reader & tStream, const int tileIdx);
    
    /// Decodes header of other-than-first part of the tile,
    /// updating the tile structure accordingly.
    /// @param tStream  codestream reader for the tilepart
    /// @param tileIdx  index of the tile
    void decodeTileOtherHeader(T2Reader & tStream, const int tileIdx);
    
    /// Reads info about single codeblock from packet header, 
    /// updating decoder's image structure accordingly.
    /// @param cblk        pointer to codeblock structure
    /// @param reader      packet header bits reader
    /// @param inclTree    inclusion tag tree
    /// @param zbplnsTree  tag tree with zero-bitplanes information
    /// @param layer       zero based layer index of hte precinct
    /// @param coords      coordinates of the codeblock in precinct
    /// @param getSegInfo  pointer to getter of info about code segments
    void readCblk(Cblk * const cblk, T2Reader & reader, TagTree & inclTree,
                  TagTree & zbplnsTree, const int layer, const XY & coords,
                  const SegInfo* (*getSegInfo)(u8));
    
    /// Reads one packet from given stream (updating the stream state).
    /// @param bodies     packet body stream
    /// @param headers    packet header stream
    /// @param detectSOP  detect possible SOP markers
    /// @param detectEPH  detect possible EPH markers
    /// @param iter       pointer to packet iterator for the tile
    /// @return true = packet read OK, false otherwise (e.g. end of codestream)
    bool readPacket(T2Reader & bodies, T2Reader & headers, 
                    const bool detectSOP, const bool detectEPH,
                    PacketIterator * const iter);
    
    /// Processes tile-part marker semgent and following data.
    /// @param tStream  marker segment stream, starting after len field
    /// @param len      length of the marker segment
    /// @return Total size of the tilepart (SOT marker segment + SOD + data)
    size_t decodeTilepart(T2Reader & tStream, const u16 len);
    
    /// Processes one marker segment, advancing the codestream reader 
    /// to next one.
    /// @param cStream  codestream reader for processed marker segment
    /// @return true if some marker segment may follow, false if end reached
    bool processMainHeaderMarkerSegment(T2Reader & cStream);
    
public:
    
    /// Decodes given codestream on CPU.
    /// @param image  pointer to output image structure
    /// @param codestreamPtr  pointer to input codestream
    /// @param codestreamSize  input codestream size in bytes
    /// @param log  logger for tracing decoding process
    void analyze(Image * const image,
                 const u8 * const codestreamPtr,
                 const size_t codestreamSize,
                 Logger * const log);

    /// Extracts image info directly from the codestream.
    /// @param cStreamPtr   pointer to codestream
    /// @param cStreamSize  size of codestream in bytes
    /// @param outInfoPtr   pointer to output structure
    static void getImageInfo(const u8 * const cStreamPtr,
                             const size_t cStreamSize,
                             ImageInfo * const outInfoPtr);

    /// Extract info about one component directly from codestream.
    /// @param cStreamPtr   pointer to codestream
    /// @param cStreamSize  size of codestream in bytes
    /// @param compIdx      index of the component
    /// @param outInfoPtr   output structure for component info
    static void getCompInfo(const u8 * const cStreamPtr,
                            const size_t cStreamSize,
                            const int compIdx,
                            ComponentInfo * const outInfoPtr);
}; // end of class Tier2



} // end of namespace cuj2kd


#endif // J2KD_T2_H

