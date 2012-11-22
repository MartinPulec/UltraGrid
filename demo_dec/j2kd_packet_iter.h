///
/// @file    j2kd_packet_iter.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Packet ordering definition for T2 of JPEG 2000 decoder.
///


#ifndef J2KD_PACKET_ITER_H
#define J2KD_PACKET_ITER_H

#include "j2kd_t2_type.h"
#include "j2kd_buffer.h"

namespace cuj2kd {


/// Packet iterator for T2 of JPEG 2000 decoder.
class PacketIterator {
private:
    /// Represents reference to single packet (a part of some 
    /// precinct in some layer).
    struct PacketRef {
        Prec * prec;        ///< pointer to the packet's precinct
        XY  pos;            ///< position of packet's precinct
        unsigned int res;   ///< resolution of packet' precinct (e.g. 0 for LL)
        unsigned int comp;  ///< component index of packet's precinct
        unsigned int layer; ///< zero-based index of the layer of precinct
    };
    
    /// Represents progression volume.
    struct ProgVolume {
        unsigned int resBegin;  ///< first resolution of the volume
        unsigned int resEnd;    ///< first resolution NOT in the volume
        unsigned int compBegin; ///< first component in the volume
        unsigned int compEnd;   ///< first component NOT in the volume
        unsigned int layerEnd;  ///< first layer not in the volume
        ProgOrder progression;  ///< progression in the volume
    };
    
    /// Predicate getting true for packets contained in given volume.
    class InVolumePredicate {
    private:
        const ProgVolume volume;
    public:
        InVolumePredicate(const ProgVolume & volume) : volume(volume) {}
        bool operator() (const PacketRef & packet) const;
    }; // end of class InVolumePredicate
    
    // Packet comparators for various progressions.
    static bool packetCompareRLCP(const PacketRef & a, const PacketRef & b);
    static bool packetCompareLRCP(const PacketRef & a, const PacketRef & b);
    static bool packetCompareRPCL(const PacketRef & a, const PacketRef & b);
    static bool packetComparePCRL(const PacketRef & a, const PacketRef & b);
    static bool packetCompareCPRL(const PacketRef & a, const PacketRef & b);
    
    /// Array of packet references.
    Buffer<PacketRef> packets;
    
    /// Array of progression volumes.
    Buffer<ProgVolume> volumes; 
    
    /// Index of next packet to be read.
    unsigned int nextReadIdx;
    
    /// true if packets are ordered, false if not yet
    bool ordered;
    
    /// default progression for packets not included in any volumes
    ProgOrder defaultProgression;
    
    /// reorders packets, making them ready for returning from 'next' function 
    void reorder();
    
    /// sorts given range of packets using given progression order
    static void sortPackets(PacketRef * const begin, PacketRef * const end,
                            const ProgOrder progression);
    
public:
    /// Removes all progression volumes and precincts 
    /// and reinitializes internal stuff.
    void reset();
    
    /// Initializes according to given packet iterator.
    /// @param iter  reference to sample packet iterator
    void init(const PacketIterator & iter);
    
    /// Adds new progression volume, affecting all following precincts.
    void addVolume(const unsigned int resBegin,
                   const unsigned int resEnd,
                   const unsigned int compBegin,
                   const unsigned int compEnd,
                   const unsigned int layerEnd,
                   const ProgOrder progression);
    
    /// Sets default progression order (used if not overriden by POC).
    /// @param progression  the progression order to be used by default
    void setDefault(const ProgOrder progression);
    
    /// Adds new precinct into iterator.
    /// @param prec     pointer to the precinct
    /// @param compIdx  index of precinct's component
    /// @param resIdx   index of packet's resolution (e.g. 0 for LL resolution)
    /// @param pos      position of the precinct (in reference sample grid)
    /// @param layers   number of layers for the precinct
    void addPrecinct(Prec * const prec, const int compIdx, const int resIdx,
                     const XY & pos, const unsigned int layers);
    
    /// Gets pointer to next precinct in current order or null if there are 
    /// no more packets.
    /// @param layer  reference to integer for layer number of the packet
    /// @return pointer to the precinct
    Prec * next(int & layer);
    
    /// Gets count of added packets.
    size_t count() const { return packets.count(); }
}; // end of class PacketIterator


} // end of namespace cuj2kd


#endif // J2KD_PACKET_ITER_H

