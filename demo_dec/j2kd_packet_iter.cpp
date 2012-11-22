///
/// @file    j2kd_packet_iter.cpp
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Packet ordering implementation for T2 of JPEG 2000 decoder.
///

#include <algorithm>
#include "j2kd_packet_iter.h"

namespace cuj2kd {



/// Removes all progression volumes and precincts.
void PacketIterator::reset() {
    ordered = false;
    nextReadIdx = 0;
    defaultProgression = PO_LRCP;
    packets.clear();
    volumes.clear();
}



/// Initializes according to given packet iterator.
/// @param iter  reference to sample packet iterator
void PacketIterator::init(const PacketIterator & iter) {
    // reset this and copy everything except of the packet list
    reset();
    volumes.reserveMore(iter.volumes.count());
    for(size_t i = 0; i < iter.volumes.count(); i++) {
        volumes[i] = iter.volumes[i];
    }
    nextReadIdx = 0;
    ordered = false;
    defaultProgression = iter.defaultProgression;
}



/// Adds new progression volume, affecting all following precincts.
void PacketIterator::addVolume(const unsigned int resBegin,
                               const unsigned int resEnd,
                               const unsigned int compBegin,
                               const unsigned int compEnd,
                               const unsigned int layerEnd,
                               const ProgOrder progression) {
    // add the volume
    ProgVolume volume;
    volume.resBegin = resBegin;
    volume.resEnd = resEnd;
    volume.compBegin = compBegin;
    volume.compEnd = compEnd;
    volume.layerEnd = layerEnd;
    volume.progression = progression;
    *(volumes.reservePtr()) = volume;
    
    // mark packet list as not ordered
    ordered = false;
}



/// Sets default progression order (used if not overriden by POC).
/// @param progression  the progression order to be used by default
void PacketIterator::setDefault(const ProgOrder progression) {
    defaultProgression = progression;
    ordered = false;
}



/// Adds new precinct into iterator.
/// @param prec     pointer to the precinct
/// @param compIdx  index of precinct's component
/// @param resIdx   index of packet's resolution (e.g. 0 for LL resolution)
/// @param pos      position of the precinct (in reference sample grid)
void PacketIterator::addPrecinct(Prec * const prec, const int compIdx,
                                 const int resIdx, const XY & pos,
                                 const unsigned int layers) {
    // prepare packet template for the precinct
    PacketRef packet;
    packet.prec = prec;
    packet.pos = pos;
    packet.res = resIdx;
    packet.comp = compIdx;
    
    // add all packets for the precinct
    for(packet.layer = 0; packet.layer < layers; packet.layer++) {
        *(packets.reservePtr()) = packet;
    }
    
    // mark list as nor ordered
    ordered = false;
}



/// Gets pointer to next precinct in current order.
/// @param layer  reference to integer for layer number of the packet
/// @return pointer to the precinct
Prec * PacketIterator::next(int & layer) {
    if(!ordered) {
        reorder();
    }
    if(nextReadIdx < packets.count()) {
        layer = packets[nextReadIdx].layer;
        return packets[nextReadIdx++].prec;
    }
    layer = 0;
    return 0;
}



/// reorders packets, making them ready for returning from 'next' function 
void PacketIterator::reorder() {
    // pointers to begin and end of the range of unsorted packets
    PacketRef * begin = &packets[nextReadIdx];
    PacketRef * const end = &packets[packets.count()];
    
    // for each progression volume:
    for(unsigned int volIdx = 0; volIdx < volumes.count(); volIdx++) {
        // partition packets into 2 groups: contained in the volume and others
        InVolumePredicate predicate(volumes[volIdx]);
        PacketRef * const volEnd = std::partition(begin, end, predicate);
        
        // sort packets in the volume
        sortPackets(begin, volEnd, volumes[volIdx].progression);
        
        // advance begin to first packet not included in any volume yet
        begin = volEnd;
    }
    
    // finally, use default progression for packets not contained in any volume
    sortPackets(begin, end, defaultProgression);
    ordered = true;
}



/// sorts given range of packets using given progression order
void PacketIterator::sortPackets(PacketRef * const begin,
                                        PacketRef * const end,
                                        const ProgOrder progression) {
    // select the right compare functor according to progression order
    switch(progression) {
        case PO_LRCP: std::sort(begin, end, packetCompareLRCP); break;
        case PO_RLCP: std::sort(begin, end, packetCompareRLCP); break;
        case PO_RPCL: std::sort(begin, end, packetCompareRPCL); break;
        case PO_PCRL: std::sort(begin, end, packetComparePCRL); break;
        case PO_CPRL: std::sort(begin, end, packetCompareCPRL); break;
        default: throw Error(J2KD_ERROR_UNKNOWN, "Unknown progression.");
    }
}



/// Implementation of packet partitioning into volumes.
bool PacketIterator::InVolumePredicate::
operator()(const PacketIterator::PacketRef& packet) const {
    return (packet.comp < volume.compEnd)
        && (packet.comp >= volume.compBegin)
        && (packet.layer < volume.layerEnd)
        && (packet.res < volume.resEnd)
        && (packet.res >= volume.resBegin);
}



/// Implementation of packet comparator for LRCP progression order.
bool PacketIterator::packetCompareLRCP(const PacketIterator::PacketRef& a,
                                       const PacketIterator::PacketRef& b) {
    if(a.layer != b.layer) { return a.layer < b.layer; }
    if(a.res != b.res) { return a.res < b.res; }
    if(a.comp != b.comp) { return a.comp < b.comp; }
    if(a.pos.y != b.pos.y) { return a.pos.y < b.pos.y; }
    if(a.pos.x != b.pos.x) { return a.pos.x < b.pos.x; }
    return false;
}



/// Implementation of packet comparator for RLCP progression order.
bool PacketIterator::packetCompareRLCP(const PacketIterator::PacketRef& a,
                                       const PacketIterator::PacketRef& b) {
    if(a.res != b.res) { return a.res < b.res; }
    if(a.layer != b.layer) { return a.layer < b.layer; }
    if(a.comp != b.comp) { return a.comp < b.comp; }
    if(a.pos.y != b.pos.y) { return a.pos.y < b.pos.y; }
    if(a.pos.x != b.pos.x) { return a.pos.x < b.pos.x; }
    return false;
}



/// Implementation of packet comparator for RPCL progression order.
bool PacketIterator::packetCompareRPCL(const PacketIterator::PacketRef& a,
                                       const PacketIterator::PacketRef& b) {
    if(a.res != b.res) { return a.res < b.res; }
    if(a.pos.y != b.pos.y) { return a.pos.y < b.pos.y; }
    if(a.pos.x != b.pos.x) { return a.pos.x < b.pos.x; }
    if(a.comp != b.comp) { return a.comp < b.comp; }
    if(a.layer != b.layer) { return a.layer < b.layer; }
    return false;
}



/// Implementation of packet comparator for PCRL progression order.
bool PacketIterator::packetComparePCRL(const PacketIterator::PacketRef& a,
                                       const PacketIterator::PacketRef& b) {
    if(a.pos.y != b.pos.y) { return a.pos.y < b.pos.y; }
    if(a.pos.x != b.pos.x) { return a.pos.x < b.pos.x; }
    if(a.comp != b.comp) { return a.comp < b.comp; }
    if(a.res != b.res) { return a.res < b.res; }
    if(a.layer != b.layer) { return a.layer < b.layer; }
    return false;
}



/// Implementation of packet comparator for CPRL progression order.
bool PacketIterator::packetCompareCPRL(const PacketIterator::PacketRef& a,
                                       const PacketIterator::PacketRef& b) {
    if(a.comp != b.comp) { return a.comp < b.comp; }
    if(a.pos.y != b.pos.y) { return a.pos.y < b.pos.y; }
    if(a.pos.x != b.pos.x) { return a.pos.x < b.pos.x; }
    if(a.res != b.res) { return a.res < b.res; }
    if(a.layer != b.layer) { return a.layer < b.layer; }
    return false;
}


} // end of namespace cuj2kd

