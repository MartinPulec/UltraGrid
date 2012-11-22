///
/// @file    j2kd_tag_tree.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Tag tree for T2 of JPEG 2000 decoder.
///

#ifndef J2KD_TAG_TREE_H
#define J2KD_TAG_TREE_H

#include "j2kd_t2_reader.h"

namespace cuj2kd {


/// The tag tree for decoder.
class TagTree {
    /// Represents one node of the tag tree.
    struct Node {
        Node * parent; ///< pointer to the parent node or null for root
        bool closed;   ///< true if all bits for this node were read
        int value;     ///< current value of te tree (including sum of parents)
    };
    
    XY dim;         ///< size of the tag tree
    Node * nodes;   ///< pointer to allocated nodes
    int nodeCount;  ///< count of currently allocated nodes
    
    /// Finds the value of the node using recursive implementation.
    static int value(Node * node, const int limit, T2Reader & reader);
public:
    /// Standard constructor.
    TagTree();
    
    /// Standard destructor - releases all associated resources.
    ~TagTree();
    
    /// Resizes the tag tree to be able to decode 2D array of codeblocks
    /// @param size  numbers of codeblocks along both axes
    void reset(const XY & size);
    
    /// Gets size of the tag tree.
    /// @return  numbers of codeblocks along both axes.
    const XY & size() const { return dim; }
    
    /// Gets value of the tree at given position, up to given limit (inclusive)
    /// Gets some number greater than limit if the value is greater 
    /// than the limit.
    int value(const XY & pos, const int limit, T2Reader & reader);
    
    /// Gets value of the tree at given position (with no limit)
    int value(const XY & pos, T2Reader & reader);
    
}; // end of class TagTree
    
    
} // end of namespace cuj2kd


#endif // J2KD_TAG_TREE_H

