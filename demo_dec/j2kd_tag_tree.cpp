///
/// @file    j2kd_tag_tree.cpp
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Implementation of the tag tree for T2 of JPEG 2000 decoder.
///

#include "j2kd_tag_tree.h"
#include "j2kd_error.h"

namespace cuj2kd {



/// Half with rounding up.
inline int halfUp(const int n) {
    return (n + 1) >> 1;
}



/// Gets number of nodes of tag tree with base with given size.
static int treeNodeCount(const XY & size) {
    int count = 1;  // 1 for root
    int sx = size.x;
    int sy = size.y;
    while(sx > 1 || sy > 1) {
        count += sx * sy;
        sx = halfUp(sx);
        sy = halfUp(sy);
    }
    return count;
}



/// Standard tag tree constructor.
TagTree::TagTree() : dim(0, 0), nodes(0), nodeCount(0) {}



/// Standard tag tree destructor - releases all associated resources.
TagTree::~TagTree() {
    if(nodes) {
        delete [] nodes;
    }
}



/// Resizes the tag tree to be able to decode 2D array of codeblocks
/// @param size  numbers of codeblocks along both axes
void TagTree::reset(const XY & size) {
    // minimal required node count
    const int requiredNodeCount = treeNodeCount(size);
    
    // Recreate the tree structure if tree size is different
    if(size != dim) {
        // Reallocate if haven't got enough nodes.
        if(requiredNodeCount > nodeCount) {
            if(nodes) {
                delete [] nodes;
            }
            nodes = new Node [requiredNodeCount];
            nodeCount = requiredNodeCount;
        }
        
        // remember new size
        dim = size;
        
        // initialize nodes
        Node * node = nodes;
        XY levelSize = size;
        while(levelSize.x > 1 || levelSize.y > 1) {
            Node * const parents = node + levelSize.x * levelSize.y;
            const XY parentSize = XY(halfUp(levelSize.x), halfUp(levelSize.y));
            for(int y = 0; y < levelSize.y; y++) {
                const int parentRowOffset = (y >> 1) * parentSize.x;
                for(int x = 0; x < levelSize.x; x++) {
                    node->parent = parents + (x >> 1) + parentRowOffset;
                    node++;
                }
            }
            levelSize = parentSize;
        }
        
        // reinitialize parent of the root node
        node->parent = 0;
    }
    
    // always reinitialize flags of all nodes
    for(int i = requiredNodeCount; i--; ) {
        nodes[i].closed = false;
        nodes[i].value = -1;
    }
    nodes[requiredNodeCount - 1].value = 0;  // root's value is read initially
}



int TagTree::value(Node * node, const int limit, T2Reader & reader) {
    if(-1 == node->value) {
        const int parentValue = value(node->parent, limit, reader);
        if(parentValue == limit) {
            return limit;
        }
        if(node->parent->closed) {
            node->value = parentValue;
        }
    }
    if(!node->closed) {
        node->value += reader.readZeroBits(limit - node->value, node->closed);
    }
    return node->value;
}


/// Gets value of the tree at given position, up to given limit (inclusive)
int TagTree::value(const XY & pos, const int limit, T2Reader & reader) {
    // start recursive value query in the leaf node
    // TODO: try to remove recursion
    return value(nodes + pos.x + pos.y * dim.x, limit, reader);
}



/// Gets value of the tree at given position (with no limit)
int TagTree::value(const XY & pos, T2Reader & reader) {
    return value(pos, 1 << 30, reader);
}



} // end of namespace cuj2kd

