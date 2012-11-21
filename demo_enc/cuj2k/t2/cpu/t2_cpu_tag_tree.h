///
/// @file    t2_cpu_tag_tree.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Tag tree structure for encoding JPEG2000 packet headers.
///


#ifndef T2_TAG_TREE_H
#define T2_TAG_TREE_H


#include "t2_cpu_output.h"



/// Tag tree for encoding packet headers.
class t2_cpu_tag_tree_t {
private:
    /// Node of the tag tree.
    struct node_t {
        // inclusion layer value of the node
        unsigned char ilayer_value;
        
        // number of inclusion 0 bits already encoded
        unsigned char ilayer_encoded;
        
        // number of zero-bitplanes-count bits to be encoded for this node
        unsigned char zbplns_diff;
    }; // end of struct t2_cpu_tag_tree_node_t
    
    /// maximal width of the base (in codeblocks) of tree
    const int max_size_x;
    
    /// maximal height of the base (in codeblocks) of tree
    const int max_size_y;
    
    /// index of current root level (depends of size of encoded rectangle)
    int root_level_idx;
    
    /// pointer to array of all nodes of the tree
    node_t * nodes;
    
    /// pointers to first nodes of all rows of the tree
    node_t ** rows;
    
    /// Array of pointers to first rows of some all levels
    node_t *** levels;
    
    /// Allocates all nodes, rows and levels
    void alloc(
        const int level_sx,
        const int level_sy,
        const int level_count,
        const int row_count,
        const int node_count
    );
    
    /// Finalizes some subtree bounded by given rectangle from given 
    /// level up to the root.
    void subtree_finalize(
        const int level_sx,
        const int level_sy,
        const int level_idx
    );
    
    /// Gets correponding node at given level.
    node_t * get_node(const int node_x, const int node_y, const int level)
    {
        return &levels[level][node_y >> level][node_x >> level];
    }
    
public:
    /// Initializes tag tree for encoding codeblock array up to specified size.
    /// @param max_cblks_x  maximal width (in codeblocks) of the tree base
    /// @param max_cblks_x  maximal height (in codeblocks) of the tree base
    t2_cpu_tag_tree_t(const int max_cblks_x, const int max_cblks_y);
    
    
    /// Releases resources associated with the tag tree.
    ~t2_cpu_tag_tree_t();
    
    
    /// Sets attribute of some leaf of the tag tree. All nodes in some rectangle 
    /// must be initialized this way before actual encoding. Size of this rectangle
    /// is given to method 'finalize'.
    /// @param cblk_x  x-axis index of the leaf node to be set
    /// @param cblk_y  y-axis index of the leaf node to be set
    /// @param ilayer  value to be assigned to the node
    /// @param zbplns  number of zero most significant bitplanes
    void set(
        const int cblk_x,
        const int cblk_y,
        const unsigned char ilayer,
        const unsigned char zbplns)
    {
        // just find the node in level #0 and set the value
        node_t * const node = &levels[0][cblk_y][cblk_x];
        node->ilayer_value = ilayer;
        node->zbplns_diff = zbplns;
    }
        
    
    /// Gets inclusion layer of given leaf.
    /// @param cblk_x   x-axis index of the leaf
    /// @param cblk_y   y-axis index of the leaf
    /// @return inclusion layer value for given leaf
    unsigned char get_ilayer(const int cblk_x, const int cblk_y) const
    {
        return levels[0][cblk_y][cblk_x].ilayer_value;
    }
    
    
    /// Finalize (reduce) initialization of some rectangle of nodes.
    /// @param num_cblks_x  width of initialized rectangle
    /// @param num_cblks_y  height of initialized rectangle
    void finalize(const int num_cblks_x, const int num_cblks_y)
    {
        subtree_finalize(num_cblks_x, num_cblks_y, 0);
    }
    
    
    /// Encodes inclusion information of specified node of the tag tree.
    /// @param cblk_x   x-axis index of the leaf to be encoded
    /// @param cblk_y   y-axis index of the leaf to be encoded
    /// @param limit    maximal number of layers to be encoded
    /// @param encoder  object for encoding bits of packet headers
    void encode_ilayer(
        const int cblk_x,
        const int cblk_y,
        const int limit,
        t2_cpu_output_t * const encoder)
    {
        // go down from the root to find first not-fully-encoded node
        int level = root_level_idx;
        node_t * node = get_node(cblk_x, cblk_y, level);
        while(node->ilayer_encoded > node->ilayer_value && level)
        {
            node = get_node(cblk_x, cblk_y, --level);
        }
        
        // encode bits up to given limit
        while(node->ilayer_encoded <= limit)
        {
            // encode another bit
            if(node->ilayer_encoded++ == node->ilayer_value)
            {
                // encoded all node's bits => terminate level with 1 bit ...
                encoder->put_one();
                
                // ... and jump up to child level:
                if(level--)
                {
                    node = get_node(cblk_x, cblk_y, level);
                }
                else
                {
                    break; // leaf encoded => no more levels
                }
            }
            else
            {
                // not all node's bits encoded => put 0 bit and stay in this level
                encoder->put_zero();
            }
        }
    }
    
    
    /// Encodes zero bitplanes information of specified node of the tag tree.
    /// @param cblk_x   x-axis index of the leaf to be encoded
    /// @param cblk_y   y-axis index of the leaf to be encoded
    /// @param encoder  object for encoding bits of packet headers
    void encode_zbplns(
        const int cblk_x,
        const int cblk_y,
        t2_cpu_output_t * const encoder)
    {
        // go down from the root to find first not-yet-encoded node
        // (assumes that each leaf gets encoded at most once)
        int level = root_level_idx;
        node_t * node = get_node(cblk_x, cblk_y, level);
        while(node->zbplns_diff == 0xFF)
        {
            node = get_node(cblk_x, cblk_y, --level);
        }
        
        // encode all bits of all levels up to the leaf
        while(1)
        {
            // express the level value with corresponding number of 0 bits
            encoder->put_zeros(node->zbplns_diff);
            
            // terminate the level with single 1 bit
            encoder->put_one();
            
            // set zero bitplanes information of the node to 'already encoded'
            node->zbplns_diff = 0xFF;
            
            // advance to next level
            if(level--)
            {
                // get pointer to corrreponding child node in upper level
                node = get_node(cblk_x, cblk_y, level);
            }
            else
            {
                //  this was the last level => done
                break;
            }
        }
    }
    
}; // end of class t2_cpu_tag_tree_t



#endif // T2_TAG_TREE_H

