///
/// @file    t2_cpu_tag_tree.c
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Tag tree implementation for encoding JPEG2000 packet headers.
///

#include "t2_cpu_tag_tree.h"
#include <new>



/// Allocates all nodes, rows and levels.
void t2_cpu_tag_tree_t::alloc(const int level_sx,
                              const int level_sy,
                              const int level_count,
                              const int row_count,
                              const int node_count)
{
    // parameters of this level
    const int level_nodes = level_sx * level_sy;
    const int level_rows = level_sy;

    if(level_nodes > 1)
    {
        // not the last (root) level => let parents allocate all the buffers
        alloc(
            (level_sx + 1) >> 1,
            (level_sy + 1) >> 1,
            level_count + 1,
            row_count + level_rows,
            node_count + level_nodes
        ); 
    }
    else
    {
        // last (root) level => allocate buffers here
        nodes = new node_t [node_count + 1];
        rows = new node_t* [row_count + 1];
        levels = new node_t** [level_count + 1];
        
        // make sure that buffers are allocated
        if(nodes == 0 || rows == 0 || levels == 0)
        {
            throw std::bad_alloc();
        }
    }
    
    // set pointers to all level's rows
    for(int r = level_rows; r--;)
    {
        rows[row_count + r] = nodes + node_count + level_sx * r;
    }
    
    // set pointer to first row of the level
    levels[level_count] = rows + row_count;
}



/// Initializes tag tree for encoding codeblock array up to specified size.
/// @param max_cblks_x  maximal width (in codeblocks) of the tree base
/// @param max_cblks_x  maximal height (in codeblocks) of the tree base
t2_cpu_tag_tree_t::t2_cpu_tag_tree_t(const int max_cblks_x,
                                     const int max_cblks_y)
        : max_size_x(max_cblks_x), max_size_y(max_cblks_y)
{
    // initialize the structure
    levels = 0;
    nodes = 0;
    rows = 0;
    root_level_idx = 0;
       
    // try to allocate buffers and initialize the structure
    alloc(max_cblks_x, max_cblks_y, 0, 0, 0);
}



/// Releases resources associated with given tag tree.
/// @param tree  pointer to some tag tree data
t2_cpu_tag_tree_t::~t2_cpu_tag_tree_t()
{
    if(levels) delete [] levels;
    if(nodes) delete [] nodes;
    if(rows) delete [] rows;
}



/// Finalizes some subtree bounded by given rectangle from given 
/// level up to the root.
void t2_cpu_tag_tree_t::subtree_finalize(const int level_sx,
                                         const int level_sy,
                                         const int level_idx)
{
    if(level_sx * level_sy <= 1)
    {
        // last (root) level => only remember its index
        root_level_idx = level_idx;
        
        // and set root's encoded bits to 0
        levels[level_idx][0][0].ilayer_encoded = 0;
    }
    else
    {
        // not last level
        node_t * const * const level = levels[level_idx];
        node_t * const * const parents = levels[level_idx + 1];
        
        // put minima of all four-tuples into parents
        for(int y = 0; y < level_sy; y++)
        {
            for(int x = 0; x < level_sx; x++)
            {
                // get pointers to the node and its parent
                node_t * const node = &level[y][x];
                node_t * const parent = &parents[y >> 1][x >> 1];
                
                // is this the first child of the parent?
                const bool first_child = (1 & (x | y)) == 0;
                
                // clear number of encoded inclusion layer bits
                node->ilayer_encoded = 0;
                
                // put minima into the parent
                if(first_child || node->ilayer_value < parent->ilayer_value)
                    parent->ilayer_value = node->ilayer_value;
                if(first_child || node->zbplns_diff < parent->zbplns_diff)
                    parent->zbplns_diff = node->zbplns_diff;
            }
        }
        
        // subtract parent values (minima) from all values of this level
        for(int y = 0; y < level_sy; y++)
        {
            for(int x = 0; x < level_sx; x++)
            {
                node_t * const node = &level[y][x];
                const node_t * const parent = &parents[y >> 1][x >> 1];
                node->zbplns_diff -= parent->zbplns_diff;
                node->ilayer_encoded = parent->ilayer_value;
            }
        }
        
        // continue with parent level
        subtree_finalize(
            (level_sx + 1) >> 1,
            (level_sy + 1) >> 1,
            level_idx + 1
        );
    }
}

