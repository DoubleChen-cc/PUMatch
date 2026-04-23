#include "missing_tree.cuh"
#include <cuda.h>


__device__ void initTree(MissingTree* tree) {
    int tid = threadIdx.x;
    if (tid == 0) {
        tree->used = 1;                       
        tree->nodes[0] = MBNode{-1, -1, -1, -1, -1};
        for (int i = 0; i < (MSTREE_SIZE + 31) / 32; ++i) tree->bitmap[i] = 0;
        setBit(tree->bitmap, 0);
        tree->lock = 0;                       
    }
    __syncthreads();
}


__device__ int allocNodeWater(MissingTree* tree) {
    int idx = -1;
    
        int start = tree->used;
        
        if (start < 0) start = 1;  
        if (start >= MSTREE_SIZE) start = 1;
        
        for (int i = start; i < MSTREE_SIZE; ++i) {
            if (!testBit(tree->bitmap, i)) {
                idx = i;
                setBit(tree->bitmap, i);
                tree->used = i + 1;
                break;
            }
        }
        
        if (idx < 0) {
            for (int i = 1; i < start; ++i) {  
                if (!testBit(tree->bitmap, i)) {
                    idx = i;
                    setBit(tree->bitmap, i);
                    tree->used = i + 1;
                    break;
                }
            }
        }
    
    if (idx >= MSTREE_SIZE || idx < 0) return -1;
    return idx;
}


__device__ int allocNodeWarp(MissingTree* tree) {
    const int tid = threadIdx.x & 31;
    int idx = -1;
    uint32_t w = 0xFFFFFFFF;
    const int bitmap_size = (MSTREE_SIZE + 31) / 32;
    if (tid < bitmap_size) w = tree->bitmap[tid];          
    int bit = (w != 0xFFFFFFFF) ? __ffs(~w) - 1 : -1;
    int off = (bit >= 0) ? (tid << 5) + bit : -1;
    
    if (off >= MSTREE_SIZE) off = -1;

    
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        int other = __shfl_down_sync(0xFFFFFFFF, off, mask);
        if (other >= 0 && (off < 0 || other < off)) off = other;
    }
    idx = off;
    
    if (idx >= 0 && idx < MSTREE_SIZE && threadIdx.x == 0) {
        setBit(tree->bitmap, idx);
    }
    return idx;
}





















__device__ int insertNode(MissingTree* tree,
                          int parentIdx,
                          int vid,
                          int depth)
{
    int idx = -1;
    
    
        
        if (parentIdx >= 0 && parentIdx < MSTREE_SIZE) {
            
            int child = tree->nodes[parentIdx].firstChild;
            int loop_count = 0;
            while (child != -1 && loop_count < MSTREE_SIZE) {
                
                if (child < 0 || child >= MSTREE_SIZE) break;
                if (tree->nodes[child].vid == vid) {
                    idx = child;        
                    break;
                }
                child = tree->nodes[child].nextBro;
                loop_count++;
            }


            if (idx < 0) {
                
                const int lane_id = threadIdx.x & 31;
                if(lane_id == 0) 
                {
                    tree_lock(&tree->lock);
                }
                __syncwarp();
                if(lane_id == 0) 
                {
                    idx = allocNodeWater(tree);   
                }
                idx = __shfl_sync(0xFFFFFFFF, idx, 0);
                if (idx >= 0 && idx < MSTREE_SIZE) {
                    
                    if(lane_id == 0) {
                        MBNode* n = &tree->nodes[idx];
                        n->vid   = vid;
                        n->depth = static_cast<int8_t>(depth);
                        n->parent = static_cast<int16_t>(parentIdx);
                        n->nextBro = -1;
                        n->firstChild = -1;

                        
                        int old = tree->nodes[parentIdx].firstChild;
                        
                        if (old != idx) {
                            
                            if (old < 0 || old >= MSTREE_SIZE) old = -1;
                            tree->nodes[parentIdx].firstChild = idx;
                            n->nextBro = static_cast<int16_t>(old);
                        }
                    }
                    __syncwarp();
                }
                if(lane_id == 0) 
                {       
                    tree_unlock(&tree->lock);
                }
                __syncwarp();
            }
        }
    
    idx = __shfl_sync(0xFFFFFFFF, idx, 0);
    return idx;
}


__device__ int insertBranch(MissingTree* tree,
                            CallStack* stk,
                            Pattern* pat,
                            int level,  
                            int k,        
                            int currVid)        
{
    
    int prevNode = 0;                            
    
    for (int l = -1; l < level; ++l) {
        int vid = path(stk, pat, l, k);          
        prevNode = insertNode(tree, prevNode, vid, l);
        if (prevNode < 0) return -1;             
    }

    
    return insertNode(tree, prevNode, currVid, level);
}



__device__ int longestMatch(const MissingTree* tree,
                            int depth,
                            CallStack* stk, 
                            Pattern* pat, 
                            int k)
{
    int result = -1;
    const int lane_id = threadIdx.x & 31;
    
    
    
    int node = 0;
    int match = 0;
    int dSeq[5];
    #pragma unroll
    for (int l = 0; l-1 < depth; ++l) {
        dSeq[l] = path(stk, pat, l-1, k);   
    }
    if(lane_id == 0) 
    {
        tree_lock((int*)&tree->lock);
    }
    __syncwarp();
    
    for (int lvl = 0; lvl < depth+1; ++lvl) {
        int want = dSeq[lvl];
        int child = tree->nodes[node].firstChild;
        bool found = false;
        int loop_count = 0;
        while (child != -1 && loop_count < MSTREE_SIZE) {
            
            if (child < 0 || child >= MSTREE_SIZE) break;
            const MBNode* c = &tree->nodes[child];
            if (c->vid == want) {
                node  = child;
                found = true;
                ++match;
                break;
            }
            int next = c->nextBro;
            
            if (next < 0 || next >= MSTREE_SIZE) break;
            child = next;
            loop_count++;
        }
        if (!found) break;
    }
    if(match == depth + 1){
        result = tree->nodes[node].firstChild;
    }else{
        result = -1;
    }
    if(lane_id == 0) 
    {
        tree_unlock((int*)&tree->lock);
    }
    __syncwarp();
    
    result = __shfl_sync(0xFFFFFFFF, result, 0);
    return result;
}


__device__ void freeNode(MissingTree* tree, int idx) {
    clearBit(tree->bitmap, idx);
}


__device__ int deleteBranch(MissingTree* tree, int leafIdx)
{
    int freed = 0;
    const int lane_id = threadIdx.x & 31;
    
    if (leafIdx <= 0) {  
        freed = 0;
    } else if (leafIdx >= MSTREE_SIZE) {  
        freed = 0;
    } else {
        if(lane_id == 0) 
        {
            tree_lock(&tree->lock);
        }
        __syncwarp();
        
        
        if(lane_id == 0) {
            int node = leafIdx;
            int max_iterations = MSTREE_SIZE;  
            int iteration_count = 0;
            
            int visited_nodes[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
            int visited_count = 0;
            
            while (node > 0 && node < MSTREE_SIZE && iteration_count < max_iterations) {
                iteration_count++;
                MBNode* n = &tree->nodes[node];
                int parent = n->parent;
                
                
                if (parent < 0 || parent >= MSTREE_SIZE) break;  
                
                
                bool found_loop = false;
                for (int i = 0; i < visited_count && i < 8; i++) {
                    if (visited_nodes[i] == parent) {
                        
                        found_loop = true;
                        
                        n->parent = -1;
                        break;
                    }
                }
                if (found_loop) {
                    
                    clearBit(tree->bitmap, node);
                    ++freed;
                    break;
                }
                
                
                if (visited_count < 8) {
                    visited_nodes[visited_count++] = node;
                } else {
                    
                    for (int i = 0; i < 7; i++) {
                        visited_nodes[i] = visited_nodes[i + 1];
                    }
                    visited_nodes[7] = node;
                }
                
                
                int head = tree->nodes[parent].firstChild;
                int  curr = head;
                int  prev = -1;
                int loop_count = 0;  
                while (curr != -1 && loop_count < MSTREE_SIZE) {  
                    if (curr < 0 || curr >= MSTREE_SIZE) break;  
                    if (curr == node) {
                        int next = tree->nodes[curr].nextBro;
                        
                        if (next < 0 || next >= MSTREE_SIZE) next = -1;
                        if (prev == -1)
                            tree->nodes[parent].firstChild = next;
                        else if (prev >= 0 && prev < MSTREE_SIZE)
                            tree->nodes[prev].nextBro = next;
                        break;
                    }
                    prev = curr;
                    int next = tree->nodes[curr].nextBro;
                    
                    if (next < 0 || next >= MSTREE_SIZE) break;
                    curr = next;
                    loop_count++;
                }

                
                clearBit(tree->bitmap, node);
                ++freed;

                
                if (tree->nodes[parent].firstChild != -1) break;   
                node = parent;            
            }
        }
        __syncwarp();
        
        if(lane_id == 0) 
        {
            tree_unlock(&tree->lock);
        }
        __syncwarp();
        
        
        freed = __shfl_sync(0xFFFFFFFF, freed, 0);
    }
    return freed;
}