#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "callstack.h"
#include "common.cuh"
#include "config.h"


struct MBNode {
    int32_t vid;        
    int16_t  depth;      
    int16_t parent;     
    int16_t nextBro;    
    int16_t firstChild; 
    
};



struct MissingTree {
    MBNode  nodes[MSTREE_SIZE];   
    uint32_t bitmap[(MSTREE_SIZE + 31) / 32];   
    int      used;         
    int      lock;         
};


__device__ __forceinline__ void tree_lock(int* lock) {
    while (atomicCAS((int*)lock, 0, 1) != 0) {
        
    }
}

__device__ __forceinline__ void tree_unlock(int* lock) {
    atomicExch((int*)lock, 0);
}


__device__ __forceinline__ bool testBit(const uint32_t* bmp, int i) {
    return (bmp[i >> 5] & (1U << (i & 31))) != 0;
}
__device__ __forceinline__ void setBit(uint32_t* bmp, int i) {
    atomicOr(&bmp[i >> 5], 1U << (i & 31));
}
__device__ __forceinline__ void clearBit(uint32_t* bmp, int i) {
    atomicAnd(&bmp[i >> 5], ~(1U << (i & 31)));
}


__device__ void initTree(MissingTree* tree);


__device__ int allocNodeWater(MissingTree* tree);


__device__ int allocNodeWarp(MissingTree* tree);





















__device__ int insertNode(MissingTree* tree,
                          int parentIdx,
                          int vid,
                          int depth);



__device__ int insertBranch(MissingTree* tree,
                            CallStack* stk,
                            Pattern* pat,
                            int level,  
                            int k,        
                            int currVid);       

__device__ int longestMatch(const MissingTree* tree,
                            int depth,
                            CallStack* stk, 
                            Pattern* pat, 
                            int k);

__device__ void freeNode(MissingTree* tree, int idx);

__device__ int deleteBranch(MissingTree* tree, int leafIdx);