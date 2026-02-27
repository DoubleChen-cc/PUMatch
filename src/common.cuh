  #pragma once
  #include "config.h"
  #include "callstack.h"
  #include "pattern.h"
  
  __forceinline__ __device__ graph_node_t path(CallStack* stk, Pattern* pat, int level, int k) {
    if (level > 0)
      return stk->slot_storage[pat->rowptr[level]][stk->uiter[level]][stk->iter[level] + k];
    else {
      return stk->slot_storage[0][stk->uiter[0]][stk->iter[0] + k + (level + 1) * JOB_CHUNK_SIZE];
    }
  }

  __forceinline__ __device__ graph_node_t* path_address(CallStack* stk, Pattern* pat, int level, int k) {
    if (level > 0)
      return &(stk->slot_storage[pat->rowptr[level]][stk->uiter[level]][stk->iter[level] + k]);
    else {
      return &(stk->slot_storage[0][stk->uiter[0]][stk->iter[0] + k + (level + 1) * JOB_CHUNK_SIZE]);
    }
  }