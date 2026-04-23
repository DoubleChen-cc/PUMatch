#pragma once
#include <cuda.h>
namespace STMatch {

  typedef int graph_node_t;
  typedef long graph_edge_t;
  typedef char pattern_node_t;
  typedef char set_op_t;
  typedef unsigned int bitarray32;
  typedef uint64_t hash_t;

  inline constexpr size_t PAT_SIZE = 7;
  inline constexpr size_t GRAPH_DEGREE = 4096;
  inline constexpr size_t MAX_SLOT_NUM = 15;

#include "config_for_ae/fig_local_global_unroll.h" 

  inline constexpr int GRID_DIM = 82;
  inline constexpr int BLOCK_DIM = 256;
  inline constexpr int WARP_SIZE = 32;
  inline constexpr int NWARPS_PER_BLOCK = (BLOCK_DIM / WARP_SIZE);
  inline constexpr int NWARPS_TOTAL = ((GRID_DIM * BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE);

  inline constexpr graph_node_t JOB_CHUNK_SIZE = 8;
  inline constexpr unsigned HASH_TABLE_SIZE=0X28000000; //40gb
  inline constexpr int BUCKET_SIZE=8;
  //static_assert(2 * JOB_CHUNK_SIZE <= GRAPH_DEGREE); 

  // this is the maximum unroll size

  inline constexpr int DETECT_LEVEL = 1;
  inline constexpr int STOP_LEVEL = 2;
  static void check_cuda(const cudaError_t e, const char* file,
                             const int line) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s (%d)\n", file, line, cudaGetErrorString(e), e);
    exit(1);
  }
}
#define check_cuda_error(x) check_cuda(x, __FILE__, __LINE__)
}
