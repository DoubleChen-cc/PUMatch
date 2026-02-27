#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include "src/buffer.cuh"
#include "src/graph.h"
#include "src/pattern.h"
#include "src/job_queue.h"
#include "src/missing_tree.cuh"
#include "src/gpu_match.cuh"
#include <thread>

int main(int argc, char* argv[]) {

  cudaSetDevice(0);

  GraphPreprocessor g(argv[1]);
  GraphPreprocessor g2(argv[1]);
  PatternPreprocessor p(argv[2]);
  float ratio = 0.6;
  graph_edge_t edge_limit_num = g2.g.nedges * ratio;
  std::cout<<"edge_limit_num:"<<edge_limit_num<<std::endl;
      
  int idx = -1;
  for (int i = 0; i < g2.g.nnodes + 1; ++i) {
    if (g2.g.rowptr[i] > edge_limit_num) {
      idx = i;
      break;
    }
  }
      
      
  
  
    int        n      = g2.g.nnodes + 1;
    int        half   = idx;                       
    auto*      ptr    = g2.g.rowptr;                  
    graph_edge_t last = ptr[half - 1];               

    
    std::fill(ptr + half, ptr + n, last);
  
  
  
  Graph* gpu_graph = g2.to_gpu();
  Graph* gpu_graph_managed = g.to_gpu_managed();
  Pattern* gpu_pattern = p.to_gpu();
  JobQueuePreprocessor queue = JobQueuePreprocessor(g.g, p);
  JobQueue* gpu_queue = queue.to_gpu();
  CallStack* gpu_callstack;
  size_t partial_graph_memory_usage = sizeof(bitarray32) * g.g.nnodes + sizeof(graph_node_t) * g2.g.rowptr[g.g.nnodes] + sizeof(graph_edge_t) * (g.g.nnodes+1);
  std::cout<<"partial graph memory usage: "<<partial_graph_memory_usage<<" B"<<std::endl;
  size_t job_memory_usage = sizeof(Job) * queue.q.length;
  std::cout<<"job queue size: "<<queue.q.length<<", memory usage: "<<job_memory_usage<<" B"<<std::endl;

  
  graph_node_t* slot_storage;
  cudaMalloc(&slot_storage, sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
  
  size_t stack_memory_usage = sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE;
  std::cout << "stack memory usage: " << stack_memory_usage << " B" <<std::endl;

  std::vector<CallStack> stk(NWARPS_TOTAL);

  for (int i = 0; i < NWARPS_TOTAL; i++) {
    auto& s = stk[i];
    memset(s.iter, 0, sizeof(s.iter));
    memset(s.slot_size, 0, sizeof(s.slot_size));
    s.slot_storage = (graph_node_t(*)[UNROLL][GRAPH_DEGREE])((char*)slot_storage + i * sizeof(graph_node_t) * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
  }
  cudaMalloc(&gpu_callstack, NWARPS_TOTAL * sizeof(CallStack));
  cudaMemcpy(gpu_callstack, stk.data(), sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice);

  size_t* gpu_res;
  cudaMalloc(&gpu_res, sizeof(size_t) * NWARPS_TOTAL);
  cudaMemset(gpu_res, 0, sizeof(size_t) * NWARPS_TOTAL);
  size_t* res = new size_t[NWARPS_TOTAL];

  int* idle_warps;
  cudaMalloc(&idle_warps, sizeof(int) * GRID_DIM);
  cudaMemset(idle_warps, 0, sizeof(int) * GRID_DIM);

  int* idle_warps_count;
  cudaMalloc(&idle_warps_count, sizeof(int));
  cudaMemset(idle_warps_count, 0, sizeof(int));

  int* global_mutex;
  cudaMalloc(&global_mutex, sizeof(int) * GRID_DIM);
  cudaMemset(global_mutex, 0, sizeof(int) * GRID_DIM);

  bool* stk_valid;
  cudaMalloc(&stk_valid, sizeof(bool) * GRID_DIM);
  cudaMemset(stk_valid, 0, sizeof(bool) * GRID_DIM);

  
    
    
    
  
  int* write_block_flags;
  cudaMalloc(&write_block_flags, sizeof(int) * GRID_DIM);
  cudaMemset(write_block_flags, 0, sizeof(int) * GRID_DIM);
  size_t buffer_memory_usage = MAX_PACKAGE_EACH_BUFFER * MAX_PACKAGE_SIZE * sizeof(int);
  std::cout<<"buffer memory usage:"<<buffer_memory_usage<<"B"<<std::endl;
  size_t other_memory_usage = sizeof(CallStack) * NWARPS_TOTAL + sizeof(size_t) * NWARPS_TOTAL + sizeof(int) * GRID_DIM + sizeof(int) + sizeof(int) * GRID_DIM + sizeof(bool) * GRID_DIM + sizeof(Pattern);
  size_t UM_memory_usage = (size_t)1 * 1024 * 1024 * 1024;

  
  int* cpu_visible_queue_raw;
  cudaHostAlloc((void**)&cpu_visible_queue_raw, GRID_DIM * CHUNKS_PER_BLOCK * (MAX_QUEUE_SIZE/2) * sizeof(int), cudaHostAllocMapped);
  int (*cpu_visible_queue)[CHUNKS_PER_BLOCK][MAX_QUEUE_SIZE/2] = 
      (int (*)[CHUNKS_PER_BLOCK][MAX_QUEUE_SIZE/2])cpu_visible_queue_raw;

    SyncFlags2* d_flags;
    cudaMallocManaged(&d_flags, sizeof(SyncFlags2), cudaMemAttachGlobal);
    
    d_flags->matching_over = 0;
    d_flags->cpu_process_over = 0;
    d_flags->global_update_token = 0;

    for (int b = 0; b < GRID_DIM; ++b) {
      d_flags->block_update_token[b] = 0;
      d_flags->process_vertices_flag[b] = 0;
      for(int i = 0;i<CHUNKS_PER_BLOCK;++i){
        d_flags->queue_flushed_flags[b][i] = 0;
      }
    }

    
    Buffer host_buffer = create_buffer_from_graph(g.g, idx-1);
    
    
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU剩余内存: " << free_mem << " B" << std::endl;
    
    size_t occupy_size = (size_t)40960 * 1024 * 1024 - job_memory_usage - stack_memory_usage - other_memory_usage - UM_memory_usage-partial_graph_memory_usage - buffer_memory_usage;
    std::cout<<"occupy_size: "<< occupy_size <<" B "<<std::endl;


    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaEvent_t init_done;
    cudaEventCreate(&init_done);

    
    Buffer* d_buffer = buffer_init_on_gpu(host_buffer, stream1);   

    graph_node_t* occupied_space;
    cudaMalloc(&occupied_space, occupy_size);
    cudaMemset(occupied_space,0,occupy_size);

    
    cudaEventRecord(init_done, stream1);

    
    std::vector<std::thread> cpu_threads;
    for (int tid = 0; tid < GRID_DIM; ++tid) {
        cpu_threads.emplace_back([&, tid] {
            
            cudaEventSynchronize(init_done);
            cpu_process_multiple_queue(g, &host_buffer, d_buffer, cpu_visible_queue, CHUNKS_PER_BLOCK, 
            d_flags, tid, idx-1, stream2);
            
        });
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream1);

    
    _parallel_match<<<GRID_DIM, BLOCK_DIM, 0, stream1>>>(
        gpu_graph, gpu_graph_managed, gpu_pattern, gpu_callstack, gpu_queue, gpu_res,
        idle_warps, idle_warps_count, global_mutex,
        d_buffer, d_flags,write_block_flags, (int*)cpu_visible_queue, idx-1);

    
    cudaEventRecord(stop, stream1);
    cudaEventSynchronize(stop);
    d_flags->matching_over = 1;
    
    for (auto& th : cpu_threads) th.join();

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  

  cudaMemcpy(res, gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost);

  unsigned long long tot_count = 0;
  for (int i=0; i<NWARPS_TOTAL; i++) tot_count += res[i];

  if(!LABELED) tot_count = tot_count * p.PatternMultiplicity;
  
  std::cout<<milliseconds<<' '<<tot_count<<std::endl;

  return 0;
}
