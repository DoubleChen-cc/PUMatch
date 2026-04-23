#include <string>
#include <iostream>
#include "src/gpu_match.cuh"

using namespace std;
using namespace STMatch;

int main(int argc, char* argv[]) {

  cudaSetDevice(0);
  
  //STMatch::GraphPreprocessor g(argv[1],2);
  STMatch::GraphPreprocessor g(argv[1],1);
  STMatch::PatternPreprocessor p(argv[2]);
  //g.build_partial_graph();
  g.build_contiguous_partial_graph(0.6);  
  int partial_on_gpu_id = 0;
  PartialGraph* gpu_partial_graph = g.partial_to_gpu(partial_on_gpu_id);
  // copy graph and pattern to GPU global memory
  Graph* gpu_graph = g.to_gpu();
  size_t graph_memory_usage = sizeof(bitarray32) * g.g.nnodes + sizeof(graph_node_t) * g.g.nedges + sizeof(graph_edge_t) * (g.g.nnodes+1);
  cout<<"graph memory usage: "<<graph_memory_usage<<" B"<<endl;
  size_t partial_graph_memory_usage = sizeof(bitarray32) * g.g.nnodes + sizeof(graph_node_t) * g.p_graph[partial_on_gpu_id].nedges + sizeof(graph_edge_t) * (g.g.nnodes+1);
  cout<<"partial graph memory usage: "<<partial_graph_memory_usage<<" B"<<endl;
  Pattern* gpu_pattern = p.to_gpu();
  JobQueuePreprocessor queue = JobQueuePreprocessor(g.g, p);
  JobQueue* gpu_queue = queue.to_gpu();
  size_t job_memory_usage = sizeof(Job) * queue.q.length;
  cout<<"job queue size: "<<queue.q.length<<", memory usage: "<<job_memory_usage<<" B"<<endl;

  CallStack* gpu_callstack;

  // allocate the callstack for all warps in global memory
  graph_node_t* slot_storage;
  size_t stack_memory_usage = sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE;
  cudaMalloc(&slot_storage, sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
  cout << "stack memory usage: " << stack_memory_usage << " B" << endl;

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

  size_t other_memory_usage = sizeof(CallStack) * NWARPS_TOTAL + sizeof(size_t) * NWARPS_TOTAL + sizeof(int) * GRID_DIM + sizeof(int) + sizeof(int) * GRID_DIM + sizeof(bool) * GRID_DIM + sizeof(Pattern);

  // size_t PUM_memory_usage = (size_t)15 * 1024 * 1024 * 1024;
  size_t UM_memory_usage = (size_t)1 * 1024 * 1024 * 1024;
  // if(PUM_memory_usage > partial_graph_memory_usage){
  //   UM_memory_usage = PUM_memory_usage - partial_graph_memory_usage;
  // }
  cout<<"um memory usage: "<<UM_memory_usage << " B"<<endl;
  // size_t occupy_size = (size_t)40960 * 1024 * 1024 - job_memory_usage - stack_memory_usage - other_memory_usage - UM_memory_usage;
  size_t occupy_size = (size_t)40960 * 1024 * 1024 - job_memory_usage - stack_memory_usage - other_memory_usage - UM_memory_usage-partial_graph_memory_usage;
  // size_t free_mem = 0, total_mem = 0;
  // cudaMemGetInfo(&free_mem, &total_mem);
  // std::cout << "GPU剩余内存: " << free_mem << " B" << std::endl;
  // size_t occupy_size = free_mem- UM_memory_usage;
  cout<<"occupy_size: "<< occupy_size <<" B "<<endl;
  graph_node_t* occupied_space;
  cudaMalloc(&occupied_space, occupy_size);
  cudaMemset(occupied_space,0,occupy_size);
  // Bucket* hash_table;
  // cudaMalloc(&hash_table,sizeof(Bucket)*hash_size);
  // cudaMemset(hash_table,1,sizeof(Bucket)*hash_size);

  //cout << "occupied memory usage: " << OCCUPIED_SIZE << " GB" << endl;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  //cout << "shared memory usage: " << sizeof(Graph) << " " << sizeof(Pattern) << " " << sizeof(JobQueue) << " " << sizeof(CallStack) * NWARPS_PER_BLOCK << " " << NWARPS_PER_BLOCK * 33 * sizeof(int) << " Bytes" << endl;

  _parallel_match<< <GRID_DIM, BLOCK_DIM >> > (gpu_graph, gpu_partial_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res, idle_warps, idle_warps_count, global_mutex);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("matching time: %f ms\n", milliseconds);

  cudaMemcpy(res, gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost);

  unsigned long long tot_count = 0;
  for (int i=0; i<NWARPS_TOTAL; i++) tot_count += res[i];

  if(!LABELED) tot_count = tot_count * p.PatternMultiplicity;
  
  printf("%s\t%f\t%llu\n", argv[2], milliseconds, tot_count);
  FILE* fout = fopen("result.txt", "a");
  if (fout) {
      // argv[2]: 图数据，argv[3]: 模式图（假定按常见顺序参数），milliseconds: 时间，tot_count: 数量
      fprintf(fout, "%s\t%s\t%f\t%llu\n", argv[1], argv[2], milliseconds, tot_count);
      fclose(fout);
  } else {
      fprintf(stderr, "无法打开result.txt进行写入\n");
  }
  //cout << "count: " << tot_count << endl;
  return 0;
}
