#pragma once
#include <omp.h>

#include "graph.h"

namespace STMatch {

  typedef struct {
    graph_node_t nodes[PAT_SIZE];
  } Job;

  typedef struct {
    Job* q;
    int start_level;
    graph_edge_t length;
    graph_node_t cur;
    int mutex = 0;
  } JobQueue;

  struct JobQueuePreprocessor {

    JobQueue q;


    JobQueuePreprocessor(Graph& g, PatternPreprocessor& p) {
      std::vector<graph_node_t> vr, vc;
      // int num_threads = 8;
      // std::vector<std::vector<graph_node_t>> local_vr(num_threads);
      // std::vector<std::vector<graph_node_t>> local_vc(num_threads);
       //#pragma omp parallel num_threads(num_threads)
      // {
      //   int thread_id = omp_get_thread_num();
      //   std::vector<graph_node_t>& my_vr = local_vr[thread_id];
      //   std::vector<graph_node_t>& my_vc = local_vc[thread_id];
      // #pragma omp parallel for schedule(dynamic, 2)
      for (graph_node_t r = 0; r < g.nnodes; r++) {
        for (graph_edge_t j = g.rowptr[r]; j < g.rowptr[r + 1]; j++) {
          graph_node_t c = g.colidx[j];
          if ((!LABELED && p.partial[0][0] == 1 && r > c) || LABELED || p.partial[0][0] != 1) {
            if ((g.vertex_label[r] == (1 << p.vertex_labels[0])) && (g.vertex_label[c] == (1 << p.vertex_labels[1]))) {
              if (g.rowptr[r + 1] - g.rowptr[r] >= p.pat.degree[0] && g.rowptr[c + 1] - g.rowptr[c] >= p.pat.degree[1]) {
                bool valid = false;
                for (graph_edge_t d = g.rowptr[c]; d < g.rowptr[c + 1]; d++) {
                  graph_node_t v = g.colidx[d];
                  if (g.rowptr[v + 1] - g.rowptr[v] >= p.pat.degree[2]) {
                    valid = true;
                    break;
                  }
                }

                if (valid) {
                  vr.push_back(r);
                  vc.push_back(c);
                }
              }
            }
          }
        }
      }
    //}
    // std::cout<<"paralell done!"<<std::endl;
    //     // 合并所有线程的结果
    // for (int i = 0; i < num_threads; i++) {
    //     vr.insert(vr.end(), local_vr[i].begin(), local_vr[i].end());
    //     vc.insert(vc.end(), local_vc[i].begin(), local_vc[i].end());
    // }
    
      q.q = new Job[vr.size()];
      for (graph_node_t i = 0; i < vr.size(); i++) {
        (q.q[i].nodes)[0] = vr[i];
        (q.q[i].nodes)[1] = vc[i];
      }
      q.length = vr.size();
      if(q.length>100000000)q.length = vr.size()/1000;
      else if(q.length>10000000)q.length = vr.size()/100;
      q.cur = 0;
      q.start_level = 2;

    }

    JobQueue* to_gpu() {
      JobQueue qcopy = q;
      cudaMalloc(&qcopy.q, sizeof(Job) * q.length);
      cudaMemcpy(qcopy.q, q.q, sizeof(Job) * q.length, cudaMemcpyHostToDevice);

      JobQueue* gpu_q;
      cudaMalloc(&gpu_q, sizeof(JobQueue));
      cudaMemcpy(gpu_q, &qcopy, sizeof(JobQueue), cudaMemcpyHostToDevice);
      return gpu_q;
    }
//     JobQueue* to_gpu() {
//     // 分配统一内存存储JobQueue结构
//     JobQueue* gpu_q;
//     check_cuda_error(cudaMallocManaged(&gpu_q, sizeof(JobQueue)));
    
//     // 复制原始队列数据到新结构
//     *gpu_q = q;
    
//     // 为队列数据分配统一内存并复制内容
//     check_cuda_error(cudaMallocManaged(&gpu_q->q, sizeof(Job) * q.length));
//     memcpy(gpu_q->q, q.q, sizeof(Job) * q.length);
    
//     // 设置内存建议（可选）
//     int device = 0; // 假设使用GPU 0
//     cudaMemAdvise(gpu_q->q, sizeof(Job) * q.length, cudaMemAdviseSetReadMostly, device);
    
//     // 预取数据到GPU（可选优化）
//     //cudaMemPrefetchAsync(gpu_q->q, sizeof(Job) * q.length, device);
    
//     return gpu_q;
// }
  };
}