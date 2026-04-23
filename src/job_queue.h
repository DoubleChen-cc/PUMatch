#pragma once

#include <cuda_runtime.h>

#include "graph.h"

typedef struct {
graph_node_t nodes[2];
} Job;

typedef struct {
Job* q;
int start_level;
graph_node_t length;
graph_node_t cur;
int mutex = 0;
graph_node_t work_lo;
graph_node_t work_hi;
graph_node_t preempt_at;
int preempt_latched;
int* preempt_flag;

  // Double-buffered task intervals for persistent kernel:
  // - current interval: [work_lo, work_hi)
  // - next interval:    [next_lo, next_hi) (ready when next_ready==1)
  graph_node_t next_lo;
  graph_node_t next_hi;
  graph_node_t next_preempt_at;
  int next_ready;

  // Set to 1 by host when no more intervals will be provided.
  int done;
} JobQueue;

struct JobQueuePreprocessor {

JobQueue q;

JobQueuePreprocessor(Graph& g, PatternPreprocessor& p) {
      std::vector<graph_node_t> vr, vc;
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

      q.q = new Job[vr.size()];
      for (graph_node_t i = 0; i < vr.size(); i++) {
        (q.q[i].nodes)[0] = vr[i];
        (q.q[i].nodes)[1] = vc[i];
      }
      q.length = vr.size();
      std::cout<<"vr.size(): "<<vr.size()<<std::endl;
      
      
      q.cur = 0;
      q.start_level = 2;
      q.work_lo = 0;
      q.work_hi = q.length;
      q.preempt_at = q.length;
      q.preempt_latched = 0;
      q.preempt_flag = nullptr;

      q.next_lo = 0;
      q.next_hi = 0;
      q.next_preempt_at = 0;
      q.next_ready = 0;
      q.done = 0;

    }

    JobQueue* to_gpu() {
      JobQueue qcopy = q;
      qcopy.q = nullptr;

      if (q.length > 0) {
        cudaError_t err = cudaMalloc(&qcopy.q, sizeof(Job) * q.length);
        if (err != cudaSuccess) {
          std::cerr << "JobQueue to_gpu: cudaMalloc(qcopy.q) failed, length=" << q.length
                    << " err=" << cudaGetErrorString(err) << "\n";
          return nullptr;
        }
        err = cudaMemcpy(qcopy.q, q.q, sizeof(Job) * q.length, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
          std::cerr << "JobQueue to_gpu: cudaMemcpy(qcopy.q) failed, length=" << q.length
                    << " err=" << cudaGetErrorString(err) << "\n";
          cudaFree(qcopy.q);
          qcopy.q = nullptr;
          return nullptr;
        }
      }

      JobQueue* gpu_q = nullptr;
      cudaError_t err = cudaMalloc(&gpu_q, sizeof(JobQueue));
      if (err != cudaSuccess) {
        std::cerr << "JobQueue to_gpu: cudaMalloc(gpu_q) failed err=" << cudaGetErrorString(err) << "\n";
        return nullptr;
      }
      err = cudaMemcpy(gpu_q, &qcopy, sizeof(JobQueue), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        std::cerr << "JobQueue to_gpu: cudaMemcpy(gpu_q) failed err=" << cudaGetErrorString(err) << "\n";
        cudaFree(gpu_q);
        return nullptr;
      }
      return gpu_q;
    }



































































    






















};
