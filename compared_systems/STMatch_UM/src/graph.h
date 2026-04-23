#pragma once

#include <cstddef>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cassert>
#include "config.h"


namespace STMatch {

  typedef struct {

    graph_node_t nnodes = 0;
    graph_edge_t nedges = 0;
    bitarray32* vertex_label;
    graph_edge_t* rowptr;
    graph_node_t* colidx;
    int* partition_id;
    int partition_num;
  } Graph;

typedef struct {
    graph_node_t nnodes = 0;
    graph_edge_t nedges = 0;
    bitarray32* vertex_label;
    graph_edge_t* rowptr;
    graph_node_t* colidx;
  } PartialGraph;


  struct GraphPreprocessor {

    Graph g;
    PartialGraph* p_graph;
    

    GraphPreprocessor(std::string filename, int p_num) {
      g.partition_num = p_num;
      readfile(filename);
    }

    void build_partial_graph(){
      int partition_num = g.partition_num;
      p_graph = new PartialGraph[partition_num];
        //构造部分图的rowptr和colidx，rowptr包含所有顶点
      int vnum = g.nnodes;
      std::vector<std::vector<int>> degree_list(partition_num, std::vector<int>(vnum, 0));
        //算每个分区每个顶点的度

      for(graph_node_t i = 0;i < vnum; i++){
          int p = g.partition_id[i];
          int degree =  g.rowptr[i+1] - g.rowptr[i];
          degree_list[p][i] = degree;
      }
      for(int p = 0; p < partition_num; p++){
          p_graph[p].nnodes = vnum;
          p_graph[p].rowptr = new graph_edge_t[vnum+1];
          p_graph[p].rowptr[0] = 0;
          for(int i=0; i<vnum; i++){
              p_graph[p].rowptr[i+1] = p_graph[p].rowptr[i]+degree_list[p][i];
          }
          p_graph[p].nedges = p_graph[p].rowptr[vnum];
          p_graph[p].colidx = new graph_node_t[p_graph[p].nedges];
          
          for(int i=0; i<vnum; i++){
              if(degree_list[p][i]!=0){
                  graph_edge_t src = g.rowptr[i];
                  graph_edge_t dst = p_graph[p].rowptr[i];
                  for(int j = 0; j < degree_list[p][i]; j++){
                      p_graph[p].colidx[dst+j] = g.colidx[src+j];
                  }
              }
          }
          p_graph[p].vertex_label = new bitarray32[vnum];
          for(int i = 0; i < vnum; i++){
            p_graph[p].vertex_label[i] = g.vertex_label[i];
          }
      }
    }

    void build_contiguous_partial_graph(float ratio){
      p_graph = new PartialGraph[1];
      p_graph[0].nnodes = g.nnodes;
      p_graph[0].rowptr = new graph_edge_t[g.nnodes + 1];
      p_graph[0].rowptr[0] = 0;
      graph_edge_t edge_limit_num = g.nedges * ratio;
      graph_node_t idx = -1;
      for (graph_node_t i = 0; i < g.nnodes + 1; ++i) {
        if (g.rowptr[i] > edge_limit_num) {
          idx = i;
          break;
        }
      }
      for(graph_node_t i = 0; i < g.nnodes; i++){
        if(i<idx){
          p_graph[0].rowptr[i+1] = p_graph[0].rowptr[i] + g.rowptr[i+1] - g.rowptr[i];
        }else{
          p_graph[0].rowptr[i+1] = p_graph[0].rowptr[i];
        }
      }
      p_graph[0].nedges = p_graph[0].rowptr[g.nnodes];
      p_graph[0].colidx = new graph_node_t[p_graph[0].nedges];
      for(graph_node_t i = 0; i<idx; i++){
        for(graph_edge_t j = g.rowptr[i]; j < g.rowptr[i+1]; j++){
          p_graph[0].colidx[j] = g.colidx[j];
        }
      }
      p_graph[0].vertex_label = new bitarray32[g.nnodes];
      for(int i = 0; i < g.nnodes; i++){
        p_graph[0].vertex_label[i] = g.vertex_label[i];
      }
    }

    PartialGraph* partial_to_gpu(int pid){
      PartialGraph p_graph_copy = p_graph[pid];
      cudaMalloc(&p_graph_copy.vertex_label, sizeof(bitarray32) * p_graph[pid].nnodes);
      cudaMalloc(&p_graph_copy.rowptr, sizeof(graph_edge_t) * (p_graph[pid].nnodes + 1));
      cudaMalloc(&p_graph_copy.colidx, sizeof(graph_node_t) * p_graph[pid].nedges);
      cudaMemcpy(p_graph_copy.vertex_label, p_graph[pid].vertex_label, sizeof(bitarray32) * p_graph[pid].nnodes, cudaMemcpyHostToDevice);
      cudaMemcpy(p_graph_copy.rowptr, p_graph[pid].rowptr, sizeof(graph_edge_t) * (p_graph[pid].nnodes + 1), cudaMemcpyHostToDevice);
      cudaMemcpy(p_graph_copy.colidx, p_graph[pid].colidx, sizeof(graph_node_t) * p_graph[pid].nedges, cudaMemcpyHostToDevice);

      PartialGraph* gpu_pg;
      cudaMalloc(&gpu_pg, sizeof(PartialGraph));
      cudaMemcpy(gpu_pg, &p_graph_copy, sizeof(PartialGraph), cudaMemcpyHostToDevice);
      // check_cuda_error(cudaMallocManaged(&gpu_g, sizeof(Graph)));
      // cudaMemAdvise(gpu_g, sizeof(Graph), cudaMemAdviseSetReadMostly, 0);
      return gpu_pg;
    }

    // Graph* to_gpu() {
    //   Graph gcopy = g;

    //   cudaMalloc(&gcopy.vertex_label, sizeof(bitarray32) * g.nnodes);
    //   cudaMalloc(&gcopy.rowptr, sizeof(graph_edge_t) * (g.nnodes + 1));
    //   cudaMalloc(&gcopy.colidx, sizeof(graph_node_t) * g.nedges);
    //   cudaMemcpy(gcopy.vertex_label, g.vertex_label, sizeof(bitarray32) * g.nnodes, cudaMemcpyHostToDevice);
    //   cudaMemcpy(gcopy.rowptr, g.rowptr, sizeof(graph_edge_t) * (g.nnodes + 1), cudaMemcpyHostToDevice);
    //   cudaMemcpy(gcopy.colidx, g.colidx, sizeof(graph_node_t) * g.nedges, cudaMemcpyHostToDevice);
    //   Graph* gpu_g;
    //   cudaMalloc(&gpu_g, sizeof(Graph));
    //   cudaMemcpy(gpu_g, &gcopy, sizeof(Graph), cudaMemcpyHostToDevice);
    //   // check_cuda_error(cudaMallocManaged(&gpu_g, sizeof(Graph)));
    //   // cudaMemAdvise(gpu_g, sizeof(Graph), cudaMemAdviseSetReadMostly, 0);
    //   return gpu_g;
    // }
Graph* to_gpu() {
    // 分配统一内存存储Graph结构
    Graph* gpu_g;
    check_cuda_error(cudaMallocManaged(&gpu_g, sizeof(Graph)));
    
    // 为成员分配统一内存
    check_cuda_error(cudaMallocManaged(&gpu_g->vertex_label, sizeof(bitarray32) * g.nnodes));
    check_cuda_error(cudaMallocManaged(&gpu_g->rowptr, sizeof(graph_edge_t) * (g.nnodes + 1)));
    check_cuda_error(cudaMallocManaged(&gpu_g->colidx, sizeof(graph_node_t) * g.nedges));
    
    // 复制数据内容
    memcpy(gpu_g->vertex_label, g.vertex_label, sizeof(bitarray32) * g.nnodes);
    memcpy(gpu_g->rowptr, g.rowptr, sizeof(graph_edge_t) * (g.nnodes + 1));
    memcpy(gpu_g->colidx, g.colidx, sizeof(graph_node_t) * g.nedges);
    
    // 设置内存建议
    int device = 0; // 假设使用GPU 0
    cudaMemAdvise(gpu_g->vertex_label, sizeof(bitarray32) * g.nnodes, cudaMemAdviseSetReadMostly, device);
    cudaMemAdvise(gpu_g->rowptr, sizeof(graph_edge_t) * (g.nnodes + 1), cudaMemAdviseSetReadMostly, device);
    cudaMemAdvise(gpu_g->colidx, sizeof(graph_node_t) * g.nedges, cudaMemAdviseSetReadMostly, device);
    
    return gpu_g;
}
    // TODO: dryadic graph format 

    void readfile(std::string& filename) {
      //read_lg_file(filename);
      read_bin_file(filename);

    }

    void read_lg_file(std::string& filename) {
      std::ifstream fin(filename);
      std::string line;
      while (std::getline(fin, line) && (line[0] == '#'));
      g.nnodes = 0;
      std::vector<int> vertex_labels;
      do {
        std::istringstream sin(line);
        char tmp;
        int v;
        int label;
        sin >> tmp >> v >> label;
        vertex_labels.push_back(label);
        g.nnodes++;
      } while (std::getline(fin, line) && (line[0] == 'v'));
      std::vector<std::vector<graph_node_t>> adj_list(g.nnodes);
      do {
        std::istringstream sin(line);
        char tmp;
        int v1, v2;
        int label;
        sin >> tmp >> v1 >> v2 >> label;
        adj_list[v1].push_back(v2);
        adj_list[v2].push_back(v1);
      } while (getline(fin, line));

      assert(vertex_labels.size() == g.nnodes);

      g.vertex_label = new bitarray32[vertex_labels.size()];
      for (int i = 0; i < g.nnodes; i++) {
        g.vertex_label[i] = (1 << vertex_labels[i]);
      }
      // memcpy(g.vertex_label, vertex_labels.data(), sizeof(int) * vertex_labels.size());

      g.rowptr = new graph_edge_t[g.nnodes + 1];
      g.rowptr[0] = 0;

      std::vector<graph_node_t> colidx;

      for (graph_node_t i = 0; i < g.nnodes; i++) {
        sort(adj_list[i].begin(), adj_list[i].end());
        int pos = 0;
        for (graph_node_t j = 1; j < adj_list[i].size(); j++) {
          if (adj_list[i][j] != adj_list[i][pos]) adj_list[i][++pos] = adj_list[i][j];
        }

        if (adj_list[i].size() > 0)
          colidx.insert(colidx.end(), adj_list[i].data(), adj_list[i].data() + pos + 1);  // adj_list is sorted

        adj_list[i].clear();
        g.rowptr[i + 1] = colidx.size();
      }
      g.nedges = colidx.size();
      g.colidx = new graph_node_t[colidx.size()];

      memcpy(g.colidx, colidx.data(), sizeof(graph_node_t) * colidx.size());

     // std::cout << "Graph read complete. Number of vertex: " << g.nnodes << std::endl;
    }


    template<typename T>
    void read_subfile(std::string fname, T*& pointer, size_t elements) {
      pointer = (T*)malloc(sizeof(T) * elements);
      assert(pointer);
      std::ifstream inf(fname.c_str(), std::ios::binary);
      if (!inf.good()) {
        std::cerr << "Failed to open file: " << fname << "\n";
        exit(1);
      }
      inf.read(reinterpret_cast<char*>(pointer), sizeof(T) * elements);
      inf.close();
    }

    std::vector<int> ReadPartitionFile(const std::string& filename, int vertex_count) {
      std::string tag = "read partition file";
      std::cout << "read partition file: " << filename << std::endl;

      std::vector<int> vertex_partition(vertex_count);  // vertex id -> partition
      std::ifstream file(filename.c_str(), std::fstream::in);
      assert(file.is_open());
      std::string line;
      int vid = 0;
      while (std::getline(file, line)) {
        int partition = std::stoi(line);
        vertex_partition[vid] = partition;
        vid++;
      }
      assert(vid == vertex_count);
      return vertex_partition;
    }


    void read_bin_file(std::string& filename) {
      std::ifstream f_meta((filename + ".meta.txt").c_str());
      assert(f_meta);

      graph_node_t n_vertices;
      graph_edge_t n_edges;
      int vid_size;
      graph_node_t max_degree;
      f_meta >> n_vertices >> n_edges >> vid_size >> max_degree;
      assert(sizeof(graph_node_t) == vid_size);
      f_meta.close();

      g.nnodes = n_vertices;
      g.nedges = n_edges;
      read_subfile(filename + ".vertex.bin", g.rowptr, n_vertices + 1);
      read_subfile(filename + ".edge.bin", g.colidx, n_edges);

      int* lb = new int[n_vertices];
      memset(lb, 1, n_vertices * sizeof(int));
      g.vertex_label = new bitarray32[n_vertices];
      if(LABELED) {
        read_subfile(filename + ".label_1000.bin", lb, n_vertices);
      }
      for (int i = 0; i < n_vertices; i++) {
        g.vertex_label[i] = (1 << lb[i]);
      }
      delete[] lb;

      //read partition file
      //std::string partition_file_name = filename + "_cut_" + std::to_string(g.partition_num) + ".txt";
      // g.partition_id = new int[n_vertices];
      // std::vector<int> partition_id = ReadPartitionFile("../data/com-friendster_greedy_2.txt", n_vertices);
      // for (int i = 0; i < n_vertices; i++) {
      //   g.partition_id[i] = partition_id[i];
      // }
    }

  };
}