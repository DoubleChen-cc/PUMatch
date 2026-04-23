#pragma once

#include <cstddef>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cassert>
#include "config.h"
#include <string>
#include <cstring>   

typedef struct {
    graph_node_t nnodes = 0;
    graph_node_t pnodes = 0;
    graph_edge_t nedges = 0;
    graph_edge_t* rowptr;
    graph_node_t* colidx;
    bitarray32* vertex_label;
    int* partitions;
    int threshold = 1024;
} Graph;

typedef struct{
    graph_node_t v[2];
}Edge;

typedef struct{
    graph_edge_t cut_num;
    Edge* edges;
}CutEdges;

struct GraphPreprocessor {
    
    Graph g; 
    CutEdges cut_edges;
    std::vector<int> partitions;
    GraphPreprocessor(std::string filename) {
        readfile(filename);
    }

    int build_partial_graph(float ratio){
        graph_edge_t edge_limit_num = g.nedges * ratio;
    
        
        std::vector<int> degree_thresholds = {1024, 512, 256, 128, 64, 32, 16, 0};
        std::vector<graph_node_t> selected_vertices;
        graph_edge_t selected_edges = 0;

        std::vector<int> degrees(g.nnodes);
        for (graph_node_t i = 0; i < g.nnodes; ++i) {
            degrees[i] = g.rowptr[i + 1] - g.rowptr[i];
        }

        std::vector<bool> selected(g.nnodes, false);
        graph_edge_t accumulated_degree = 0;
        int final_threshold = 1024;
        graph_node_t final_node = 0;
        for (int t = 0; t < degree_thresholds.size(); ++t) {
            int threshold = degree_thresholds[t];
            bool flag = false;
            for (graph_node_t i = 0; i < g.nnodes; ++i) {
                if (!selected[i] && degrees[i] >= threshold) {
                    selected[i] = true;
                    selected_vertices.push_back(i);
                    accumulated_degree += degrees[i];
                    if(accumulated_degree >= edge_limit_num){
                        flag = true;
                        final_threshold = threshold;
                        final_node = i;
                        std::cout<<"threshold:"<<threshold<<"  idx:"<<final_node<<std::endl;
                        break;
                    }
                }
            }
            if(flag){
                break;
            }
            
            
            
            
            
        }

        
        std::vector<graph_edge_t> new_rowptr(g.nnodes + 1, 0);
        std::vector<graph_node_t> new_colidx;
        new_colidx.reserve(accumulated_degree); 

        graph_edge_t cur_edge_ptr = 0;
        for (graph_node_t i = 0; i < g.nnodes; ++i) {
            if (selected[i]) {
                graph_edge_t deg = degrees[i];
                new_rowptr[i + 1] = new_rowptr[i] + deg;
                for (graph_edge_t eid = g.rowptr[i]; eid < g.rowptr[i + 1]; ++eid) {
                    new_colidx.push_back(g.colidx[eid]);
                }
            } else {
                
                new_rowptr[i + 1] = new_rowptr[i];
            }
        }

        
        free(g.rowptr);
        free(g.colidx);
        g.rowptr = (graph_edge_t*)malloc(sizeof(graph_edge_t) * (g.nnodes + 1));
        memcpy(g.rowptr, new_rowptr.data(), sizeof(graph_edge_t) * (g.nnodes + 1));
        g.colidx = (graph_node_t*)malloc(sizeof(graph_node_t) * new_colidx.size());
        memcpy(g.colidx, new_colidx.data(), sizeof(graph_node_t) * new_colidx.size());
        g.nedges = new_colidx.size();
        g.pnodes = selected_vertices.size();
        g.threshold = final_threshold;
        return final_node;
    }

    void build_partial_graph(std::string partition_filename,int partition_num){
        partitions = read_partitions(partition_filename);
        g.partitions = new int[g.nnodes];
        memcpy(g.partitions, partitions.data(), sizeof(int) * g.nnodes);
        generate_part_graphs(partitions,partition_num);
    }
    void generate_part_graphs(std::vector<int> partitions,int partition_num){
        graph_node_t vnum = g.nnodes;
        const int first_partition = 0; 
        
        
        std::vector<int> degree_list(vnum, 0);
        for(graph_node_t i = 0; i < vnum; i++){
            if(partitions[i] == first_partition){
                degree_list[i] = g.rowptr[i+1] - g.rowptr[i];
            } else {
                degree_list[i] = 0; 
            }
        }
        
        
        std::vector<graph_edge_t> new_rowptr(vnum + 1, 0);
        new_rowptr[0] = 0;
        for(graph_node_t i = 0; i < vnum; i++){
            new_rowptr[i+1] = new_rowptr[i] + degree_list[i];
        }
        
        
        std::vector<graph_node_t> new_colidx;
        new_colidx.reserve(new_rowptr[vnum]); 
        
        for(graph_node_t i = 0; i < vnum; i++){
            if(partitions[i] == first_partition){
                
                for(graph_edge_t j = g.rowptr[i]; j < g.rowptr[i+1]; j++){
                    new_colidx.push_back(g.colidx[j]);
                }
            }
            
        }
        
        
        free(g.rowptr);
        free(g.colidx);
        g.rowptr = (graph_edge_t*)malloc(sizeof(graph_edge_t) * (vnum + 1));
        memcpy(g.rowptr, new_rowptr.data(), sizeof(graph_edge_t) * (vnum + 1));
        g.colidx = (graph_node_t*)malloc(sizeof(graph_node_t) * new_colidx.size());
        memcpy(g.colidx, new_colidx.data(), sizeof(graph_node_t) * new_colidx.size());
        g.nedges = new_colidx.size();
    }
    Graph* to_gpu() {
      Graph gcopy = g;

      cudaMalloc(&gcopy.vertex_label, sizeof(bitarray32) * g.nnodes);
      cudaMalloc(&gcopy.rowptr, sizeof(graph_edge_t) * (g.nnodes + 1));
      cudaMalloc(&gcopy.colidx, sizeof(graph_node_t) * g.nedges);
      cudaMalloc(&gcopy.partitions, sizeof(int) * g.nnodes);
      cudaMemcpy(gcopy.vertex_label, g.vertex_label, sizeof(bitarray32) * g.nnodes, cudaMemcpyHostToDevice);
      cudaMemcpy(gcopy.rowptr, g.rowptr, sizeof(graph_edge_t) * (g.nnodes + 1), cudaMemcpyHostToDevice);
      cudaMemcpy(gcopy.colidx, g.colidx, sizeof(graph_node_t) * g.nedges, cudaMemcpyHostToDevice);
      cudaMemcpy(gcopy.partitions, g.partitions, sizeof(int) * g.nnodes, cudaMemcpyHostToDevice);
      Graph* gpu_g;
      cudaMalloc(&gpu_g, sizeof(Graph));
      cudaMemcpy(gpu_g, &gcopy, sizeof(Graph), cudaMemcpyHostToDevice);
      return gpu_g;
    }

    Graph* to_gpu_managed() {
        
        Graph* gpu_g;
        cudaMallocManaged(&gpu_g, sizeof(Graph));
        
        
        cudaMallocManaged(&gpu_g->vertex_label, sizeof(bitarray32) * g.nnodes);
        cudaMallocManaged(&gpu_g->rowptr, sizeof(graph_edge_t) * (g.nnodes + 1));
        cudaMallocManaged(&gpu_g->colidx, sizeof(graph_node_t) * g.nedges);
        
        
        memcpy(gpu_g->vertex_label, g.vertex_label, sizeof(bitarray32) * g.nnodes);
        memcpy(gpu_g->rowptr, g.rowptr, sizeof(graph_edge_t) * (g.nnodes + 1));
        memcpy(gpu_g->colidx, g.colidx, sizeof(graph_node_t) * g.nedges);
        
        
        int device = 0; 
        cudaMemAdvise(gpu_g->vertex_label, sizeof(bitarray32) * g.nnodes, cudaMemAdviseSetReadMostly, device);
        cudaMemAdvise(gpu_g->rowptr, sizeof(graph_edge_t) * (g.nnodes + 1), cudaMemAdviseSetReadMostly, device);
        cudaMemAdvise(gpu_g->colidx, sizeof(graph_node_t) * g.nedges, cudaMemAdviseSetReadMostly, device);
        
        return gpu_g;
    }


    void readfile(std::string& filename) {
        read_bin_file(filename);
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
            read_subfile(filename + ".label.bin", lb, n_vertices);
        }
        for (int i = 0; i < n_vertices; i++) {
            g.vertex_label[i] = (1 << lb[i]);
        }
        delete[] lb;
    }

    std::vector<int> read_partitions(const std::string& filename) {

        std::cout << "read partition file: " << filename << std::endl;
        
        std::vector<int> vertex_partition(g.nnodes);  
        std::ifstream file(filename.c_str(), std::fstream::in);
        assert(file.is_open());
        std::string line;
        int vid = 0;
        while (std::getline(file, line)) {
            int partition = std::stoi(line);
            vertex_partition[vid] = partition;
            vid++;
        }

        return vertex_partition;
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
};




