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

typedef struct {
    graph_node_t nnodes = 0;
    graph_edge_t nedges = 0;
    graph_edge_t* rowptr;
    graph_node_t* colidx;
} Graph;

typedef struct{
    graph_node_t v[2];
}Edge;

typedef struct{
    graph_edge_t cut_num;
    Edge* edges;
}CutEdges;

struct GraphPreprocessor {
    
    Graph *g; 
    CutEdges cut_edges;
    std::vector<int> partitions;
    GraphPreprocessor(std::string filename, int partition_num) {
        g =new Graph[partition_num+1];
        readfile(filename);
        std::string partition_filename=filename+"_"+std::to_string(partition_num)+".txt";

        partitions = read_partitions(partition_filename);
        generate_part_graphs(partitions,partition_num);
        
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

        g[0].nnodes = n_vertices;
        g[0].nedges = n_edges;
        read_subfile(filename + ".vertex.bin", g[0].rowptr, n_vertices + 1);
        read_subfile(filename + ".edge.bin", g[0].colidx, n_edges);
    }

    std::vector<int> read_partitions(const std::string& filename) {

        std::cout << "read partition file: " << filename << std::endl;
        
        std::vector<int> vertex_partition(g[0].nnodes);  
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

    void generate_part_graphs(std::vector<int> partitions,int partition_num){
        graph_node_t vnum = g[0].nnodes;
        std::vector<std::vector<int>> degree_list(partition_num, std::vector<int>(vnum, 0));
        
        for(graph_node_t i = 0;i < vnum; i++){
            int p = partitions[i];
            int degree =  g[0].rowptr[i+1] - g[0].rowptr[i];
            degree_list[p][i] = degree;
        }
        
        for(int p = 0; p < partition_num; p++){
            g[p+1].rowptr = new graph_edge_t[vnum+1];
            g[p+1].rowptr[0] = 0;
            for(int i=0; i<vnum; i++){
                g[p+1].rowptr[i+1] = g[p+1].rowptr[i]+degree_list[p][i];
            }
            g[p+1].colidx = new graph_node_t[g[p+1].rowptr[vnum]];
            for(int i=0; i<vnum; i++){
                if(degree_list[p][i]!=0){
                    graph_edge_t src = g[0].rowptr[i];
                    graph_edge_t dst = g[p+1].rowptr[i];
                    for(int j = 0; j < degree_list[p][i]; j++){
                        g[p+1].colidx[dst+j] = g[0].colidx[src+j];
                    }
                }
            }
        }

        
        std::vector<std::pair<graph_node_t, graph_node_t>> cedges;
        for (graph_node_t u = 0; u < vnum; ++u) {  
            int u_partition = partitions[u];
            for (graph_edge_t j = g[0].rowptr[u]; j < g[0].rowptr[u + 1]; ++j) {
                graph_node_t v = g[0].colidx[j];
            
            if (partitions[v] != u_partition) {
                
                if (u < v) {
                    cedges.emplace_back(u, v);
                }
            }
            }
        }

        cut_edges.cut_num = cedges.size();
        cut_edges.edges = new Edge[cut_edges.cut_num];
        graph_edge_t index = 0;
        for(auto edge : cedges){
            cut_edges.edges[index].v[0]=edge.first;
            cut_edges.edges[index].v[1]=edge.second;
        }
    }
};




