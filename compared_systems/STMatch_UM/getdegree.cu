#include <string>
#include <iostream>
#include "src/gpu_match.cuh"
using namespace std;
using namespace STMatch;
int main(int argc, char* argv[]) {

  cudaSetDevice(0);
  
  STMatch::GraphPreprocessor g(argv[1]);
  std::vector<std::pair<graph_node_t, graph_node_t>> v_degree(g.g.nnodes);

  for(graph_node_t i = 0;i<g.g.nnodes;i++){
    graph_node_t d = static_cast<graph_node_t>(g.g.rowptr[i+1]-g.g.rowptr[i]);
    v_degree.push_back(std::make_pair(i,d));
  }
  std::sort(v_degree.begin(), v_degree.end(), 
    [](const auto& a, const auto& b) {
        return a.second > b.second; // 比较pair的第二个元素（度数d）
    }
  );

  auto middle_data = v_degree[g.g.nnodes/2];
  std::cout<<"中位数度： "<<middle_data.second<<" 顶点号："<<middle_data.first<<std::endl;
  int less_than_1000 = 0;
  for(graph_node_t i = 0;i<g.g.nnodes;i++){
    graph_node_t d = v_degree[i].second;
    if(d<1000){
        less_than_1000 = i;
        break;
    }
  }
  int count = g.g.nnodes - less_than_1000;
  float rate = static_cast<float> (count) / g.g.nnodes;
  std::cout<<"度小于1000的数量: "<<count<<" 占比："<<rate<<std::endl;
  return 0;
}