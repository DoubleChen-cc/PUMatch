#include<iostream>
#include<stdlib.h>
#include<vector>
#include "utils.h"
#include<fstream>
#include <sstream>
#include<string.h>
using namespace std;
class queryGraph{
public:
	uint32_t vertex_num;
	vector<uint8_t> vertex_label;
	vector<vector<uint32_t>> adj_list;
	vector<vector<uint32_t>> order_list;
	vector<uint32_t> core;
	vector<uint32_t> satellite;
	void readFromFile(const char filename [],bool labeled) {
		std::ifstream file_read;
		file_read.open(filename, std::ios::in);
		std::string line;
		while (std::getline(file_read, line) && (line[0] == '#'));
		vertex_num = 0;

		do {
			std::istringstream sin(line);
			char tmp;
			int v;
			int label;
			sin >> tmp >> v >> label;
			if(labeled){
			vertex_label.push_back(static_cast<uint8_t>(label));
			}
			else{
			vertex_label.push_back(1);
			}
			vertex_num++;
		} while (std::getline(file_read, line) && (line[0] == 'v'));

		int adj_matrix_[7][7];
		memset(adj_matrix_, 0, sizeof(adj_matrix_));
		do {
		std::istringstream sin(line);
		char tmp;
		int v1, v2;
		int label;
		sin >> tmp >> v1 >> v2 >> label;
		adj_matrix_[v1][v2] = label;
		adj_matrix_[v2][v1] = label;
		} while (getline(file_read, line));

		//file_read >> vertex_num;
		//cout << vertex_num << endl;
		// for (uint32_t i = 0; i < vertex_num; i ++) {
		// 	uint32_t label_now;
		// 	file_read >> label_now;
		// 	//cout << label_now << " ";
		// 	vertex_label.push_back((uint8_t)label_now);
		// }
		//cout << endl;
		for (uint32_t i = 0; i < vertex_num; i ++) {
			vector<uint32_t> v;
			vector<uint32_t> order_v;
			uint32_t nbr_size = 0;
			//file_read >> nbr_size;
			//cout << nbr_size << " ";
			for(uint32_t j = 0; j < vertex_num; j ++){
				if(adj_matrix_[i][j] != 0){
					v.push_back(j);
					order_v.push_back(j);
					nbr_size++;
				}
			}
			// uint32_t nbr_now;
			// for (uint32_t j = 0; j < nbr_size; j ++) {
			// 	file_read >> nbr_now;
			// 	//cout << nbr_now << " ";
			// 	v.push_back(nbr_now);
			// }
			adj_list.push_back(v);
			order_list.push_back(order_v);
			//cout << endl;
		}
		// for (uint32_t i = 0; i < vertex_num; i ++) {
		// 	vector<uint32_t> v;
		// 	uint32_t nbr_size;
		// 	file_read >> nbr_size;
		// 	//cout << nbr_size << " ";
		// 	uint32_t order_nbr_now;
		// 	for (uint32_t j = 0; j < nbr_size; j ++) {
		// 		file_read >> order_nbr_now;
		// 		//cout << order_nbr_now << " ";
		// 		v.push_back(order_nbr_now);
		// 	}
		// 	order_list.push_back(v);
		// 	//cout << endl;
		// }
		file_read.close();
		for (uint32_t i = 0; i < vertex_num; i ++) {
			if (adj_list[i].size() > 1) {
				core.push_back(i);
			}
			else {
				satellite.push_back(i);
			}
		}
		return ;
	}
	void clear() {
		for (uint32_t i = 0; i < vertex_num; i ++) {
			adj_list[i].clear();
			order_list[i].clear();
		}
		adj_list.clear();
		order_list.clear();
		core.clear();
		satellite.clear();
		return ;
	}
};
