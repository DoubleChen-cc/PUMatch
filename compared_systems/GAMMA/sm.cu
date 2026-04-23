#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "utils.h"
#include "accessMode.cuh"
#include "expand.cuh"
#include "queryGraph.cuh"
#include <cuda_runtime.h>

using namespace std;
__global__ void set_validation(CSRGraph g, uint8_t *valid_candi, uint32_t nnodes, uint8_t lab, uint32_t min_deg, bool labeled) {
	uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
	if(labeled){
		for (uint32_t i = tid; i < nnodes; i += (blockDim.x*gridDim.x)) {
			if (g.getDegree(i) >= min_deg && g.getData(i) == lab)
				valid_candi[i] = 1;
		}
	}
	else{
		for (uint32_t i = tid; i < nnodes; i += (blockDim.x*gridDim.x)) {
			if (g.getDegree(i) >= min_deg)
				valid_candi[i] = 1;
		}
	}
	return ;
}
int main(int argc, char *argv[]) {
	if (argc < 5) {
		printf("usage: ./sm <data_graph> <query_graph> <graph_mt> [truncate10|full] debug\n");
		printf("  truncate10 : after core[0]->core[1] expansion, keep ~first 10%% (natural order)\n");
		printf("  full       : keep all embeddings at level 1 (default if mode omitted)\n");
		return 0;
	}
	bool keep_all_after_l1 = true;
	if (argc >= 6) {
		std::string mode = argv[4];
		if (mode == "truncate10")
			keep_all_after_l1 = false;
		else if (mode == "full")
			keep_all_after_l1 = true;
		else {
			printf("usage: ./sm <data_graph> <query_graph> <graph_mt> [truncate10|full] debug\n");
			printf("  unknown mode: %s (use truncate10 or full)\n", argv[4]);
			return 0;
		}
	}
	if (string(argv[argc - 1]) != "debug") {
		log_set_quiet(true);
	}
	Clock start("Total");
	Clock graph_load("GraphLoad");
	Clock init("Init");
	/* 占位显存：扩展前申请，使空闲显存≈UM_memory_usage+expand预留，限制图UM等在GPU上的可用池；结束前 cudaFree */
	void *gpu_vram_pressure = nullptr;
	
	//assert(k <= embedding_max_length);
	queryGraph Gquery;
	graph_load.start();
	Gquery.readFromFile(argv[2],true);
	std::string file_name = argv[1];
	CSRGraph data_graph;
	mem_type mt_emb = (mem_type)1;//0 GPU 1 Unified 2 Zero 3 Combine
	mem_type mt_graph = (mem_type)atoi(argv[3]);
	if (mt_graph > 1)
		check_cuda_error(cudaSetDeviceFlags(cudaDeviceMapHost));
	data_graph.read(file_name, true, mt_graph);//for sm. we should definitely read vertex labels
	/* 图边表为 UM 时倾向驻留在 CPU，减轻与扩展中间结果抢 GPU 显存（按需仍会自动迁到 GPU） */
	if (mt_graph == UNIFIED_MEM && data_graph.edge_dst != nullptr) {
		check_cuda_error(cudaMemAdvise(data_graph.edge_dst, (size_t)data_graph.nedges * sizeof(uint32_t),
		                               cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	}
	log_info(graph_load.count("nedges %lu, nnodes %d", data_graph.get_nedges(), data_graph.get_nnodes()));
	
	start.start();
	init.start();
	EmbeddingList emb_list;
	uint32_t nnodes = data_graph.get_nnodes();
	uint64_t nedges = data_graph.get_nedges();

	uint8_t query_node = Gquery.core[0];
	uint8_t q_label = Gquery.vertex_label[query_node];
	uint32_t q_deg = Gquery.adj_list[query_node].size();
	
	printf("=== DEBUG: Initial Candidate Filtering ===\n");
	printf("Query graph info:\n");
	printf("  - Query node (first core): %d\n", query_node);
	printf("  - Query label: %d\n", q_label);
	printf("  - Query degree: %d\n", q_deg);
	printf("  - Query core size: %lu\n", Gquery.core.size());
	printf("  - Query vertex num: %d\n", Gquery.vertex_num);
	
	KeyT *seq, *results; 
	cudaMalloc((void **)&seq, sizeof(KeyT)*nnodes);
	check_cuda_error(cudaMalloc((void **)&results, sizeof(KeyT)*nnodes));
	check_cuda_error(cudaMemset(results, -1, sizeof(KeyT)*nnodes));
	uint8_t *valid_candi;
	check_cuda_error(cudaMalloc((void **)&valid_candi, sizeof(uint8_t)*nnodes));
	check_cuda_error(cudaMemset(valid_candi, 0, sizeof(uint8_t)*nnodes));
	set_validation<<<10000, 256>>>(data_graph, valid_candi, nnodes, q_label, q_deg, true);
	thrust::sequence(thrust::device, seq, seq + nnodes);
	uint32_t valid_node_num = thrust::copy_if(thrust::device, seq, seq + nnodes, valid_candi, results, is_valid())- results;
	check_cuda_error(cudaDeviceSynchronize());
	
	printf("  - Valid candidate nodes after filtering: %d\n", valid_node_num);
	if (valid_node_num == 0) {
		printf("ERROR: No valid candidate nodes found! Check:\n");
		printf("  1. Data graph labels match query label (%d)\n", q_label);
		printf("  2. Data graph has nodes with degree >= %d\n", q_deg);
		printf("  3. Label file is correctly loaded\n");
		return 0;
	}
	
	// Copy first few candidates for debugging
	KeyT *results_h = (KeyT*)malloc(sizeof(KeyT) * min(valid_node_num, 10));
	check_cuda_error(cudaMemcpy(results_h, results, sizeof(KeyT) * min(valid_node_num, 10), cudaMemcpyDeviceToHost));
	printf("  - First %d candidate nodes: ", min(valid_node_num, 10));
	for (int i = 0; i < min(valid_node_num, 10); i++) {
		printf("%d ", results_h[i]);
	}
	printf("\n");
	free(results_h);
	
	emb_list.init(valid_node_num, Gquery.core.size(), mt_emb, false);
	emb_list.copy_to_level(0, results, 0, valid_node_num);
	check_cuda_error(cudaFree(seq));
	check_cuda_error(cudaFree(valid_candi));
	check_cuda_error(cudaFree(results));

	{
		size_t free_mem = 0, total_mem = 0;
		check_cuda_error(cudaMemGetInfo(&free_mem, &total_mem));
		std::cout << "GPU剩余内存(embedding初始化后): " << free_mem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
	}

	//emb_list.init(nnodes, Gquery.core.size(), mt_emb, false);//TODO: need better initialization strategy: take degree and vertex label into consideration
	log_info(init.count("embedding initialization done!"));
	//TODO: here we plan to add a optimizer to determine expand order, expand constraint, and so on.
	//set the first level
	//KeyT *seq = (KeyT*)malloc(sizeof(KeyT)*nnodes);
	//thrust::sequence(thrust::host, seq, seq + nnodes);
	//emb_list.copy_to_level(0, seq, 0, nnodes);
	//free(seq);
	//log_info("the valid num of layer 0 is %lu",emb_list.check_valid_num(0));
	//set the second level
	//emb_list.add_level(nedges);
	//expand for every vertex in the query graph
	access_mode_controller access_controller;
	access_controller.set_vertex_page_border(data_graph);
	log_info(init.count("access controller initalization done!"));

	/* 目标：占位后 cudaMemGetInfo 的空闲 ≈ UM_memory_usage + expand_dynamic 一次申请的 emb_vid/emb_idx */
	const size_t UM_memory_usage = (size_t)8 * 1024 * 1024 * 1024;
	const size_t reserve_expand_bytes =
	    (size_t)EMB_FTR_CACHE_SIZE * sizeof(KeyT) + (size_t)EMB_FTR_CACHE_SIZE * sizeof(emb_off_type) + 64u * 1024 * 1024;
	{
		size_t free_mem = 0, total_mem = 0;
		check_cuda_error(cudaMemGetInfo(&free_mem, &total_mem));
		const size_t target_free = UM_memory_usage + reserve_expand_bytes;
		if (free_mem > target_free) {
			size_t pressure_bytes = free_mem - target_free;
			check_cuda_error(cudaMalloc(&gpu_vram_pressure, pressure_bytes));
			std::cout << "GPU显存占位: " << pressure_bytes / (1024.0 * 1024.0 * 1024.0) << " GiB"
			          << " (保留空闲≈" << UM_memory_usage / (1024.0 * 1024.0 * 1024.0) << " GiB 给图UM等 + expand预留 "
			          << reserve_expand_bytes / (1024.0 * 1024.0 * 1024.0) << " GiB)" << std::endl;
		} else {
			std::cout << "GPU显存占位: 跳过(空闲不足 target_free=" << target_free / (1024.0 * 1024.0 * 1024.0)
			          << " GiB)" << std::endl;
		}
		check_cuda_error(cudaMemGetInfo(&free_mem, &total_mem));
		std::cout << "GPU剩余内存(占位后): " << free_mem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
	}

	Clock Expand("Expand");
	log_info(Expand.start());
	/*for (uint32_t i = 0; i < Gquery.vertex_num; i ++) {
		cout << "here is vertex " << i << ":" << endl;
		cout << "nbr ";
		for (uint32_t j = 0; j < Gquery.adj_list[i].size(); j ++) {
			cout << Gquery.adj_list[i][j] << " ";
		}
		cout << "order nbr:";
		for (uint32_t j = 0; j < Gquery.order_list[i].size(); j ++) {
			cout << Gquery.order_list[i][j] << " ";
		}
		cout << endl;
	}
	for (uint32_t i = 0; i < Gquery.core.size(); i ++)
		cout << (uint32_t)Gquery.core[i] << " ";
	cout << endl;
	for (uint32_t i = 0; i < Gquery.satellite.size(); i ++)
		cout << (uint32_t)Gquery.satellite[i] << " ";
	cout << endl;*/

	for (int i = 1; i < Gquery.core.size(); i ++) {
		//construct the expand constraint
		uint64_t _nbrs = 0, _order_nbr = 0;
		//int8_t *_order_nbr_cmp = new int8_t [i];
		uint32_t query_node = Gquery.core[i];
		//cout << "the query node now is " << query_node << endl;
		uint32_t e_nbr_size = 0, e_order_nbr_size = 0;
		for (uint32_t j = 0; j < Gquery.adj_list[query_node].size(); j ++) {
			uint32_t cur_nbr = Gquery.adj_list[query_node][j];
			//cout << query_node << " " << cur_nbr << " " ; cout << Gquery.adj_list[query_node][j];
			if (query_node > Gquery.adj_list[query_node][j]) {//here we use id as matching order, resulting in this
				_nbrs = _nbrs | ((cur_nbr&0xff) << (j*8));
				e_nbr_size ++;
			}
			//cout  << " " << e_nbr_size << endl;
		}
		// Order constraint construction - DISABLED
		//for (uint32_t j = 0; j < Gquery.order_list[query_node].size(); j ++) {
		//	if (query_node > Gquery.order_list[query_node][j]) {
		//		_order_nbr = _order_nbr | ((Gquery.order_list[query_node][j]&0xff) << (j*8));
		//		e_order_nbr_size ++;
		//	}
		//}
		expand_constraint ec((node_data_type)Gquery.vertex_label[query_node], Gquery.adj_list[query_node].size(),
							 _nbrs, e_nbr_size, 
							 (emb_order)1, 0, 0);  // order_nbr=0, order_nbr_size=0 (disabled)
		
		printf("=== DEBUG: Expansion Iteration %d ===\n", i);
		printf("  - Query node: %d\n", query_node);
		printf("  - Query label: %d\n", Gquery.vertex_label[query_node]);
		printf("  - Query degree: %lu\n", Gquery.adj_list[query_node].size());
		printf("  - Previous level size: %lu\n", emb_list.size(i-1));
		printf("  - Neighbor size in constraint: %d\n", e_nbr_size);
		printf("  - Order neighbor size: %d\n", e_order_nbr_size);
		
		//cout << "adjacency list and order list length is " << e_nbr_size << " " << e_order_nbr_size << endl;
		//expand
		log_info(Expand.count("for the %dth iteration, start expand... ...",i));
		bool write_back = (i == Gquery.core.size()-1) ? false : true;
		printf("  - Write back: %s (last level: %s)\n", write_back ? "true" : "false", 
		       (i == Gquery.core.size()-1) ? "yes" : "no");
		// use fixed cache size (EMB_FTR_CACHE_SIZE) again for stability
		expand_dynamic(data_graph, emb_list, i, ec, write_back);
		//expand_in_batch(data_graph, emb_list, i, ec);
		log_info(Expand.count("for the %dth iteration, end expand",i));
		
		emb_off_type current_size = emb_list.size(i);

		if (i == 1 && current_size > 0 && !keep_all_after_l1) {
			emb_off_type keep_size = current_size / 10;
			if (keep_size == 0)
				keep_size = 1;
			if (keep_size < current_size) {
				printf("  - Truncating level %d embeddings from %lu to %lu (top 10%% by natural order)\n", i,
				       (unsigned long)current_size, (unsigned long)keep_size);
				emb_list.remove_tail(keep_size);
				current_size = keep_size;
			}
		}

		printf("  - Current level size after expand: %lu\n", current_size);
		if (current_size == 0 && i < Gquery.core.size()-1) {
			printf("WARNING: No embeddings found at level %d! Expansion may have failed.\n", i);
			printf("  This means the query structure may not exist in the data graph.\n");
		} else if (current_size == 0 && i == Gquery.core.size()-1) {
			printf("WARNING: Final level has 0 embeddings!\n");
			printf("  This could be due to:\n");
			printf("  1. Query structure doesn't exist in data graph\n");
			printf("  2. Final expansion constraints too strict\n");
			printf("  3. Results not properly written back (check copy_back logic)\n");
		}
		//log_info("the valid num of layer %d is %lu", i, results);
		//set access mode
		if (mt_graph == 3) {
			Expand.pause();
			access_controller.cal_access_mode_by_EL(data_graph, ec, emb_list);
			Expand.goon();
		}
		log_info(Expand.count("for the %dth iteration, end set access mode",i));
		//delete ec;
	}
	log_info(Expand.count("end expand"));
	log_info(start.count("subgraph matching ends."));
	
	// Print detailed timing summary
	printf("\n=== TIMING SUMMARY ===\n");
	printf("Graph loading time: %.6f seconds\n", graph_load.get_time());
	printf("Initialization time: %.6f seconds\n", init.get_time());
	printf("Expansion time: %.6f seconds\n", Expand.get_time());
	printf("Total execution time: %.6f seconds\n", start.get_time());
	printf("========================================\n");
	
	printf("\n=== DEBUG: Final Results ===\n");
	for (int i = 0; i < Gquery.core.size(); i++) {
		printf("  - Level %d size: %lu\n", i, emb_list.size(i));
	}
	emb_off_type final_result = emb_list.size(Gquery.core.size()-1);
	printf("the numbers of the matched subgraph is %lu\n", final_result);

	// Write results to result.txt in format: data_graph query_graph total_time match_count
	double total_time = start.get_time();  // algorithm total time (excluding graph loading)
	std::ofstream result_file("result.txt", std::ios::app);
	if (result_file.is_open()) {
		result_file << file_name << " " << argv[2] << " " << total_time << " " << final_result << std::endl;
		result_file.close();
	} else {
		std::cerr << "Failed to open result.txt for writing" << std::endl;
	}

	if (final_result == 0) {
		printf("\n=== TROUBLESHOOTING ===\n");
		printf("Possible reasons for zero matches:\n");
		printf("1. Initial candidate filtering too strict (check labels and degrees)\n");
		printf("2. Query graph structure doesn't exist in data graph\n");
		printf("3. Label mismatch between query and data graph\n");
		printf("4. Query graph format issue (check query file)\n");
		printf("5. Data graph format issue (check .label.bin file)\n");
	}
	if (gpu_vram_pressure) {
		check_cuda_error(cudaFree(gpu_vram_pressure));
		gpu_vram_pressure = nullptr;
	}
	emb_list.clean();
	access_controller.clean();
	data_graph.clean();
	return 0;
}
	
