/*
 * 多机多 GPU 子图匹配（基于 sm.cu）
 * - MPI：进程间仅做最终匹配数归约；各 rank 持有完整数据图与查询图副本。
 * - 负载均衡：按第一层有效候选顶点在 rank 间做块划分（静态划分）。
 *
 * 用法:
 *   mpirun -np P ./sm_mpi <data_graph> <query_graph> graph_mt [truncate10|full] debug
 * graph_mt / debug 与 sm 相同；可选 [truncate10|full] 与 sm 相同（省略则 full）。
 *
 * 运行结束后 rank 0 向 gamma_result.txt 追加一行（制表符分隔）：
 *   数据图路径  pattern路径  rank0总时间(s) rank1总时间(s) ... 全局匹配总数
 *   总时间为各 rank 上 Clock start 起的 wall time（含读图、过滤、本 rank 扩展等）。
 *
 * 与 sm.cu 相同：UNIFIED_MEM 图边表 MemAdvise 倾向 CPU；各 rank 在本地 GPU 上做显存占位(UM_memory_usage+expand预留)。
 */
#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include "utils.h"
#include "accessMode.cuh"
#include "expand.cuh"
#include "queryGraph.cuh"
#include <cuda_runtime.h>

using namespace std;

__global__ void set_validation(CSRGraph g, uint8_t *valid_candi, uint32_t nnodes, uint8_t lab,
                               uint32_t min_deg, bool labeled) {
	uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (labeled) {
		for (uint32_t i = tid; i < nnodes; i += (blockDim.x * gridDim.x)) {
			if (g.getDegree(i) >= min_deg && g.getData(i) == lab)
				valid_candi[i] = 1;
		}
	} else {
		for (uint32_t i = tid; i < nnodes; i += (blockDim.x * gridDim.x)) {
			if (g.getDegree(i) >= min_deg)
				valid_candi[i] = 1;
		}
	}
}

/* 将 total 个任务划分到 world_size 个进程：前 rem 个 rank 多分配 1 个 */
static void partition_range(int rank, int world_size, uint32_t total, uint32_t *out_begin,
                            uint32_t *out_count) {
	if (world_size <= 0) {
		*out_begin = 0;
		*out_count = total;
		return;
	}
	if (total == 0) {
		*out_begin = 0;
		*out_count = 0;
		return;
	}
	uint32_t base = total / (uint32_t)world_size;
	uint32_t rem = total % (uint32_t)world_size;
	uint32_t lo = (uint32_t)rank * base + (uint32_t)min(rank, (int)rem);
	uint32_t cnt = base + ((uint32_t)rank < rem ? 1u : 0u);
	*out_begin = lo;
	*out_count = cnt;
}

static void bind_gpu_to_local_rank() {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm node_comm;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
	int local_rank = 0;
	if (node_comm != MPI_COMM_NULL)
		MPI_Comm_rank(node_comm, &local_rank);
	int n_gpu = 0;
	cudaGetDeviceCount(&n_gpu);
	if (n_gpu <= 0) {
		if (rank == 0)
			fprintf(stderr, "sm_mpi: no CUDA device\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	check_cuda_error(cudaSetDevice(local_rank % n_gpu));
	if (node_comm != MPI_COMM_NULL)
		MPI_Comm_free(&node_comm);
}

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	int mpi_rank, mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	bind_gpu_to_local_rank();

	if (argc < 5) {
		if (mpi_rank == 0) {
			printf("usage: mpirun -np P ./sm_mpi <data_graph> <query_graph> graph_mt [truncate10|full] debug\n");
			printf("  truncate10 : after level-1 expansion, keep ~10%% per rank (natural order)\n");
			printf("  full       : keep all (default if mode omitted)\n");
		}
		MPI_Finalize();
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
			if (mpi_rank == 0)
				printf("sm_mpi: unknown mode %s (use truncate10 or full)\n", argv[4]);
			MPI_Finalize();
			return 0;
		}
	}
	if (string(argv[argc - 1]) != "debug") {
		log_set_quiet(true);
	}

	Clock start("Total");
	Clock graph_load("GraphLoad");
	Clock init("Init");

	queryGraph Gquery;
	graph_load.start();
	Gquery.readFromFile(argv[2], true);
	std::string file_name = argv[1];
	CSRGraph data_graph;
	mem_type mt_emb = (mem_type)1;
	mem_type mt_graph = (mem_type)atoi(argv[3]);
	if (mt_graph > 1)
		check_cuda_error(cudaSetDeviceFlags(cudaDeviceMapHost));
	data_graph.read(file_name, true, mt_graph);
	if (mt_graph == UNIFIED_MEM && data_graph.edge_dst != nullptr) {
		check_cuda_error(cudaMemAdvise(data_graph.edge_dst, (size_t)data_graph.nedges * sizeof(uint32_t),
		                               cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	}
	if (mpi_rank == 0)
		log_info(graph_load.count("nedges %lu, nnodes %d", data_graph.get_nedges(), data_graph.get_nnodes()));

	start.start();
	init.start();
	uint32_t nnodes = data_graph.get_nnodes();
	uint64_t nedges = data_graph.get_nedges();
	(void)nedges;

	uint8_t query_node = Gquery.core[0];
	uint8_t q_label = Gquery.vertex_label[query_node];
	uint32_t q_deg = Gquery.adj_list[query_node].size();

	KeyT *seq, *results;
	cudaMalloc((void **)&seq, sizeof(KeyT) * nnodes);
	check_cuda_error(cudaMalloc((void **)&results, sizeof(KeyT) * nnodes));
	check_cuda_error(cudaMemset(results, -1, sizeof(KeyT) * nnodes));
	uint8_t *valid_candi;
	check_cuda_error(cudaMalloc((void **)&valid_candi, sizeof(uint8_t) * nnodes));
	check_cuda_error(cudaMemset(valid_candi, 0, sizeof(uint8_t) * nnodes));
	set_validation<<<10000, 256>>>(data_graph, valid_candi, nnodes, q_label, q_deg, true);
	thrust::sequence(thrust::device, seq, seq + nnodes);
	uint32_t valid_node_num =
	    thrust::copy_if(thrust::device, seq, seq + nnodes, valid_candi, results, is_valid()) - results;
	check_cuda_error(cudaDeviceSynchronize());

	if (mpi_rank == 0) {
		printf("[sm_mpi] MPI ranks: %d, valid candidates (level 0): %u\n", mpi_size, valid_node_num);
	}
	if (valid_node_num == 0) {
		if (mpi_rank == 0) {
			printf("ERROR: No valid candidate nodes.\n");
		}
		check_cuda_error(cudaFree(seq));
		check_cuda_error(cudaFree(valid_candi));
		check_cuda_error(cudaFree(results));
		data_graph.clean();
		MPI_Finalize();
		return 0;
	}

	uint32_t my_begin, my_count;
	partition_range(mpi_rank, mpi_size, valid_node_num, &my_begin, &my_count);
	if (mpi_rank == 0)
		printf("[sm_mpi] static partition: per-rank chunk ~ %u (rank0 count %u)\n",
		       valid_node_num / (uint32_t)mpi_size + 1, my_count);

	check_cuda_error(cudaFree(seq));
	check_cuda_error(cudaFree(valid_candi));

	uint64_t local_match = 0;
	/* 匹配时间：仅多层 expand（含 mt_graph==3 时 access 模式计算中的 Expand 暂停段），不含读图/查询、候选过滤、embedding 与 access 预计算 */
	double match_time_sec = 0.0;
	/* 与打印的 Total execution time 一致：自 start.start() 起的 wall time（本 rank） */
	double total_exec_sec = 0.0;
	void *gpu_vram_pressure = nullptr;

	if (my_count == 0) {
		check_cuda_error(cudaFree(results));
		total_exec_sec = start.get_time();
	} else {
		EmbeddingList emb_list;
		emb_list.init(my_count, Gquery.core.size(), mt_emb, false);
		KeyT *results_slice = results + my_begin;
		emb_list.copy_to_level(0, results_slice, 0, (int)my_count);
		check_cuda_error(cudaFree(results));

		{
			size_t free_mem = 0, total_mem = 0;
			check_cuda_error(cudaMemGetInfo(&free_mem, &total_mem));
			std::cout << "[rank " << mpi_rank << "] GPU剩余内存(embedding初始化后): " << free_mem / (1024.0 * 1024.0 * 1024.0)
			          << " GB" << std::endl;
		}

		log_info(init.count("embedding initialization done!"));

		access_mode_controller access_controller;
		access_controller.set_vertex_page_border(data_graph);
		log_info(init.count("access controller initalization done!"));

		const size_t UM_memory_usage = (size_t)4 * 1024 * 1024 * 1024;
		const size_t reserve_expand_bytes =
		    (size_t)EMB_FTR_CACHE_SIZE * sizeof(KeyT) + (size_t)EMB_FTR_CACHE_SIZE * sizeof(emb_off_type) + 64u * 1024 * 1024;
		{
			size_t free_mem = 0, total_mem = 0;
			check_cuda_error(cudaMemGetInfo(&free_mem, &total_mem));
			const size_t target_free = UM_memory_usage + reserve_expand_bytes;
			if (free_mem > target_free) {
				size_t pressure_bytes = free_mem - target_free;
				check_cuda_error(cudaMalloc(&gpu_vram_pressure, pressure_bytes));
				std::cout << "[rank " << mpi_rank << "] GPU显存占位: " << pressure_bytes / (1024.0 * 1024.0 * 1024.0)
				          << " GiB (保留空闲≈" << UM_memory_usage / (1024.0 * 1024.0 * 1024.0) << " GiB + expand预留 "
				          << reserve_expand_bytes / (1024.0 * 1024.0 * 1024.0) << " GiB)" << std::endl;
			} else {
				std::cout << "[rank " << mpi_rank << "] GPU显存占位: 跳过(target_free="
				          << target_free / (1024.0 * 1024.0 * 1024.0) << " GiB)" << std::endl;
			}
			check_cuda_error(cudaMemGetInfo(&free_mem, &total_mem));
			std::cout << "[rank " << mpi_rank << "] GPU剩余内存(占位后): " << free_mem / (1024.0 * 1024.0 * 1024.0) << " GB"
			          << std::endl;
		}

		Clock Expand("Expand");
		log_info(Expand.start());

		for (int i = 1; i < (int)Gquery.core.size(); i++) {
			uint64_t _nbrs = 0;
			uint32_t qn = Gquery.core[i];
			uint32_t e_nbr_size = 0;
			for (uint32_t j = 0; j < Gquery.adj_list[qn].size(); j++) {
				uint32_t cur_nbr = Gquery.adj_list[qn][j];
				if (qn > cur_nbr) {
					_nbrs = _nbrs | ((cur_nbr & 0xff) << (j * 8));
					e_nbr_size++;
				}
			}
			expand_constraint ec((node_data_type)Gquery.vertex_label[qn], Gquery.adj_list[qn].size(), _nbrs,
			                     e_nbr_size, (emb_order)1, 0, 0);

			log_info(Expand.count("for the %dth iteration, start expand... ...", i));
			bool write_back = (i == (int)Gquery.core.size() - 1) ? false : true;
			expand_dynamic(data_graph, emb_list, i, ec, write_back);
			log_info(Expand.count("for the %dth iteration, end expand", i));

			emb_off_type current_size = emb_list.size(i);
			if (i == 1 && current_size > 0 && !keep_all_after_l1) {
				emb_off_type keep_size = current_size / 10;
				if (keep_size == 0)
					keep_size = 1;
				if (keep_size < current_size) {
					if (mpi_rank == 0)
						printf("  - Truncating level %d embeddings from %lu to %lu (per-rank top 10%%)\n", i,
						       (unsigned long)current_size, (unsigned long)keep_size);
					emb_list.remove_tail(keep_size);
				}
			}

			if (mt_graph == 3) {
				Expand.pause();
				access_controller.cal_access_mode_by_EL(data_graph, ec, emb_list);
				Expand.goon();
			}
			log_info(Expand.count("for the %dth iteration, end set access mode", i));
		}

		log_info(Expand.count("end expand"));
		log_info(start.count("subgraph matching ends."));

		match_time_sec = Expand.get_time();
		total_exec_sec = start.get_time();
		if (mpi_rank == 0) {
			printf("\n=== TIMING SUMMARY (rank 0) ===\n");
			printf("Graph loading time: %.6f seconds\n", graph_load.get_time());
			printf("Initialization time: %.6f seconds\n", init.get_time());
			printf("Expansion time (match, rank0): %.6f seconds\n", match_time_sec);
			printf("Total execution time: %.6f seconds\n", total_exec_sec);
		}

		local_match = emb_list.size(Gquery.core.size() - 1);

		if (mpi_rank == 0) {
			printf("\n=== DEBUG: Final Results (rank 0 local) ===\n");
			for (int i = 0; i < (int)Gquery.core.size(); i++)
				printf("  - Level %d size: %lu\n", i, emb_list.size(i));
			printf("local matched subgraph count (rank 0): %lu\n", (unsigned long)local_match);
		}

		if (gpu_vram_pressure) {
			check_cuda_error(cudaFree(gpu_vram_pressure));
			gpu_vram_pressure = nullptr;
		}
		emb_list.clean();
		access_controller.clean();
	}

	MPI_Barrier(MPI_COMM_WORLD);
	uint64_t global_match = 0;
	MPI_Allreduce(&local_match, &global_match, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

	std::vector<double> all_total_exec;
	if (mpi_rank == 0)
		all_total_exec.resize((size_t)mpi_size);
	MPI_Gather(&total_exec_sec, 1, MPI_DOUBLE, mpi_rank == 0 ? all_total_exec.data() : nullptr, 1, MPI_DOUBLE, 0,
	           MPI_COMM_WORLD);

	if (mpi_rank == 0) {
		printf("\n=== GLOBAL (all ranks) ===\n");
		printf("the numbers of the matched subgraph is %lu\n", (unsigned long)global_match);

		std::ofstream out("gamma_result.txt", std::ios::app);
		if (out.is_open()) {
			out << std::fixed << std::setprecision(6);
			out << file_name << '\t' << argv[2];
			for (int r = 0; r < mpi_size; r++)
				out << '\t' << all_total_exec[(size_t)r];
			out << '\t' << global_match << '\n';
			out.close();
		} else {
			std::cerr << "Failed to open gamma_result.txt for writing" << std::endl;
		}
	}

	data_graph.clean();
	MPI_Finalize();
	return 0;
}
