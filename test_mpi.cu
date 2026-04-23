#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "src/buffer.cuh"
#include "src/graph.h"
#include "src/gpu_match.cuh"
#include "src/job_queue.h"
#include "src/missing_tree.cuh"
#include "src/pattern.h"

namespace {

constexpr int kTagClaim = 71001;
constexpr int kTagGrant = 71002;
constexpr int kTagWorkerDone = 49996;

int getenv_int(const char* key, int fallback) {
  if (const char* s = std::getenv(key)) {
    try {
      return std::stoi(s);
    } catch (...) {
      return fallback;
    }
  }
  return fallback;
}

int local_mpi_rank() {
  int lr = getenv_int("OMPI_COMM_WORLD_LOCAL_RANK", -1);
  if (lr >= 0) return lr;
  lr = getenv_int("MV2_COMM_WORLD_LOCAL_RANK", -1);
  if (lr >= 0) return lr;
  lr = getenv_int("SLURM_LOCALID", -1);
  if (lr >= 0) return lr;
  return 0;
}

void mpi_rank0_service(int64_t& next_dynamic, graph_node_t N, int mpi_size, std::ofstream* plog) {
  int flag = 0;
  MPI_Status st;
  MPI_Iprobe(MPI_ANY_SOURCE, kTagClaim, MPI_COMM_WORLD, &flag, &st);
  while (flag) {
    const int src = st.MPI_SOURCE;
    int want = 0;
    MPI_Recv(&want, 1, MPI_INT, src, kTagClaim, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int64_t lo = next_dynamic;
    int64_t hi = std::min<int64_t>((int64_t)N, lo + want);
    next_dynamic = hi;
    int64_t pkt[2] = {lo, hi};
    MPI_Send(pkt, 2, MPI_LONG_LONG, src, kTagGrant, MPI_COMM_WORLD);
    if (plog && plog->good()) {
      *plog << "mpi grant to rank " << src << " interval [" << lo << "," << hi << ")";
      if (lo >= hi) *plog << " (empty, pool exhausted)";
      *plog << "\n";
      plog->flush();
    }
    MPI_Iprobe(MPI_ANY_SOURCE, kTagClaim, MPI_COMM_WORLD, &flag, &st);
  }
}

void claim_dynamic_batch(int mpi_rank, int mpi_size, int64_t& counter_rank0, graph_node_t N,
                         graph_node_t static_split, int batch_jobs, graph_node_t& out_lo,
                         graph_node_t& out_hi, std::ofstream* plog) {
  if (static_split >= N || batch_jobs <= 0) {
    out_lo = out_hi = 0;
    return;
  }
  if (mpi_size <= 1) {
    int64_t lo = counter_rank0;
    int64_t hi = std::min<int64_t>((int64_t)N, lo + (int64_t)batch_jobs);
    counter_rank0 = hi;
    out_lo = (graph_node_t)lo;
    out_hi = (graph_node_t)hi;
    return;
  }
  if (mpi_rank == 0) {
    int64_t lo = counter_rank0;
    int64_t hi = std::min<int64_t>((int64_t)N, lo + (int64_t)batch_jobs);
    counter_rank0 = hi;
    out_lo = (graph_node_t)lo;
    out_hi = (graph_node_t)hi;
    if (plog && plog->good()) {
      *plog << "rank0 local dynamic claim [" << out_lo << "," << out_hi << ")\n";
      plog->flush();
    }
    return;
  }
  MPI_Send(&batch_jobs, 1, MPI_INT, 0, kTagClaim, MPI_COMM_WORLD);
  int64_t pkt[2];
  MPI_Recv(pkt, 2, MPI_LONG_LONG, 0, kTagGrant, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  out_lo = (graph_node_t)pkt[0];
  out_hi = (graph_node_t)pkt[1];
  if (plog && plog->good()) {
    *plog << "rank " << mpi_rank << " recv dynamic [" << out_lo << "," << out_hi << ")\n";
    plog->flush();
  }
}

void push_job_interval(JobQueue* gpu_q, JobQueue& host_mirror, graph_node_t lo, graph_node_t hi,
                       double interval_preempt_ratio, int* managed_preempt_flag) {
  cudaMemcpy(&host_mirror, gpu_q, sizeof(JobQueue), cudaMemcpyDeviceToHost);
  host_mirror.work_lo = lo;
  host_mirror.work_hi = hi;
  host_mirror.cur = lo;
  host_mirror.preempt_latched = 0;
  host_mirror.preempt_flag = managed_preempt_flag;
  if (hi > lo) {
    const double span = (double)(hi - lo);
    graph_node_t delta = (graph_node_t)(span * interval_preempt_ratio);
    if (delta >= hi - lo) delta = (hi - lo > 0) ? (hi - lo - 1) : 0;
    host_mirror.preempt_at = lo + delta;
  } else {
    host_mirror.preempt_at = lo;
  }
  *managed_preempt_flag = 0;
  cudaMemcpy(gpu_q, &host_mirror, sizeof(JobQueue), cudaMemcpyHostToDevice);
}

static inline graph_node_t compute_preempt_at(graph_node_t lo, graph_node_t hi,
                                               double interval_preempt_ratio) {
  if (hi <= lo) return lo;
  const double span = (double)(hi - lo);
  graph_node_t delta = (graph_node_t)(span * interval_preempt_ratio);
  if (delta >= hi - lo) delta = (hi - lo > 0) ? (hi - lo - 1) : 0;
  return lo + delta;
}

void set_jobqueue_next_interval(JobQueue* gpu_q, graph_node_t next_lo, graph_node_t next_hi,
                                double interval_preempt_ratio, cudaStream_t ctl_stream) {
  const graph_node_t preempt_at = compute_preempt_at(next_lo, next_hi, interval_preempt_ratio);
  const int zero = 0;
  const int one = 1;
  // IMPORTANT: must be async and on a non-legacy stream, otherwise a persistent kernel in another stream
  // can be implicitly synchronized causing a deadlock.
  cudaMemcpyAsync((char*)gpu_q + offsetof(JobQueue, next_ready), &zero, sizeof(int), cudaMemcpyHostToDevice,
                  ctl_stream);
  cudaMemcpyAsync((char*)gpu_q + offsetof(JobQueue, next_lo), &next_lo, sizeof(graph_node_t), cudaMemcpyHostToDevice,
                  ctl_stream);
  cudaMemcpyAsync((char*)gpu_q + offsetof(JobQueue, next_hi), &next_hi, sizeof(graph_node_t), cudaMemcpyHostToDevice,
                  ctl_stream);
  cudaMemcpyAsync((char*)gpu_q + offsetof(JobQueue, next_preempt_at), &preempt_at, sizeof(graph_node_t),
                  cudaMemcpyHostToDevice, ctl_stream);
  cudaMemcpyAsync((char*)gpu_q + offsetof(JobQueue, next_ready), &one, sizeof(int), cudaMemcpyHostToDevice,
                  ctl_stream);
}

void set_jobqueue_done(JobQueue* gpu_q, cudaStream_t ctl_stream) {
  const int one = 1;
  const int zero = 0;
  cudaMemcpyAsync((char*)gpu_q + offsetof(JobQueue, done), &one, sizeof(int), cudaMemcpyHostToDevice, ctl_stream);
  cudaMemcpyAsync((char*)gpu_q + offsetof(JobQueue, next_ready), &zero, sizeof(int), cudaMemcpyHostToDevice,
                  ctl_stream);
}

void reset_flags_for_kernel(SyncFlags2* d_flags) {
  d_flags->matching_over = 0;
  d_flags->cpu_process_over = 0;
  d_flags->global_update_token = 0;
  for (int b = 0; b < GRID_DIM; ++b) {
    d_flags->block_update_token[b] = 0;
    d_flags->process_vertices_flag[b] = 0;
    for (int i = 0; i < CHUNKS_PER_BLOCK; ++i) {
      d_flags->queue_flushed_flags[b][i] = 0;
    }
  }
}

}  // namespace

namespace {

bool arg_is_auto_mode(const char* s) {
  if (!s || !*s) return false;
  std::string a(s);
  for (char& c : a) c = (char)std::tolower((unsigned char)c);
  return a == "auto" || a == "a10" || a == "slice" || a == "slices";
}

}  // namespace

int main(int argc, char* argv[]) {
  int prov = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);

  int mpi_rank = 0;
  int mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (argc < 3) {
    if (mpi_rank == 0) {
      std::cerr << "用法: " << (argv[0] ? argv[0] : "test_mpi")
                << " <graph> <pattern> [static_ratio] [interval_preempt_ratio] [dynamic_batch] [dynamic_slices]\n"
                << "  dynamic_batch:\n"
                << "    缺省 或 auto — 动态区均分 dynamic_slices 段，批大小 = ceil(动态任务数/dynamic_slices)\n"
                << "    正整数 — 每批固定这么多 job（手动批大小）\n";
    }
    MPI_Finalize();
    return 1;
  }

  const double static_ratio = (argc >= 4) ? std::stod(argv[3]) : 0.8;
  const double interval_preempt_ratio = (argc >= 5) ? std::stod(argv[4]) : 0.8;
  bool dynamic_batch_adaptive = true;
  int dynamic_batch_manual = 0;
  if (argc >= 6) {
    const char* db = argv[5];
    if (arg_is_auto_mode(db)) {
      dynamic_batch_adaptive = true;
    } else {
      char* end = nullptr;
      const long v = std::strtol(db, &end, 10);
      if (!end || *end != '\0' || v <= 0 || v > INT_MAX) {
        if (mpi_rank == 0) {
          std::cerr << "dynamic_batch 须为正整数，或 auto（均分 dynamic_slices）\n";
        }
        MPI_Finalize();
        return 1;
      }
      dynamic_batch_adaptive = false;
      dynamic_batch_manual = (int)v;
    }
  }

  int dynamic_slices = 10;
  if (argc >= 7) {
    const char* ds = argv[6];
    char* end = nullptr;
    const long v = std::strtol(ds, &end, 10);
    if (!end || *end != '\0' || v <= 0 || v > INT_MAX) {
      if (mpi_rank == 0) {
        std::cerr << "dynamic_slices 须为正整数\n";
      }
      MPI_Finalize();
      return 1;
    }
    dynamic_slices = (int)v;
  }
  const float edge_ratio = 0.6f;

  int dev_count = 0;
  cudaGetDeviceCount(&dev_count);
  if (dev_count <= 0) {
    if (mpi_rank == 0) std::cerr << "未发现 CUDA 设备\n";
    MPI_Finalize();
    return 1;
  }
  const int lid = local_mpi_rank();
  cudaSetDevice(lid % dev_count);

  std::ostringstream logname;
  logname << "pumatch_preempt_rank_" << mpi_rank << ".log";
  std::ofstream preempt_log(logname.str(), std::ios::out | std::ios::app);
  preempt_log << "=== run start rank " << mpi_rank << " static_ratio=" << static_ratio
              << " interval_preempt_ratio=" << interval_preempt_ratio
              << " dynamic_batch_mode=" << (dynamic_batch_adaptive ? "slices" : "manual")
              << (dynamic_batch_adaptive ? (" dynamic_slices=" + std::to_string(dynamic_slices)) : "")
              << (dynamic_batch_adaptive ? "" : (" manual_batch=" + std::to_string(dynamic_batch_manual)))
              << " ===\n";
  preempt_log.flush();

  GraphPreprocessor g(argv[1]);
  GraphPreprocessor g2(argv[1]);
  PatternPreprocessor p(argv[2]);

  graph_edge_t edge_limit_num = (graph_edge_t)((double)g2.g.nedges * (double)edge_ratio);

  int idx = -1;
  for (int i = 0; i < g2.g.nnodes + 1; ++i) {
    if (g2.g.rowptr[i] > edge_limit_num) {
      idx = i;
      break;
    }
  }

  int n = g2.g.nnodes + 1;
  int half = idx;
  auto* ptr = g2.g.rowptr;
  graph_edge_t last = ptr[half - 1];
  std::fill(ptr + half, ptr + n, last);

  Graph* gpu_graph = g2.to_gpu();
  Graph* gpu_graph_managed = g.to_gpu_managed();
  Pattern* gpu_pattern = p.to_gpu();

  JobQueuePreprocessor queue = JobQueuePreprocessor(g.g, p);
  std::cout << "rank " << mpi_rank << " jobs H2D (length=" << queue.q.length << ") ..." << std::endl
            << std::flush;
  JobQueue* gpu_queue = queue.to_gpu();
  std::cout << "rank " << mpi_rank << " job queue on GPU ready" << std::endl << std::flush;
  JobQueue host_q_mirror{};
  cudaMemcpy(&host_q_mirror, gpu_queue, sizeof(JobQueue), cudaMemcpyDeviceToHost);

  int* preempt_flag = nullptr;
  cudaMallocManaged(&preempt_flag, sizeof(int));
  *preempt_flag = 0;
  host_q_mirror.preempt_flag = preempt_flag;
  cudaMemcpy(gpu_queue, &host_q_mirror, sizeof(JobQueue), cudaMemcpyHostToDevice);

  const graph_node_t N = queue.q.length;
  const graph_node_t static_split = (graph_node_t)((double)N * static_ratio);
  /* 动态区 [static_split, N) 约占总任务的后 (1-static_ratio) */
  const int64_t dynamic_total64 = (int64_t)N - (int64_t)static_split;
  int dynamic_batch = 0;
  if (dynamic_total64 > 0) {
    if (dynamic_batch_adaptive) {
      dynamic_batch = (int)((dynamic_total64 + dynamic_slices - 1) / dynamic_slices);
    } else {
      dynamic_batch = dynamic_batch_manual;
    }
  }
  if (mpi_rank == 0) {
    std::cout << "dynamic_jobs=" << dynamic_total64;
    if (dynamic_batch_adaptive) {
      std::cout << " mode=slices slices=" << dynamic_slices << " dynamic_batch=" << dynamic_batch;
    } else {
      std::cout << " mode=manual dynamic_batch=" << dynamic_batch;
    }
    std::cout << std::endl;
  }
  preempt_log << "dynamic_total=" << dynamic_total64 << " dynamic_batch_mode="
              << (dynamic_batch_adaptive ? "slices" : "manual");
  if (dynamic_batch_adaptive) {
    preempt_log << " slices=" << dynamic_slices;
  } else {
    preempt_log << " manual_batch=" << dynamic_batch_manual;
  }
  preempt_log << " dynamic_batch=" << dynamic_batch << "\n";
  preempt_log.flush();

  const graph_node_t rank_static_lo = (graph_node_t)(((int64_t)mpi_rank * (int64_t)static_split) / (int64_t)mpi_size);
  const graph_node_t rank_static_hi = (graph_node_t)(((int64_t)(mpi_rank + 1) * (int64_t)static_split) / (int64_t)mpi_size);

  CallStack* gpu_callstack;

  graph_node_t* slot_storage;
  cudaMalloc(&slot_storage, sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);

  std::vector<CallStack> stk(NWARPS_TOTAL);
  for (int i = 0; i < NWARPS_TOTAL; i++) {
    auto& s = stk[i];
    memset(s.iter, 0, sizeof(s.iter));
    memset(s.slot_size, 0, sizeof(s.slot_size));
    s.slot_storage =
        (graph_node_t(*)[UNROLL][GRAPH_DEGREE])((char*)slot_storage +
                                                i * sizeof(graph_node_t) * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
  }
  cudaMalloc(&gpu_callstack, NWARPS_TOTAL * sizeof(CallStack));
  cudaMemcpy(gpu_callstack, stk.data(), sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice);

  size_t* gpu_res;
  cudaMalloc(&gpu_res, sizeof(size_t) * NWARPS_TOTAL);

  int* idle_warps;
  cudaMalloc(&idle_warps, sizeof(int) * GRID_DIM);
  int* idle_warps_count;
  cudaMalloc(&idle_warps_count, sizeof(int));
  int* global_mutex;
  cudaMalloc(&global_mutex, sizeof(int) * GRID_DIM);
  int* write_block_flags;
  cudaMalloc(&write_block_flags, sizeof(int) * GRID_DIM);

  SyncFlags2* d_flags;
  cudaMallocManaged(&d_flags, sizeof(SyncFlags2), cudaMemAttachGlobal);
  d_flags->matching_over = 0;
  d_flags->cpu_process_over = 0;
  d_flags->global_update_token = 0;
  for (int b = 0; b < GRID_DIM; ++b) {
    d_flags->block_update_token[b] = 0;
    d_flags->process_vertices_flag[b] = 0;
    for (int i = 0; i < CHUNKS_PER_BLOCK; ++i) {
      d_flags->queue_flushed_flags[b][i] = 0;
    }
  }

  int* cpu_visible_queue_raw;
  cudaHostAlloc((void**)&cpu_visible_queue_raw, GRID_DIM * CHUNKS_PER_BLOCK * (MAX_QUEUE_SIZE / 2) * sizeof(int),
                cudaHostAllocMapped);
  int (*cpu_visible_queue)[CHUNKS_PER_BLOCK][MAX_QUEUE_SIZE / 2] =
      (int (*)[CHUNKS_PER_BLOCK][MAX_QUEUE_SIZE / 2])cpu_visible_queue_raw;

  std::cout << "rank " << mpi_rank << " create_buffer_from_graph ..." << std::endl << std::flush;
  Buffer host_buffer = create_buffer_from_graph(g.g, idx - 1);
  std::cout << "rank " << mpi_rank << " host buffer ready" << std::endl << std::flush;

  size_t job_memory_usage = sizeof(Job) * queue.q.length;
  size_t stack_memory_usage = sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE;
  size_t partial_graph_memory_usage =
      sizeof(bitarray32) * g.g.nnodes + sizeof(graph_node_t) * g2.g.rowptr[g.g.nnodes] +
      sizeof(graph_edge_t) * (g.g.nnodes + 1);
  size_t buffer_memory_usage = MAX_PACKAGE_EACH_BUFFER * MAX_PACKAGE_SIZE * sizeof(int);
  size_t other_memory_usage = sizeof(CallStack) * NWARPS_TOTAL + sizeof(size_t) * NWARPS_TOTAL +
                              sizeof(int) * GRID_DIM + sizeof(int) + sizeof(int) * GRID_DIM + sizeof(Pattern);
  size_t UM_memory_usage = (size_t)1 * 1024 * 1024 * 1024;
  size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t occupy_size = (size_t)40960 * 1024 * 1024 - job_memory_usage - stack_memory_usage - other_memory_usage -
                       UM_memory_usage - partial_graph_memory_usage - buffer_memory_usage;

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStream_t ctl_stream;
  cudaStreamCreateWithFlags(&ctl_stream, cudaStreamNonBlocking);
  cudaEvent_t init_done;
  cudaEventCreate(&init_done);

  Buffer* d_buffer = buffer_init_on_gpu(host_buffer, stream1);
  graph_node_t* occupied_space;
  cudaMalloc(&occupied_space, occupy_size);
  cudaMemset(occupied_space, 0, occupy_size);
  cudaEventRecord(init_done, stream1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int64_t dynamic_next_index = static_split;

  unsigned long long grand_total = 0;
  bool have_kernel_wall = false;
  std::chrono::steady_clock::time_point kernel_wall_t0{};
  std::chrono::steady_clock::time_point kernel_wall_t1{};

  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "rank " << mpi_rank << " past MPI barrier, start CPU helper threads + kernels" << std::endl
            << std::flush;

  std::vector<std::thread> cpu_threads;
  for (int tid = 0; tid < GRID_DIM; ++tid) {
    cpu_threads.emplace_back([&, tid] {
      cudaEventSynchronize(init_done);
      cpu_process_multiple_queue(g, &host_buffer, d_buffer, cpu_visible_queue, CHUNKS_PER_BLOCK, d_flags, tid,
                                 idx - 1, stream2);
    });
  }

  // Persistent kernel: launch once, then feed next intervals into job_queue.
  reset_flags_for_kernel(d_flags);
  cudaMemset(idle_warps, 0, sizeof(int) * GRID_DIM);
  cudaMemset(idle_warps_count, 0, sizeof(int));
  cudaMemset(global_mutex, 0, sizeof(int) * GRID_DIM);
  cudaMemset(write_block_flags, 0, sizeof(int) * GRID_DIM);

  bool local_done_on_gpu = false;
  graph_node_t init_lo = rank_static_lo;
  graph_node_t init_hi = rank_static_hi;
  if (init_lo >= init_hi) {
    // Static slice is empty for this rank: grab the first dynamic batch upfront.
    graph_node_t nl, nh;
    claim_dynamic_batch(mpi_rank, mpi_size, dynamic_next_index, N, static_split, dynamic_batch, nl, nh, &preempt_log);
    if (nl < nh && nh <= N) {
      init_lo = nl;
      init_hi = nh;
    } else {
      init_lo = init_hi = 0;
      set_jobqueue_done(gpu_queue, ctl_stream);
      local_done_on_gpu = true;
    }
  }

  push_job_interval(gpu_queue, host_q_mirror, init_lo, init_hi, interval_preempt_ratio, preempt_flag);
  preempt_log << "rank " << mpi_rank << " persistent set current [" << init_lo << "," << init_hi
              << ") preempt_at=" << host_q_mirror.preempt_at << "\n";
  preempt_log.flush();

  have_kernel_wall = true;
  kernel_wall_t0 = std::chrono::steady_clock::now();

  cudaEventRecord(start, stream1);
  _parallel_match<<<GRID_DIM, BLOCK_DIM, 0, stream1>>>(
      gpu_graph, gpu_graph_managed, gpu_pattern, gpu_callstack, gpu_queue, gpu_res, idle_warps, idle_warps_count,
      global_mutex, d_buffer, d_flags, write_block_flags, (int*)cpu_visible_queue, idx - 1);
  cudaEventRecord(stop, stream1);
  while (cudaEventQuery(stop) == cudaErrorNotReady) {
    // rank 0 must keep servicing claim/grant messages while other ranks might still be preempting.
    if (mpi_size > 1 && mpi_rank == 0) {
      mpi_rank0_service(dynamic_next_index, N, mpi_size, &preempt_log);
    }

    if (__atomic_load_n(preempt_flag, __ATOMIC_ACQUIRE)) {
      if (!local_done_on_gpu) {
        // 先处理其它 rank 的 claim，再本地领批，避免 rank0 先把指针推到 N 而其它 rank 仍以为有货
        if (mpi_size > 1 && mpi_rank == 0) {
          mpi_rank0_service(dynamic_next_index, N, mpi_size, &preempt_log);
        }

        graph_node_t nl, nh;
        claim_dynamic_batch(mpi_rank, mpi_size, dynamic_next_index, N, static_split, dynamic_batch, nl, nh,
                            &preempt_log);
        if (nl < nh && nh <= N) {
          set_jobqueue_next_interval(gpu_queue, nl, nh, interval_preempt_ratio, ctl_stream);
          preempt_log << "rank " << mpi_rank << " preempt set next [" << nl << "," << nh << ")\n";
        } else {
          set_jobqueue_done(gpu_queue, ctl_stream);
          local_done_on_gpu = true;
          preempt_log << "rank " << mpi_rank << " preempt set done (no next batch)\n";
        }
        preempt_log.flush();
      }
      __atomic_store_n(preempt_flag, 0, __ATOMIC_RELEASE);
    }

    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }

  cudaEventSynchronize(stop);
  kernel_wall_t1 = std::chrono::steady_clock::now();

  // Read total matches accumulated by the persistent kernel.
  size_t res_host[NWARPS_TOTAL];
  cudaMemcpy(res_host, gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost);
  unsigned long long raw_cumulative = 0;
  for (int i = 0; i < NWARPS_TOTAL; i++) raw_cumulative += res_host[i];
  if (!LABELED) raw_cumulative *= (unsigned long long)p.PatternMultiplicity;
  grand_total = raw_cumulative;

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "rank " << mpi_rank << " persistent kernel done kernel_ms=" << ms
            << " total_matches=" << raw_cumulative << "\n";

  // Cover any narrow gap between kernel completion and subsequent MPI servicing.
  if (mpi_size > 1 && mpi_rank == 0) {
    mpi_rank0_service(dynamic_next_index, N, mpi_size, &preempt_log);
  }

  /*
   * rank0 会先耗尽动态池并退出 for，但其它 rank 可能仍在跑最后一段 kernel，随后还要
   * MPI_Recv 领空批。若 rank0 已进 MPI_Reduce 则无法再 mpi_rank0_service → rank1 永久
   * 卡在 Recv，matching_over 无法置位，64 路 CPU 辅助线程空转占满核。
   * 非 0 rank 退出动态循环后先发 WorkerDone；rank0 收齐 (size-1) 条并持续 service 后再收尾。
   */
  if (mpi_size > 1 && mpi_rank != 0) {
    int ack = 1;
    /* 同步发送：必须等 rank0 进入 Recv 后才完成，避免缓冲导致 root 误判进度 */
    MPI_Ssend(&ack, 1, MPI_INT, 0, kTagWorkerDone, MPI_COMM_WORLD);
    preempt_log << "rank " << mpi_rank << " dynamic loop exit, sent WorkerDone to root\n";
    preempt_log.flush();
  }

  if (mpi_size > 1 && mpi_rank == 0) {
    std::vector<int> worker_done_seen(mpi_size, 0);
    worker_done_seen[0] = 1;
    int need = mpi_size - 1;
    for (;;) {
      mpi_rank0_service(dynamic_next_index, N, mpi_size, &preempt_log);
      int wflag = 0;
      MPI_Status wst;
      MPI_Iprobe(MPI_ANY_SOURCE, kTagWorkerDone, MPI_COMM_WORLD, &wflag, &wst);
      if (wflag) {
        const int src = wst.MPI_SOURCE;
        if (src > 0 && src < mpi_size && !worker_done_seen[src]) {
          int ack = 0;
          MPI_Recv(&ack, 1, MPI_INT, src, kTagWorkerDone, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          worker_done_seen[src] = 1;
          --need;
          preempt_log << "rank0 WorkerDone from rank " << src << ", need_left=" << need << "\n";
          preempt_log.flush();
        } else {
          int ack = 0;
          MPI_Recv(&ack, 1, MPI_INT, src, kTagWorkerDone, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          preempt_log << "rank0 dropped stray/extra WorkerDone from rank " << src << "\n";
          preempt_log.flush();
        }
      }
      if (need <= 0) break;
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    mpi_rank0_service(dynamic_next_index, N, mpi_size, &preempt_log);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  __atomic_store_n(&d_flags->matching_over, 1, __ATOMIC_RELEASE);
  for (auto& th : cpu_threads) th.join();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaEventDestroy(init_done);
  cudaStreamDestroy(ctl_stream);

  double wall_ms = 0.0;
  if (have_kernel_wall) {
    wall_ms =
        std::chrono::duration<double>(kernel_wall_t1 - kernel_wall_t0).count() * 1000.0;
  }

  unsigned long long local_matches = grand_total;
  unsigned long long global_matches = 0;
  MPI_Reduce(&local_matches, &global_matches, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  std::vector<double> all_wall_ms;
  if (mpi_rank == 0) all_wall_ms.resize((size_t)mpi_size);
  MPI_Gather(&wall_ms, 1, MPI_DOUBLE, mpi_rank == 0 ? all_wall_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    std::cout << "total_matches=" << global_matches << std::endl;
    std::ofstream pr("pu_result.txt", std::ios::out | std::ios::app);
    if (pr) {
      namespace fs = std::filesystem;
      pr << fs::path(argv[1]).filename().string() << '\t'
         << fs::path(argv[2]).filename().string();
      for (int r = 0; r < mpi_size; ++r) {
        pr << ' ' << all_wall_ms[(size_t)r];
      }
      pr << ' ' << global_matches << '\n';
    }
  }

  cudaFree(preempt_flag);
  MPI_Finalize();
  return 0;
}
