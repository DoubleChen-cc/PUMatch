#pragma once

#include "graph.h"
#include "pattern.h"
#include "callstack.h"
#include "job_queue.h"
#include "buffer.cuh"
#include "common.cuh"
#include "missing_tree.cuh"

__global__ void _parallel_match(Graph* dev_graph, Graph* dev_graph_managed, Pattern* dev_pattern,
    CallStack* dev_callstack, JobQueue* job_queue, size_t* res,
    int* idle_warps, int* idle_warps_count, int* global_mutex, Buffer* d_buffer, SyncFlags2* flags, int* write_block_flags, int* cpu_visible_queue, int idx);
