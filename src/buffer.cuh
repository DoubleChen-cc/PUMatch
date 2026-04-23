#pragma once
#include "config.h"
#include "cuda.h"
#include "graph.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

struct IndexEntry {
    int p_index;
    int v_index;    
    int version;  
};
struct PVindexEntry{
    int p_index;
    int v_index;
};

struct NeighborEntry{
    graph_node_t* neighbor_ptr;
    graph_node_t neighbor_num;
    int p_index;
};
struct Package
{
    graph_node_t* nodes = nullptr;
    graph_node_t* rowptr = nullptr;
    graph_node_t* colidx = nullptr;
    int node_num = 0;
    
    int write_lock = 0;
    
    int warp_read_count = 0;
};

struct CommunicationControl {
    int update_request;  
    int gpu_ready;       
    int transfer_complete; 
    int m;                
    int package_indices[MAX_PACKAGES]; 
    volatile int signal;           
};




struct LRUBucketLock {
    int lock[LRU_BUCKETS];
};

struct Buffer{
    Package* packages = nullptr;
    int package_num;
    IndexEntry* index_queue;
    graph_node_t nnodes;
    CommunicationControl* comm;
    int32_t* lru_prev;
    int32_t* lru_next;
    
    int32_t* lru_bucket_heads = nullptr;  
    int32_t* lru_bucket_tails = nullptr;  

    LRUBucketLock* lru_bucket_locks = nullptr; 
};















struct SyncFlags {
    int queue0_flushed;  
    int queue1_flushed;  
    int timer_triggered; 
    int matching_over;
    int queue_manager_over;
    int update_buffer_over;
    int cpu_process_over;
};

struct SyncFlags2 {
    int matching_over;
    int cpu_process_over;
    int global_update_token;
    int process_vertices_flag[GRID_DIM];  
    int queue_flushed_flags[GRID_DIM][CHUNKS_PER_BLOCK];  
    int block_update_token[GRID_DIM];
};


struct RequestQueue {
    int* data;               
    int head;       
    int tail;       
    int capacity;            
    int is_active; 
    unsigned long long last_switch_time;  
    int lock;
};


struct DoubleQueueManager {
    RequestQueue queues[2];  
    int active_idx; 
    unsigned long long flush_interval_cycles;  
    unsigned long long clock_freq;  
};


struct BlockQueue {
    int  head;
    int  tail;              
    int  data[MAX_QUEUE_SIZE]; 
    int  dedup_table[BLOCK_DEDUP_SIZE]; 
    int  dedup_init;                   
};


__device__ __forceinline__ int dedup_hash(int v) {
    
    unsigned int x = static_cast<unsigned int>(v);
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return static_cast<int>(x & (BLOCK_DEDUP_SIZE - 1));
}


__device__ __forceinline__ bool dedup_allow_and_set(int v, int* table) {
    int h = dedup_hash(v);
    
    for (int step = 0; step < BLOCK_DEDUP_PROBE; ++step) {
        int pos = (h + step) & (BLOCK_DEDUP_SIZE - 1);
        int old = atomicCAS(&table[pos], -1, v);
        if (old == v) {
            
            return false;
        }
        if (old == -1) {
            
            return true;
        }
        
    }
    
    return true;
}

 template<typename DATA_T, typename SIZE_T>
  __forceinline__ __device__ int binary_search(DATA_T* set2, SIZE_T set2_size, DATA_T target);
  
static inline __device__ void acquire_queue_lock(volatile int* lock) {
    
    while (atomicCAS(const_cast<int*>(lock), 0, 1) != 0);
}


static inline __device__ void release_queue_lock(volatile int* lock) {
    atomicExch(const_cast<int*>(lock), 0);
}
Buffer create_buffer_from_graph(Graph g, graph_node_t start_node);
Buffer create_buffer_from_graph2(Graph g, graph_node_t idx, int threshold);
Buffer create_buffer_from_graph3(Graph g, int* partitions);
Buffer* buffer_init_on_gpu(Buffer& hostBuffer,cudaStream_t stream);
DoubleQueueManager* init_queue_manager();
__device__ bool add_vertex(
    DoubleQueueManager* dq_manager,
    SyncFlags* flags,
    int vertex,
    int* cpu_visible_queue0,
    int* cpu_visible_queue1
);

__device__ bool add_vertex_block(int vertex,
    BlockQueue* sm_bq,
    int* cpu_visible_queue,
    int* write_block_flags,
    SyncFlags2* sync_flags); 
__device__ PVindexEntry read_vertex_index(int vertex_id, Buffer buffer);

__device__ void update_vertex_index(int vertex_id, int p_index, int v_index, Buffer buffer);

__device__ NeighborEntry get_neighbors_from_buffer(graph_node_t vid, Buffer buffer);

__device__ int switch_active_queue(DoubleQueueManager* dq_manager);

__device__ void copy_inactive_queue(DoubleQueueManager* dq_manager, int inactive_idx, 
                                   int* cpu_visible_buf, volatile int* flush_flag);

__device__ void queue_management(
    DoubleQueueManager* dq_manager,
    SyncFlags* flags,
    int* cpu_visible_queue0,  
    int* cpu_visible_queue1   
);


__device__ void lruAccess_bucket(Buffer* buf, int32_t pkgId) ;
__device__ int find_lowest_priority_packages(int m, int* result_indices, Buffer g_buffer);
__device__ int find_lowest_priority_packages_bucket(int m, int* out, Buffer buf);
__device__ void update_buffer(Buffer g_buffer,SyncFlags2* d_flags);

void copy_packages_to_gpu(Buffer* device_buffer, Buffer h_buffer, const int* package_indices, cudaStream_t stream);
void copy_packages_to_slots_async(Package* d_packages, const Buffer& h_buffer, const int* package_indices, cudaStream_t stream);

void cpu_process_double_queue(DoubleQueueManager* dq_manager, SyncFlags* d_flags, GraphPreprocessor g, Buffer* host_buffer, Buffer* device_buffer, int* cpu_visible_queue0, int* cpu_visible_queue1, cudaStream_t stream);

void cpu_process_multiple_queue(GraphPreprocessor g, Buffer* host_buffer, Buffer* device_buffer,  int (*cpu_visible_queues)[CHUNKS_PER_BLOCK][MAX_QUEUE_SIZE/2], int queue_num, SyncFlags2* d_flags, int tid, int start_node, cudaStream_t stream);

Buffer cpu_process_all_vertices(GraphPreprocessor g, int (*cpu_visible_queue)[MAX_QUEUE_SIZE / 2], int* queue_flushed_flags, int queue_num, Buffer* host_buffer,int start_node);

Buffer process_vertices(int* cpu_visible_queue, int node_num, GraphPreprocessor g);


long long get_total_cpu_m();













inline unsigned long long get_gpu_clock_freq() {
    int clockRatekHz;  
    cudaDeviceGetAttribute(&clockRatekHz, cudaDevAttrClockRate, 0);
    unsigned long long clockRate = static_cast<unsigned long long>(clockRatekHz) * 1000;
    return clockRate;
}

inline __device__ unsigned long long get_current_cycle_count() {
    unsigned long long cycle_count;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(cycle_count));
    return cycle_count;
}

