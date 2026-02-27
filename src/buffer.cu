#include "buffer.cuh"
#include <cstdint>




static constexpr std::uint64_t CPU_PACKAGE_BYTES =
    static_cast<std::uint64_t>(sizeof(graph_node_t)) *
    static_cast<std::uint64_t>(MAX_NODE_NUM_PER_PACKAGE + (MAX_NODE_NUM_PER_PACKAGE + 1) + MAX_PACKAGE_SIZE);

static long long g_total_cpu_m = 0;
static std::uint64_t g_total_cpu_bytes = 0;
static int g_cpu_threads_finished = 0;


long long get_total_cpu_m() {
    
    return __atomic_load_n(&g_total_cpu_m, __ATOMIC_ACQUIRE);
}
#include <set>
#include <cuda/atomic>
#include <queue>
#include <algorithm>
#include <vector>

__device__ bool acquire_read_lock_warp(Package* pkg) {
    
    const bool is_representative = (threadIdx.x % 32 == 0);
    
    bool read_lock_acquired = false;

    
    if (is_representative) {
        
        while (__nv_atomic_load_n(&pkg->write_lock, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)!=0){
            __nanosleep(1);
        }
        atomicAdd(&pkg->warp_read_count, 1);
        
        
        
        
        
        read_lock_acquired = true;
    }

    
    read_lock_acquired = __shfl_sync(0xFFFFFFFF, read_lock_acquired, 0);
    
    
    if (!read_lock_acquired) {
        assert(0 && "Warp read lock acquisition failed");
    }
    __threadfence();
    return read_lock_acquired;
}


__device__ void release_read_lock_warp(Package* pkg, bool read_lock_acquired) {
    const bool is_representative = (threadIdx.x % 32 == 0);

    
    if (is_representative && read_lock_acquired) {
        
        int old_count = atomicSub(&pkg->warp_read_count, 1);
        if (old_count <= 0) {
            
            atomicAdd(&pkg->warp_read_count, 1);
            
            
        }
    }

    
    __syncwarp();
    __threadfence();
}


__device__ void acquire_write_lock_warp(Package* pkg) {
    const bool is_representative = (threadIdx.x % 32 == 0);
    bool write_lock_acquired = false;

    
    if (is_representative) {
        
        while(__nv_atomic_load_n(&pkg->warp_read_count, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)>0){
            __nanosleep(1);
        }

        while (atomicCAS(&pkg->write_lock, 0, 1) != 0) {
            __nanosleep(1);
        } 
        
        
        
        
        
        write_lock_acquired = true;
    }

    
    write_lock_acquired = __shfl_sync(0xFFFFFFFF, write_lock_acquired, 0);
    
    if (!write_lock_acquired) {
        assert(0 && "Warp write lock acquisition failed");
    }
    __threadfence();
}


__device__ void release_write_lock_warp(Package* pkg) {
    const bool is_representative = (threadIdx.x % 32 == 0);

    
    if (is_representative) {
        atomicExch(&pkg->write_lock, 0);
    }

    
    __syncwarp();
    __threadfence();
}

__device__ void acquire_write_lock(Package* p){
    
    while (atomicCAS(&p->write_lock, 0, 1) != 0) {
        __nanosleep(1);  
    }
    
    while (atomicCAS(&p->warp_read_count, 0, 0) != 0) {
        __nanosleep(1);  
    }
}
__device__ void release_write_lock(Package* p)
{
    atomicExch(&p->write_lock, 0);  
}

__device__ void acquire_read_lock(Package* p)
{
    while (true) {
        
        while (atomicCAS(&p->write_lock, 0, 0) != 0) {
            __nanosleep(1);  
        }

        
        atomicAdd(&p->warp_read_count, 1);

        
        if (atomicCAS(&p->write_lock, 0, 0) == 0) {
            
            break;
        } else {
            
            atomicSub(&p->warp_read_count, 1);
            
        }
    }
}
__device__ void release_read_lock(Package* p)
{
    
    int old_count = atomicSub(&p->warp_read_count, 1);
    if (old_count <= 0) {
        
        atomicAdd(&p->warp_read_count, 1);
        
        
    }
}

  template<typename DATA_T, typename SIZE_T>
  __forceinline__ __device__
    int binary_search(DATA_T* set2, SIZE_T set2_size, DATA_T target) {
    if (set2_size <= 0) return -1;
    int mid;
    int low = 0;
    int high = set2_size - 1;
    while (low <= high) {
      mid = (low + high) / 2;
      if (target == set2[mid]) {
        return mid;
      }
      else if (target > set2[mid]) {
        low = mid + 1;
      }
      else {
        high = mid - 1;
      }
    }
    return -1;
  }


__device__ PVindexEntry read_vertex_index(int vertex_id, Buffer buffer) {
    int v1, v2, p_index, v_index;
    PVindexEntry pv_index;
    do {
        v1 = buffer.index_queue[vertex_id].version;
        
        if (v1 % 2 != 0) continue;
        pv_index.p_index = buffer.index_queue[vertex_id].p_index;
        pv_index.v_index = buffer.index_queue[vertex_id].v_index;
        v2 = buffer.index_queue[vertex_id].version;  
    } while (v1 != v2);  
    return pv_index;
}


__device__ void update_vertex_index(int vertex_id, int p_index, int v_index, Buffer buffer) {
    IndexEntry* entry = &buffer.index_queue[vertex_id];
    
    atomicAdd(&entry->version, 1);
    
    atomicExch(&entry->p_index ,p_index);
    atomicExch(&entry->v_index ,v_index);
    
    atomicAdd(&entry->version, 1);
}

__device__ NeighborEntry get_neighbors_from_buffer(graph_node_t vid, Buffer buffer){
    int tid = threadIdx.x % WARP_SIZE;
    PVindexEntry pv_index = read_vertex_index(vid, buffer);
    NeighborEntry ne;
    ne.neighbor_ptr = nullptr;
    ne.neighbor_num = 0;
    ne.p_index = -1;  
    int p_index = pv_index.p_index;
    int v_index = pv_index.v_index;
    bool lock_acquired = false;  
    if(p_index != -1 && v_index != -1){
        Package* pkg = &buffer.packages[p_index];
        
        
        
        
        
        
        
        
        
        
            graph_node_t row_idx = pkg->rowptr[v_index];
            
            ne.neighbor_ptr = &pkg->colidx[row_idx];
            ne.neighbor_num = pkg->rowptr[v_index+1] - pkg->rowptr[v_index];
            ne.p_index = p_index;
        
        
        
        
        
        
        
        

    }
    return ne;
}


Buffer* buffer_init_on_gpu(Buffer& h_buf, cudaStream_t stream)
{
    
    CommunicationControl* h_comm;
    cudaHostAlloc(&h_comm, sizeof(CommunicationControl) * GRID_DIM,
                  cudaHostAllocMapped | cudaHostAllocWriteCombined);
    memset(h_comm, 0, sizeof(CommunicationControl) * GRID_DIM);

    CommunicationControl* d_comm;
    
    for (int i = 0; i < GRID_DIM; ++i) {
        h_comm[i].update_request = false;
        h_comm[i].gpu_ready = false;
        h_comm[i].transfer_complete = false;
        h_comm[i].m = 0;
    }
    h_buf.comm = h_comm;  
    cudaHostGetDevicePointer(&d_comm, h_comm, 0);

    
    Buffer*  d_buf;
    Package* d_pkgs;
    cudaMallocManaged(&d_buf, sizeof(Buffer), cudaMemAttachGlobal);
    cudaMallocManaged(&d_pkgs, sizeof(Package) * MAX_PACKAGE_EACH_BUFFER,
                      cudaMemAttachGlobal);

    
    size_t totalNodes  = 0, totalRowptr = 0, totalColidx = 0;
    for (int i = 0; i < MAX_PACKAGE_EACH_BUFFER; ++i) {
        totalNodes  += MAX_NODE_NUM_PER_PACKAGE;
        totalRowptr += MAX_NODE_NUM_PER_PACKAGE + 1;
        totalColidx += MAX_PACKAGE_SIZE;
    }
    graph_node_t* pool;
    cudaMallocManaged(&pool,
                      (totalNodes + totalRowptr + totalColidx) * sizeof(graph_node_t),
                      cudaMemAttachGlobal);

    
    graph_node_t* off_nodes  = pool;
    graph_node_t* off_rowptr = off_nodes  + totalNodes;
    graph_node_t* off_colidx = off_rowptr + totalRowptr;
    cudaMallocManaged(&d_buf->lru_prev, MAX_PACKAGE_EACH_BUFFER * sizeof(int32_t),cudaMemAttachGlobal);
    cudaMallocManaged(&d_buf->lru_next, MAX_PACKAGE_EACH_BUFFER * sizeof(int32_t),cudaMemAttachGlobal);
    
    
    cudaMallocManaged(&d_buf->lru_bucket_heads, LRU_BUCKETS * sizeof(int32_t), cudaMemAttachGlobal);
    cudaMallocManaged(&d_buf->lru_bucket_tails, LRU_BUCKETS * sizeof(int32_t), cudaMemAttachGlobal);
    
    cudaMallocManaged(&d_buf->lru_bucket_locks, sizeof(LRUBucketLock), cudaMemAttachGlobal);

    
    
    
    for (int b = 0; b < LRU_BUCKETS; ++b) {
        d_buf->lru_bucket_heads[b] = -1;
        d_buf->lru_bucket_tails[b] = -1;
        if (d_buf->lru_bucket_locks)
            d_buf->lru_bucket_locks->lock[b] = 0;
    }
    
    int last_in_bucket[LRU_BUCKETS];
    for (int b = 0; b < LRU_BUCKETS; ++b)
        last_in_bucket[b] = -1;

    
    for (int i = 0; i < MAX_PACKAGE_EACH_BUFFER; ++i) {
        int bucket = i & (LRU_BUCKETS - 1);
        if (d_buf->lru_bucket_heads[bucket] == -1) {
            d_buf->lru_bucket_heads[bucket] = i;
        }
        if (last_in_bucket[bucket] != -1) {
            d_buf->lru_next[last_in_bucket[bucket]] = i;
            d_buf->lru_prev[i] = last_in_bucket[bucket];
        } else {
            d_buf->lru_prev[i] = -1;
        }
        d_buf->lru_next[i] = -1;
        last_in_bucket[bucket] = i;
        d_buf->lru_bucket_tails[bucket] = i;
    }

    for (int i = 0; i < MAX_PACKAGE_EACH_BUFFER; ++i) {
        d_pkgs[i].node_num        = 0;
        d_pkgs[i].write_lock      = 0;
        d_pkgs[i].warp_read_count = 0;
        d_pkgs[i].nodes           = off_nodes;
        d_pkgs[i].rowptr          = off_rowptr;
        d_pkgs[i].colidx          = off_colidx;
        
        
        off_nodes  += MAX_NODE_NUM_PER_PACKAGE;
        off_rowptr += MAX_NODE_NUM_PER_PACKAGE + 1;
        off_colidx += MAX_PACKAGE_SIZE;
    }
    
    
    

    
    d_buf->package_num = h_buf.package_num;
    d_buf->nnodes      = h_buf.nnodes;
    d_buf->packages    = d_pkgs;
    d_buf->comm        = d_comm;

    
    IndexEntry* d_idx;
    cudaMallocManaged(&d_idx, h_buf.nnodes * sizeof(IndexEntry),
                      cudaMemAttachGlobal);
    d_buf->index_queue = d_idx;

    for (int i = 0; i < h_buf.nnodes; ++i) {
        d_idx[i].p_index = -1;
        d_idx[i].v_index = -1;
        d_idx[i].version = 0;
    }


    
    
    
    
    for (int i = 0; i < h_buf.package_num; ++i) {
        Package* h_pkg = &h_buf.packages[i];
        Package* d_pkg = &d_buf->packages[i];
        
        cudaMemcpyAsync(d_pkg->nodes, h_pkg->nodes, sizeof(graph_node_t) * h_pkg->node_num, cudaMemcpyHostToDevice);
        
        cudaMemcpyAsync(d_pkg->rowptr, h_pkg->rowptr, sizeof(graph_node_t) * (h_pkg->node_num + 1), cudaMemcpyHostToDevice);
        
        int edges = 0;
        for (int k = 0; k < h_pkg->node_num; ++k) {
            graph_node_t deg = h_pkg->rowptr[k + 1] - h_pkg->rowptr[k];
            edges += deg;
        }
        cudaMemcpyAsync(d_pkg->colidx, h_pkg->colidx, sizeof(graph_node_t) * edges, cudaMemcpyHostToDevice);
        
        d_pkg->node_num = h_pkg->node_num;
        d_pkg->write_lock = h_pkg->write_lock;
        d_pkg->warp_read_count = h_pkg->warp_read_count;
    }
    
    cudaMemcpyAsync(d_buf->index_queue, h_buf.index_queue, sizeof(IndexEntry) * h_buf.nnodes, cudaMemcpyHostToDevice);

    return d_buf;
}


DoubleQueueManager* init_queue_manager() {
    
    DoubleQueueManager h_mgr = {};          
    DoubleQueueManager *d_mgr;
    unsigned long long clock_freq = get_gpu_clock_freq();
    unsigned long long interval_cycles = (clock_freq / 1000) * QUEUE_TIME_THRESHOLD_MS;
    h_mgr.active_idx           = 0;
    h_mgr.flush_interval_cycles = interval_cycles;
    h_mgr.clock_freq            = clock_freq;

    for (int q = 0; q < 2; ++q) {
        h_mgr.queues[q].capacity      = MAX_QUEUE_SIZE;
        h_mgr.queues[q].head          = 0;
        h_mgr.queues[q].tail          = 0;
        h_mgr.queues[q].is_active     = (q == 0);
        h_mgr.queues[q].last_switch_time = 0;
        h_mgr.queues[q].lock          = 0;
    }

    
    for (int q = 0; q < 2; ++q) {
        cudaMalloc(&h_mgr.queues[q].data, MAX_QUEUE_SIZE * sizeof(int));
        
        cudaMemset(h_mgr.queues[q].data, 0, MAX_QUEUE_SIZE * sizeof(int));
    }

    
    cudaMalloc(&d_mgr, sizeof(DoubleQueueManager));
    cudaMemcpy(d_mgr, &h_mgr, sizeof(DoubleQueueManager), cudaMemcpyHostToDevice);
    return d_mgr;
}



__device__ int switch_active_queue(DoubleQueueManager* dq_manager) {
    
    int old_active = __nv_atomic_load_n(&dq_manager->active_idx, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    int new_active = 1 - old_active;
    
    
    if (atomicCAS(&dq_manager->active_idx, old_active, new_active) == old_active) {
        
        atomicExch(&dq_manager->queues[old_active].is_active, 0);
        atomicExch(&dq_manager->queues[new_active].is_active, 1);
        
        
        atomicExch(&dq_manager->queues[new_active].last_switch_time, get_current_cycle_count());
        return old_active; 
    }
    
    
    
    return __nv_atomic_load_n(&dq_manager->active_idx, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
}


__device__ void copy_inactive_queue(DoubleQueueManager* dq_manager, int inactive_idx, 
                                   int* cpu_visible_buf, volatile int* flush_flag) {
    RequestQueue* queue = &dq_manager->queues[inactive_idx];
    
    
    
    int data_size = __nv_atomic_load_n(&queue->tail, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    if (data_size > 0 && !queue->is_active) { 
        
        for (int i = 0; i < data_size; i++) {
            cpu_visible_buf[i] = queue->data[i];
        }
        
        
        atomicExch(&queue->tail, 0);
        
        atomicExch(const_cast<int*>(flush_flag), 1);
        
    }
}

__device__ void wait_until_ms(unsigned long long target_ns)
{
    unsigned long long start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    unsigned long long ns_sleep = 32;          
 
    while (true) {
        unsigned long long now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (now - start >= target_ns) break;   

        
        __nanosleep(ns_sleep);
        if (ns_sleep < 1024) ns_sleep <<= 1;
    }
}


__device__ void queue_management(
    DoubleQueueManager* dq_manager,
    SyncFlags* flags,
    int* cpu_visible_queue0,  
    int* cpu_visible_queue1   
) {
    
    while(true){
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            int active_idx = __nv_atomic_load_n(&dq_manager->active_idx, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            RequestQueue* active_queue = &dq_manager->queues[active_idx];
            
            
            bool need_switch = false;

            unsigned long long current_cycles = get_current_cycle_count();
            unsigned long long elapsed_cycles = current_cycles - __nv_atomic_load_n(&active_queue->last_switch_time, __NV_ATOMIC_RELAXED,__NV_THREAD_SCOPE_DEVICE);
            if(elapsed_cycles >= dq_manager->flush_interval_cycles){
                need_switch = true;
                atomicExch(const_cast<int*>(&flags->timer_triggered), 1); 
            }
            
            
            if (need_switch && __nv_atomic_load_n(&active_queue->is_active, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE) ) {
                
                int old_active_idx = switch_active_queue(dq_manager);
                
                
                if (old_active_idx == 0) {
                    copy_inactive_queue(dq_manager, 0, cpu_visible_queue0, &flags->queue0_flushed);
                } else {
                    copy_inactive_queue(dq_manager, 1, cpu_visible_queue1, &flags->queue1_flushed);
                }
            }
            if (__nv_atomic_load_n(&flags->matching_over,__NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)){
                __nv_atomic_store_n(&flags->queue_manager_over, 1, __NV_ATOMIC_RELEASE,__NV_THREAD_SCOPE_SYSTEM);
                break;
            }
            
            __nanosleep(1024);
        }
        __syncwarp();
    }
    
}






__device__ __forceinline__ void lru_bucket_lock_acquire(volatile int* lock) {
    while (atomicCAS((int*)lock, 0, 1) != 0) {
        __nanosleep(64);
    }
}


__device__ __forceinline__ void lru_bucket_lock_release(volatile int* lock) {
    atomicExch((int*)lock, 0);
}


__device__ __forceinline__ int get_lru_bucket(int pkgId) {
    return pkgId & (LRU_BUCKETS - 1); 
}


__device__ void lruAccess_bucket(Buffer* buf, int32_t pkgId) {
    if (pkgId < 0 || pkgId >= MAX_PACKAGE_EACH_BUFFER) return;
    const bool rep = (threadIdx.x % 32 == 0);
    int bucket = get_lru_bucket(pkgId);

    int32_t* lru_bucket_heads = buf->lru_bucket_heads;
    int32_t* lru_bucket_tails = buf->lru_bucket_tails;
    LRUBucketLock* locks = buf->lru_bucket_locks;

    
    if (rep) {
        lru_bucket_lock_acquire(&locks->lock[bucket]);
    }
    __syncwarp();

    int32_t prev, next;
    
    if (rep) {
        prev = buf->lru_prev[pkgId];
        next = buf->lru_next[pkgId];
        if (prev != -1) buf->lru_next[prev] = next;
        else            lru_bucket_heads[bucket] = next; 
        if (next != -1) buf->lru_prev[next] = prev;
        else            lru_bucket_tails[bucket] = prev; 
    }
    __syncwarp();

    
    if (rep) {
        int32_t old_head = lru_bucket_heads[bucket];
        lru_bucket_heads[bucket] = pkgId; 
        if (old_head != pkgId) {
            buf->lru_prev[pkgId] = -1;
            buf->lru_next[pkgId] = old_head;
            if (old_head != -1) buf->lru_prev[old_head] = pkgId;
        }
        
        if (old_head == -1)
            lru_bucket_tails[bucket] = pkgId;
    }
    __syncwarp();

    
    if (rep) {
        lru_bucket_lock_release(&locks->lock[bucket]);
    }
    __syncwarp();
}





























































__device__ int find_lowest_priority_packages_bucket(int m, int* out, Buffer buf) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int hash = tid ^ clock64();
    int start_bucket = hash % LRU_BUCKETS;

    int cnt = 0;
    int tails[LRU_BUCKETS];

    
    for (int b = 0; b < LRU_BUCKETS; ++b) {
        lru_bucket_lock_acquire(&buf.lru_bucket_locks->lock[b]);
        tails[b] = buf.lru_bucket_tails[b];
        lru_bucket_lock_release(&buf.lru_bucket_locks->lock[b]);
    }

    bool bucket_finished[LRU_BUCKETS] = {false};
    int finished_buckets = 0;

    while (cnt < m && finished_buckets < LRU_BUCKETS) {
        
        for (int ib = 0; ib < LRU_BUCKETS && cnt < m; ++ib) {
            int b = (start_bucket + ib) % LRU_BUCKETS;
            if (!bucket_finished[b]) {
                if (tails[b] != -1) {
                    if (tails[b] < 0 || tails[b] >= MAX_PACKAGE_EACH_BUFFER) {
                        bucket_finished[b] = true;
                        ++finished_buckets;
                        continue;
                    }
                    out[cnt++] = tails[b];
                    tails[b] = buf.lru_prev[tails[b]];
                    if (tails[b] == -1) {
                        bucket_finished[b] = true;
                        ++finished_buckets;
                    }
                    if (cnt >= m) break;
                } else {
                    bucket_finished[b] = true;
                    ++finished_buckets;
                }
            }
        }
        start_bucket = (start_bucket + 1) % LRU_BUCKETS;
    }
    return cnt;
}




__device__ int find_lowest_priority_packages_partitioned(int m, int* out, Buffer buf) {
    
    const int partition_id = blockIdx.x;
    const int partitions_per_gpu = GRID_DIM;

    int buckets_per_partition = (LRU_BUCKETS + partitions_per_gpu - 1) / partitions_per_gpu;
    int start_bucket = partition_id * buckets_per_partition;
    int end_bucket = min((partition_id + 1) * buckets_per_partition, LRU_BUCKETS);

    int cnt = 0;
    int buckets_in_this_partition = end_bucket - start_bucket;
    int tails[LRU_BUCKETS]; 

    
    for (int b = start_bucket; b < end_bucket; ++b) {
        lru_bucket_lock_acquire(&buf.lru_bucket_locks->lock[b]);
        tails[b] = buf.lru_bucket_tails[b];
        lru_bucket_lock_release(&buf.lru_bucket_locks->lock[b]);
    }

    
    bool bucket_finished[LRU_BUCKETS] = {false};  
    int finished_buckets = 0;

    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int hash = tid ^ clock64();
    int round_robin_start = start_bucket + (hash % buckets_in_this_partition);

    while (cnt < m && finished_buckets < buckets_in_this_partition) {
        for (int ib = 0; ib < buckets_in_this_partition && cnt < m; ++ib) {
            int b = start_bucket + ((round_robin_start - start_bucket + ib) % buckets_in_this_partition);
            if (!bucket_finished[b]) {
                if (tails[b] != -1) {
                    if (tails[b] < 0 || tails[b] >= MAX_PACKAGE_EACH_BUFFER) {
                        bucket_finished[b] = true;
                        ++finished_buckets;
                        continue;
                    }
                    out[cnt++] = tails[b];
                    tails[b] = buf.lru_prev[tails[b]];
                    if (tails[b] == -1) {
                        bucket_finished[b] = true;
                        ++finished_buckets;
                    }
                    if (cnt >= m) break;
                } else {
                    bucket_finished[b] = true;
                    ++finished_buckets;
                }
            }
        }
        round_robin_start = (round_robin_start + 1 - start_bucket) % buckets_in_this_partition + start_bucket;
    }
    return cnt;
}



__device__ void update_buffer(Buffer g_buffer, SyncFlags2* d_flags) {
    int bid = blockIdx.x;  
    CommunicationControl* comm = &g_buffer.comm[bid];

    __nv_atomic_store_n(&d_flags->process_vertices_flag[bid], 1, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM);
    printf("warp %d update_buffer, process_vertices_flag: %d\n", threadIdx.x / 32, d_flags->process_vertices_flag[bid]);

    while (!__nv_atomic_load_n(&comm->update_request, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)) {
        __nanosleep(1);
    }
    int m = __nv_atomic_load_n(&comm->m, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    printf("warp %d update_buffer, m: %d\n", threadIdx.x / 32, m);

    int actual_count = find_lowest_priority_packages_partitioned(m, (int*)comm->package_indices, g_buffer);

    
    
    
    
    
    
    
    

    m = actual_count;

    for (int i = 0; i < m; ++i) {
        int pidx = comm->package_indices[i];
        if (pidx < 0 || pidx >= MAX_PACKAGE_EACH_BUFFER) {
            continue;
        }
        Package* pac = &g_buffer.packages[pidx];
        for (int j = 0; j < pac->node_num; ++j) {
            graph_node_t v = pac->nodes[j];
            update_vertex_index(v, -1, -1, g_buffer);
        }
    }

    
    
    
    
    
    __nv_atomic_store_n(&comm->gpu_ready, 1, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM);
    printf("warp %d update_buffer, gpu_ready: %d\n", threadIdx.x / 32, comm->gpu_ready);

    
    while (!__nv_atomic_load_n(&comm->transfer_complete, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)) {
        __nanosleep(1);
    }

    __nv_atomic_store_n(&comm->update_request, 0, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM);
    
    __nv_atomic_store_n(&comm->transfer_complete, 0, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM);
    printf("warp %d update_buffer, transfer_complete: %d\n", threadIdx.x / 32, comm->transfer_complete);
    __nv_atomic_store_n(&comm->gpu_ready, 0, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_SYSTEM);
    

    
    
    
    
    
    for (int i = 0; i < m; ++i) {
        int pidx = comm->package_indices[i];
        Package* pac = &g_buffer.packages[pidx];
        for (int j = 0; j < pac->node_num; ++j) {
            graph_node_t v = pac->nodes[j];
            update_vertex_index(v, pidx, j, g_buffer);
        }
    }
}

void copy_packages_to_slots_async(Package*      d_pkgs,          
                                  const Buffer& h_buf,           
                                  const int*    slot_indices,    
                                  cudaStream_t  stream)
{
    for (int i = 0; i < h_buf.package_num; ++i) {
        int slot = slot_indices[i];
        
        if (slot < 0 || slot >= MAX_PACKAGE_EACH_BUFFER) {
            std::cerr << "ERROR: Invalid slot=" << slot << " for package " << i 
                      << " (valid range: 0-" << (MAX_PACKAGE_EACH_BUFFER-1) << ")" << std::endl;
            continue;  
        }
        const Package& src = h_buf.packages[i];
        size_t vnum = src.node_num;
        
        if (vnum < 0 || vnum > MAX_NODE_NUM_PER_PACKAGE) {
            std::cerr << "ERROR: Invalid node_num=" << vnum << " for package " << i << std::endl;
            continue;
        }
        size_t nnz  = src.rowptr[vnum];
        
        if (nnz < 0 || nnz > MAX_PACKAGE_SIZE) {
            std::cerr << "ERROR: Invalid nnz=" << nnz << " for package " << i << std::endl;
            continue;
        }

        
        graph_node_t* d_nodes  = d_pkgs[slot].nodes;
        graph_node_t* d_rowptr = d_pkgs[slot].rowptr;
        graph_node_t* d_colidx = d_pkgs[slot].colidx;

        
        cudaMemcpyAsync(d_nodes,  src.nodes,  vnum  * sizeof(graph_node_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_rowptr, src.rowptr, (vnum+1) * sizeof(graph_node_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_colidx, src.colidx, nnz   * sizeof(graph_node_t),
                        cudaMemcpyHostToDevice, stream);

        
        cudaMemcpyAsync(&d_pkgs[slot].node_num, &src.node_num, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        int zero = 0;
        cudaMemcpyAsync(&d_pkgs[slot].write_lock,      &zero, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(&d_pkgs[slot].warp_read_count, &zero, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }
}

void copy_packages_to_gpu(Buffer* device_buffer, Buffer h_buffer, const int* package_indices, cudaStream_t stream){
    Package* device_packages = device_buffer->packages;
    
    copy_packages_to_slots_async(device_packages, h_buffer, package_indices,stream);
    cudaStreamSynchronize(stream);
}


void cpu_process_double_queue(DoubleQueueManager* dq_manager, SyncFlags* d_flags, GraphPreprocessor g, Buffer* host_buffer, Buffer* device_buffer, int* cpu_visible_queue0, int* cpu_visible_queue1,cudaStream_t stream) {
    while (true) {
        Buffer h_buffer;
        
        int queue0_flushed, queue1_flushed;
        if (__atomic_load_n(&d_flags->queue0_flushed, __ATOMIC_ACQUIRE)){
            
            int queue_size = MAX_QUEUE_SIZE; 
            h_buffer = process_vertices(cpu_visible_queue0, queue_size, g); 
            
            __atomic_store_n(&d_flags->queue0_flushed, 0, __ATOMIC_RELEASE);
            __atomic_store_n(&host_buffer->comm->m, h_buffer.package_num, __ATOMIC_RELEASE);
            __atomic_store_n(&host_buffer->comm->update_request, 1, __ATOMIC_RELEASE);
            while(!(__atomic_load_n(&host_buffer->comm->gpu_ready, __ATOMIC_ACQUIRE))){
               ;
            }
            copy_packages_to_gpu(device_buffer, h_buffer, host_buffer->comm->package_indices,stream);
            for(int i=0;i<h_buffer.package_num;i++){
                std::cout<<host_buffer->comm->package_indices[i]<<" ";
            }
            std::cout<<std::endl;
            __atomic_store_n(&host_buffer->comm->transfer_complete, 1, __ATOMIC_RELEASE);
        }

        
        if (__atomic_load_n(&d_flags->queue1_flushed, __ATOMIC_ACQUIRE)){
            
            int queue_size = MAX_QUEUE_SIZE;
            h_buffer =  process_vertices(cpu_visible_queue1, queue_size, g);
            __atomic_store_n(&d_flags->queue1_flushed, 0, __ATOMIC_RELEASE);
            __atomic_store_n(&host_buffer->comm->m, h_buffer.package_num, __ATOMIC_RELEASE);
            __atomic_store_n(&host_buffer->comm->update_request, 1, __ATOMIC_RELEASE);

            for(int i=0;i<h_buffer.package_num;i++){
                std::cout<<host_buffer->comm->package_indices[i]<<" ";
            }
            std::cout<<std::endl;
            while(!(__atomic_load_n(&host_buffer->comm->gpu_ready, __ATOMIC_ACQUIRE))){
               ;
            }
            copy_packages_to_gpu(device_buffer, h_buffer, host_buffer->comm->package_indices,stream);
            __atomic_store_n(&host_buffer->comm->transfer_complete, 1, __ATOMIC_RELEASE);
        }
        
        if (__atomic_load_n(&d_flags->matching_over,__ATOMIC_ACQUIRE)) {
            __atomic_store_n(&d_flags->cpu_process_over, 1, __ATOMIC_RELEASE);
            break;
        }
    }
}




Buffer create_buffer_from_graph(Graph g, graph_node_t start_node) {
    Buffer buffer;
    int n = g.nnodes;
    int cur = start_node;
    int package_count = 0;

    Package* packages = new Package[MAX_PACKAGE_EACH_BUFFER];


    
    IndexEntry* index_queue = new IndexEntry[n];
    for (int i = 0; i < n; ++i) {
        index_queue[i].p_index = -1;
        index_queue[i].v_index = -1;
        index_queue[i].version = 0;
    }

    while (cur < n && package_count < MAX_PACKAGE_EACH_BUFFER) {
        std::vector<graph_node_t> cur_nodes;
        std::vector<int> cur_degrees;
        int cur_sum_deg = 0;

        
        while (cur < n) {
            int deg = g.rowptr[cur + 1] - g.rowptr[cur];
            if ((!cur_nodes.empty() && cur_sum_deg + deg > MAX_PACKAGE_SIZE)
                || (cur_nodes.size() >= MAX_NODE_NUM_PER_PACKAGE)) {
                break;
            }
            cur_nodes.push_back(cur);
            cur_degrees.push_back(deg);
            cur_sum_deg += deg;
            ++cur;
            if (cur_sum_deg == MAX_PACKAGE_SIZE || cur_nodes.size() == MAX_NODE_NUM_PER_PACKAGE) {
                break;
            }
        }

        if (cur_nodes.empty()) break;

        Package& pkg = packages[package_count];
        pkg.node_num = cur_nodes.size();
        pkg.write_lock = 0;
        pkg.warp_read_count = 0;
        pkg.nodes = new graph_node_t[pkg.node_num];
        pkg.rowptr = new graph_node_t[pkg.node_num + 1];
        pkg.rowptr[0] = 0;
        for (int i = 0; i < pkg.node_num; ++i) {
            pkg.nodes[i] = cur_nodes[i];
            pkg.rowptr[i + 1] = pkg.rowptr[i] + cur_degrees[i];

            
            index_queue[cur_nodes[i]].p_index = package_count;
            index_queue[cur_nodes[i]].v_index = i;
        }
        int colidx_size = pkg.rowptr[pkg.node_num];
        pkg.colidx = new graph_node_t[colidx_size];
        for (int i = 0; i < pkg.node_num; ++i) {
            graph_node_t v = cur_nodes[i];
            graph_edge_t e_start = g.rowptr[v];
            int deg = cur_degrees[i];
            for (int k = 0; k < deg; ++k) {
                pkg.colidx[pkg.rowptr[i] + k] = g.colidx[e_start + k];
            }
        }
        ++package_count;
    }

    buffer.package_num = package_count;
    buffer.packages = packages;
    buffer.index_queue = index_queue;
    buffer.nnodes = n;
    buffer.comm = nullptr;
    return buffer;
}


Buffer create_buffer_from_graph2(Graph g, graph_node_t idx, int threshold) {
    Buffer buffer;
    int n = g.nnodes;
    int cur = idx; 
    int package_count = 0;

    Package* packages = new Package[MAX_PACKAGE_EACH_BUFFER];

    
    IndexEntry* index_queue = new IndexEntry[n];
    for (int i = 0; i < n; ++i) {
        index_queue[i].p_index = -1;
        index_queue[i].v_index = -1;
        index_queue[i].version = 0;
    }

    while (cur < n && package_count < MAX_PACKAGE_EACH_BUFFER) {
        std::vector<graph_node_t> cur_nodes;
        std::vector<int> cur_degrees;
        int cur_sum_deg = 0;

        
        while (cur < n) {
            int deg = g.rowptr[cur + 1] - g.rowptr[cur];
            if (deg < threshold || deg >= 2 * threshold) {
                ++cur;
                continue; 
            }
            if ((!cur_nodes.empty() && cur_sum_deg + deg > MAX_PACKAGE_SIZE)
                || (cur_nodes.size() >= MAX_NODE_NUM_PER_PACKAGE)) {
                break;
            }
            cur_nodes.push_back(cur);
            cur_degrees.push_back(deg);
            cur_sum_deg += deg;
            ++cur;
            if (cur_sum_deg == MAX_PACKAGE_SIZE || cur_nodes.size() == MAX_NODE_NUM_PER_PACKAGE) {
                break;
            }
        }

        if (cur_nodes.empty()) break;

        Package& pkg = packages[package_count];
        pkg.node_num = cur_nodes.size();
        pkg.write_lock = 0;
        pkg.warp_read_count = 0;
        pkg.nodes = new graph_node_t[pkg.node_num];
        pkg.rowptr = new graph_node_t[pkg.node_num + 1];
        pkg.rowptr[0] = 0;
        for (int i = 0; i < pkg.node_num; ++i) {
            pkg.nodes[i] = cur_nodes[i];
            pkg.rowptr[i + 1] = pkg.rowptr[i] + cur_degrees[i];

            
            index_queue[cur_nodes[i]].p_index = package_count;
            index_queue[cur_nodes[i]].v_index = i;
        }
        int colidx_size = pkg.rowptr[pkg.node_num];
        pkg.colidx = new graph_node_t[colidx_size];
        for (int i = 0; i < pkg.node_num; ++i) {
            graph_node_t v = cur_nodes[i];
            graph_edge_t e_start = g.rowptr[v];
            int deg = cur_degrees[i];
            for (int k = 0; k < deg; ++k) {
                pkg.colidx[pkg.rowptr[i] + k] = g.colidx[e_start + k];
            }
        }
        ++package_count;
    }

    buffer.package_num = package_count;
    buffer.packages = packages;
    buffer.index_queue = index_queue;
    buffer.nnodes = n;
    buffer.comm = nullptr;
    return buffer;
}

Buffer create_buffer_from_graph3(Graph g, int* partitions) {
    

    int n = g.nnodes;
    Buffer buffer;
    buffer.nnodes = n;
    buffer.comm = nullptr;

    Package* packages = new Package[MAX_PACKAGE_EACH_BUFFER];
    IndexEntry* index_queue = new IndexEntry[n];
    for (int i = 0; i < n; ++i) {
        index_queue[i].p_index = -1;
        index_queue[i].v_index = -1;
    }

    int package_count = 0;
    graph_node_t cur = 0;
    while (cur < n && package_count < MAX_PACKAGE_EACH_BUFFER) {
        std::vector<graph_node_t> cur_nodes;
        std::vector<int> cur_degrees;
        int cur_sum_deg = 0;
        
        while (cur < n) {
            if (partitions[cur] == 0) {
                ++cur;
                continue;
            }
            int deg = g.rowptr[cur + 1] - g.rowptr[cur];
            if ((cur_sum_deg + deg > MAX_PACKAGE_SIZE)
                || (cur_nodes.size() >= MAX_NODE_NUM_PER_PACKAGE)) {
                break;
            }
            cur_nodes.push_back(cur);
            cur_degrees.push_back(deg);
            cur_sum_deg += deg;
            ++cur;
            if (cur_sum_deg == MAX_PACKAGE_SIZE || cur_nodes.size() == MAX_NODE_NUM_PER_PACKAGE) {
                break;
            }
        }
        if (cur_nodes.empty()) break;

        Package& pkg = packages[package_count];
        pkg.node_num = cur_nodes.size();
        pkg.write_lock = 0;
        pkg.warp_read_count = 0;
        pkg.nodes = new graph_node_t[pkg.node_num];
        pkg.rowptr = new graph_node_t[pkg.node_num + 1];
        pkg.rowptr[0] = 0;
        for (int i = 0; i < pkg.node_num; ++i) {
            pkg.nodes[i] = cur_nodes[i];
            pkg.rowptr[i + 1] = pkg.rowptr[i] + cur_degrees[i];
            
            index_queue[cur_nodes[i]].p_index = package_count;
            index_queue[cur_nodes[i]].v_index = i;
        }
        int colidx_size = pkg.rowptr[pkg.node_num];
        pkg.colidx = new graph_node_t[colidx_size];
        for (int i = 0; i < pkg.node_num; ++i) {
            graph_node_t v = cur_nodes[i];
            graph_edge_t e_start = g.rowptr[v];
            int deg = cur_degrees[i];
            for (int k = 0; k < deg; ++k) {
                pkg.colidx[pkg.rowptr[i] + k] = g.colidx[e_start + k];
            }
        }
        ++package_count;
    }

    buffer.package_num = package_count;
    buffer.packages = packages;
    buffer.index_queue = index_queue;

    return buffer;
}

Buffer cpu_process_all_vertices(GraphPreprocessor g, int (*cpu_visible_queue)[MAX_QUEUE_SIZE / 2], int* queue_flushed_flags, int queue_num, Buffer* host_buffer,int start_node) {
    std::set<graph_node_t> all_vertices_set;  

    
    
    
    int processed_chunk_count = 0;
    int chunk_size = MAX_QUEUE_SIZE / 2;  
    for (int flag_idx = 0; flag_idx < queue_num; ++flag_idx) {
        if (__atomic_load_n(&queue_flushed_flags[flag_idx], __ATOMIC_ACQUIRE)) {
            
            int chunk_idx = flag_idx;
            
            for (int i = 0; i < chunk_size; ++i) {
                graph_node_t v = cpu_visible_queue[chunk_idx][i];
                
                if (host_buffer->index_queue[v].p_index == -1) {
                    all_vertices_set.insert(v);
                }
            }
            __atomic_store_n(&queue_flushed_flags[flag_idx], 0, __ATOMIC_RELEASE);
            processed_chunk_count++;
        }
    }
    

    
    std::vector<graph_node_t> nodes;
    if (hop2neighbor) {
        
        Graph graph = g.g;
        std::set<graph_node_t> expanded_set = all_vertices_set;
        for (graph_node_t v : all_vertices_set) {
            for (graph_edge_t j = graph.rowptr[v]; j < graph.rowptr[v+1]; j++) {
                if(graph.colidx[j]<start_node)continue;
                
                    expanded_set.insert(graph.colidx[j]);
                
            }
        }
        nodes.assign(expanded_set.begin(), expanded_set.end());
    } else {
        nodes.assign(all_vertices_set.begin(), all_vertices_set.end());
    }

    
    int node_num = nodes.size();
    std::vector<int> degree;
    int sum_degree = 0;
    Graph graph = g.g;
    for (int i = 0; i < node_num; ++i) {
        graph_node_t v = nodes[i];
        int d = graph.rowptr[v+1] - graph.rowptr[v];
        degree.push_back(d);
        sum_degree += d;
    }
    
    
    if (sum_degree < MAX_PACKAGE_SIZE) {
        std::set<graph_node_t> visited(nodes.begin(), nodes.end());
        std::queue<graph_node_t> bfs_queue;
        for (graph_node_t v : nodes) {
            bfs_queue.push(v);
        }
        Graph graph = g.g;
        bool reached_limit = false;
        while (sum_degree < MAX_PACKAGE_SIZE && !bfs_queue.empty() && !reached_limit) {
            int qsize = bfs_queue.size();
            std::vector<graph_node_t> new_neighbors;
            for (int i = 0; i < qsize; ++i) {
                graph_node_t v = bfs_queue.front();
                bfs_queue.pop();
                for (graph_edge_t j = graph.rowptr[v]; j < graph.rowptr[v+1]; ++j) {
                    graph_node_t u = graph.colidx[j];
                    if(u<start_node)continue;
                    
                        visited.insert(u);
                        new_neighbors.push_back(u);
                    
                }
            }
            
            for (auto u : new_neighbors) {
                int d = graph.rowptr[u+1] - graph.rowptr[u];
                if (sum_degree + d > MAX_PACKAGE_SIZE) {
                    
                    reached_limit = true;
                    break;
                }
                nodes.push_back(u);
                degree.push_back(d);
                sum_degree += d;
                bfs_queue.push(u);
                
                if (sum_degree == MAX_PACKAGE_SIZE) {
                    reached_limit = true;
                    break;
                }
            }
        }
        node_num = nodes.size();
    }
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    int package_num = sum_degree / MAX_PACKAGE_SIZE + 1;
    if (package_num > MAX_PACKAGES) package_num = MAX_PACKAGES;  
    std::vector<std::vector<int>> node_list(package_num, std::vector<int>());
    std::vector<std::vector<int>> degree_list(package_num, std::vector<int>());
    std::vector<int> contain(package_num, MAX_PACKAGE_SIZE);

    
    for (int i = 0; i < node_num; ++i) {
        for (int j = 0; j < package_num; ++j) {
            if (contain[j] >= degree[i]) {
                node_list[j].push_back(nodes[i]);
                degree_list[j].push_back(degree[i]);
                contain[j] -= degree[i];
                break;
            }
        }
    }

    
    Package* p_host = new Package[package_num];
    for (int i = 0; i < package_num; ++i) {
        p_host[i].nodes = new graph_node_t[node_list[i].size()];
        p_host[i].rowptr = new graph_node_t[node_list[i].size() + 1];
        p_host[i].node_num = node_list[i].size();
        p_host[i].write_lock = 0;
        p_host[i].warp_read_count = 0;
        memcpy(p_host[i].nodes, node_list[i].data(), node_list[i].size() * sizeof(graph_node_t));
        p_host[i].rowptr[0] = 0;
        for (int j = 0; j < node_list[i].size(); ++j) {
            p_host[i].rowptr[j+1] = p_host[i].rowptr[j] + degree_list[i][j];
        }
        p_host[i].colidx = new graph_node_t[p_host[i].rowptr[node_list[i].size()]];
        for (int j = 0; j < node_list[i].size(); ++j) {
            graph_node_t v = node_list[i][j];
            graph_edge_t src = graph.rowptr[v];
            graph_node_t dst = p_host[i].rowptr[j];
            for (int k = 0; k < degree_list[i][j]; ++k) {
                p_host[i].colidx[dst+k] = graph.colidx[src+k];
            }
        }
    }

    Buffer h_buffer;
    h_buffer.packages = p_host;
    h_buffer.package_num = package_num;
    return h_buffer;
}























































































































        



        










        
















    






































void cpu_process_multiple_queue(GraphPreprocessor g, Buffer* host_buffer, Buffer* device_buffer,  int (*cpu_visible_queues)[CHUNKS_PER_BLOCK][MAX_QUEUE_SIZE/2], int queue_num, SyncFlags2* d_flags, int tid, int start_node, cudaStream_t stream) {
    
    long long local_m = 0;
    std::uint64_t local_bytes = 0;
    while (true) {
        
        if (__atomic_load_n(&d_flags->process_vertices_flag[tid], __ATOMIC_ACQUIRE)) {
            
            
            Buffer h_buffer = cpu_process_all_vertices(g, cpu_visible_queues[tid], d_flags->queue_flushed_flags[tid], queue_num, host_buffer,start_node);
            printf("cpu_process_multiple_queue, h_buffer.package_num: %d\n", h_buffer.package_num);
            
            local_m     += static_cast<long long>(h_buffer.package_num);
            local_bytes += static_cast<std::uint64_t>(h_buffer.package_num) * CPU_PACKAGE_BYTES;
            
            __atomic_store_n(&d_flags->process_vertices_flag[tid], 0, __ATOMIC_RELEASE);
            
            
            __atomic_store_n(&host_buffer->comm[tid].m, h_buffer.package_num, __ATOMIC_RELEASE);
            int m_check = __atomic_load_n(&host_buffer->comm[tid].m, __ATOMIC_ACQUIRE);
            if (m_check != h_buffer.package_num) {
                std::cerr << "ERROR: m was corrupted immediately after setting! "
              << "Expected " << h_buffer.package_num 
              << " but got " << m_check << std::endl;
            }
            __atomic_store_n(&host_buffer->comm[tid].update_request, 1, __ATOMIC_RELEASE);
            printf("cpu_process_multiple_queue, update_request: %d\n", host_buffer->comm[tid].update_request);
            
            while (!(__atomic_load_n(&host_buffer->comm[tid].gpu_ready, __ATOMIC_ACQUIRE))) { ; }
            
            copy_packages_to_gpu(device_buffer, h_buffer, host_buffer->comm[tid].package_indices, stream);
            
            
            
            
            

            
            
            
            
            
            
            
            
            __atomic_store_n(&host_buffer->comm[tid].transfer_complete, 1, __ATOMIC_RELEASE);
            
        }
        
        if (__atomic_load_n(&d_flags->matching_over,__ATOMIC_ACQUIRE)) {
            
            __atomic_fetch_add(&g_total_cpu_m, local_m, __ATOMIC_ACQ_REL);
            __atomic_fetch_add(&g_total_cpu_bytes, local_bytes, __ATOMIC_ACQ_REL);

            
            int finished = __atomic_add_fetch(&g_cpu_threads_finished, 1, __ATOMIC_ACQ_REL);
            if (finished == GRID_DIM) {
                long long total_m = __atomic_load_n(&g_total_cpu_m, __ATOMIC_ACQUIRE);
                std::uint64_t total_bytes = __atomic_load_n(&g_total_cpu_bytes, __ATOMIC_ACQUIRE);
                double total_kb = static_cast<double>(total_bytes) / 1024.0;
            }
            __atomic_store_n(&d_flags->cpu_process_over, 1, __ATOMIC_RELEASE);
            break;
        }
    }
}


Buffer process_vertices(int* cpu_visible_queue, int node_num, GraphPreprocessor gp){
    std::vector<graph_node_t>nodes, degree;
    std::set<graph_node_t> node_set;
    Graph g = gp.g;
    if(!hop2neighbor){
        for(int i = 0;i < node_num; i ++){
            graph_node_t v = cpu_visible_queue[i];
            nodes.push_back(v);
        }
    }else{
        for(int i = 0; i < node_num; i++){
            graph_node_t v = cpu_visible_queue[i];
            node_set.insert(v);
            for(graph_edge_t j = g.rowptr[v]; j < g.rowptr[v+1]; j++){
                node_set.insert(g.colidx[j]);
            }
        }
        for(graph_node_t n : node_set){
            nodes.push_back(n);
        }
    }
    node_num = nodes.size();
    int sum_degree = 0;
    for(int i = 0; i < node_num; i ++){
        graph_node_t v = nodes[i];
        int d = g.rowptr[v+1] - g.rowptr[v];
        degree.push_back(d);
        sum_degree += d;
    }

    int package_num = sum_degree/MAX_PACKAGE_SIZE + 1;
    std::vector<std::vector<int>> node_list(package_num,std::vector<int>());
    std::vector<std::vector<int>> degree_list(package_num,std::vector<int>());
    std::vector<int> contain(package_num, MAX_PACKAGE_SIZE);
    for(int i = 0; i < node_num; i ++){
        for(int j = 0 ; j < package_num ; j++){
            if(contain[j]>=degree[i]){
                node_list[j].push_back(nodes[i]);
                degree_list[j].push_back(degree[i]);
                contain[j]-=degree[i];
                break;
            }
        }
    }


    Package* p_host = new Package[package_num];

    for(int i = 0; i < package_num; i++){
        p_host[i].nodes = new graph_node_t[node_list[i].size()];
        p_host[i].rowptr = new graph_node_t[node_list[i].size()+1];
        p_host[i].node_num = node_list[i].size();
        p_host[i].write_lock = 0;
        p_host[i].warp_read_count = 0;
        memcpy(p_host[i].nodes, node_list[i].data(), node_list[i].size() * sizeof(int));
        p_host[i].rowptr[0] = 0;
        for(int j = 0; j < node_list[i].size(); j++){
            p_host[i].rowptr[j+1] = p_host[i].rowptr[j] + degree_list[i][j];
        }
        p_host[i].colidx = new graph_node_t[p_host[i].rowptr[node_list[i].size()]];
        for(int j = 0; j < node_list[i].size(); j++){
            graph_node_t v = node_list[i][j];
            graph_edge_t src = g.rowptr[v];
            graph_node_t dst = p_host[i].rowptr[j];
            for(int k = 0; k < degree_list[i][j]; k++){
                p_host[i].colidx[dst+k] = g.colidx[src+k];
            }
        }
    }
    Buffer h_buffer;
    h_buffer.packages = p_host;
    h_buffer.package_num = package_num;
    return h_buffer;
}

Buffer process_vertices_block(int* cpu_visible_queue, int node_num, GraphPreprocessor gp, Buffer* h_buf){
    std::vector<graph_node_t>nodes, degree;
    std::set<graph_node_t> node_set;
    Graph g = gp.g;
    if(!hop2neighbor){
        for(int i = 0;i < node_num; i ++){
            graph_node_t v = cpu_visible_queue[i];
            if(h_buf->index_queue[v].p_index==-1)
                nodes.push_back(v);
        }
    }else{
        for(int i = 0; i < node_num; i++){
            graph_node_t v = cpu_visible_queue[i];
            if(h_buf->index_queue[v].p_index==-1)
                node_set.insert(v);
            for(graph_edge_t j = g.rowptr[v]; j < g.rowptr[v+1]; j++){
                graph_node_t vn = g.colidx[j];
                if(h_buf->index_queue[vn].p_index==-1)
                    node_set.insert(vn);
            }
        }
        for(graph_node_t n : node_set){
            nodes.push_back(n);
        }
    }
    node_num = nodes.size();
    int sum_degree = 0;
    for(int i = 0; i < node_num; i ++){
        graph_node_t v = nodes[i];
        int d = g.rowptr[v+1] - g.rowptr[v];
        degree.push_back(d);
        sum_degree += d;
    }

    int package_num = sum_degree/MAX_PACKAGE_SIZE + 1;
    std::vector<std::vector<int>> node_list(package_num,std::vector<int>());
    std::vector<std::vector<int>> degree_list(package_num,std::vector<int>());
    std::vector<int> contain(package_num, MAX_PACKAGE_SIZE);
    for(int i = 0; i < node_num; i ++){
        for(int j = 0 ; j < package_num ; j++){
            if(contain[j]>=degree[i]){
                node_list[j].push_back(nodes[i]);
                degree_list[j].push_back(degree[i]);
                contain[j]-=degree[i];
                break;
            }
        }
    }


    Package* p_host = new Package[package_num];

    for(int i = 0; i < package_num; i++){
        p_host[i].nodes = new graph_node_t[node_list[i].size()];
        p_host[i].rowptr = new graph_node_t[node_list[i].size()+1];
        p_host[i].node_num = node_list[i].size();
        p_host[i].write_lock = 0;
        p_host[i].warp_read_count = 0;
        memcpy(p_host[i].nodes, node_list[i].data(), node_list[i].size() * sizeof(int));
        p_host[i].rowptr[0] = 0;
        for(int j = 0; j < node_list[i].size(); j++){
            p_host[i].rowptr[j+1] = p_host[i].rowptr[j] + degree_list[i][j];
        }
        p_host[i].colidx = new graph_node_t[p_host[i].rowptr[node_list[i].size()]];
        for(int j = 0; j < node_list[i].size(); j++){
            graph_node_t v = node_list[i][j];
            graph_edge_t src = g.rowptr[v];
            graph_node_t dst = p_host[i].rowptr[j];
            for(int k = 0; k < degree_list[i][j]; k++){
                p_host[i].colidx[dst+k] = g.colidx[src+k];
            }
        }
    }
    Buffer h_buffer;
    h_buffer.packages = p_host;
    h_buffer.package_num = package_num;
    return h_buffer;
}
__device__ bool add_vertex(
    DoubleQueueManager* dq_manager,
    SyncFlags* flags,
    int vertex,
    int* cpu_visible_queue0,
    int* cpu_visible_queue1
) {
    bool switched = false;
    int active_idx = __nv_atomic_load_n(&dq_manager->active_idx, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    RequestQueue* active_queue = &dq_manager->queues[active_idx];
    int new_tail = 0;
    int idx = 0;
    
    if (__nv_atomic_load_n(&active_queue->is_active, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)) {
    
        acquire_queue_lock(&active_queue->lock);
    
    
        if (__nv_atomic_load_n(&active_queue->is_active, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE)) {
        
            new_tail = atomicAdd(&active_queue->tail, 1);
            idx = new_tail % active_queue->capacity;
        
            if (idx < active_queue->capacity) {
                active_queue->data[idx] = vertex;
            }
        }else{
            release_queue_lock(&active_queue->lock);
            active_idx = __nv_atomic_load_n(&dq_manager->active_idx, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            active_queue = &dq_manager->queues[active_idx];
            acquire_queue_lock(&active_queue->lock);
            new_tail = atomicAdd(&active_queue->tail, 1);
            idx = new_tail % active_queue->capacity;
        
        
            if (idx < active_queue->capacity) {
                active_queue->data[idx] = vertex;
            }
        }
        if(idx == MAX_QUEUE_SIZE-1){
            int old_active_idx = switch_active_queue(dq_manager);
        
            if (old_active_idx == 0) {
                copy_inactive_queue(dq_manager, 0, cpu_visible_queue0, &flags->queue0_flushed);
            } else {
                copy_inactive_queue(dq_manager, 1, cpu_visible_queue1, &flags->queue1_flushed);
            }
            
            switched = true;
        }
    
        release_queue_lock(&active_queue->lock);   
    }
    return switched;
    
}


__device__ bool add_vertex_block(int vertex,
                                      BlockQueue* sm_bq,
                                      int* g_cpu_vis,
                                      int* write_block_flags,
                                      SyncFlags2* d_flags) {
    bool half_switched = false;
    int idx = -1;
    
    bool allow = true;
    if (threadIdx.x % WARP_SIZE == 0) {
        allow = dedup_allow_and_set(vertex, sm_bq->dedup_table);
    }
    allow = __shfl_sync(0xFFFFFFFF, allow, 0);
    if (!allow) {
        return false; 
    }

    
    if (threadIdx.x % 32== 0) {
        int old_tail = atomicAdd(&sm_bq->tail, 1);
        idx = old_tail & (MAX_QUEUE_SIZE - 1);
    }

    
    idx = __shfl_sync(0xFFFFFFFF, idx, 0);

    
    if (idx >= 0) {
        sm_bq->data[idx] = vertex;
    }

    
    if (threadIdx.x % 32 == 0) {
        int head = sm_bq->head;
        int tail = sm_bq->tail;
        int fill_count = tail - head;

        
        const int max_chunks_per_block = CHUNKS_PER_BLOCK;

        if (fill_count >= (MAX_QUEUE_SIZE / 2)) {
            
            if (atomicCAS(&write_block_flags[blockIdx.x], 0, 1) == 0) {
                
                int chunk_idx = (head / (MAX_QUEUE_SIZE / 2)) % max_chunks_per_block;

                
                int* dst = g_cpu_vis + blockIdx.x * CPU_VISIBLE_SIZE + chunk_idx * (MAX_QUEUE_SIZE / 2);

                
                for (int i = 0; i < (MAX_QUEUE_SIZE / 2); ++i) {
                    dst[i] = sm_bq->data[(head + i) & (MAX_QUEUE_SIZE - 1)];
                }
                
                sm_bq->head += (MAX_QUEUE_SIZE / 2);
                half_switched = true;

                
                __threadfence(); 
                
                atomicExch(&d_flags->queue_flushed_flags[blockIdx.x][chunk_idx], 1);
                
                
                
                atomicExch(&write_block_flags[blockIdx.x], 0);
                
                for (int i = 0; i < BLOCK_DEDUP_SIZE; ++i) {
                    sm_bq->dedup_table[i] = -1;
                }
            }
        }
    }

    
    half_switched = __shfl_sync(0xFFFFFFFF, half_switched, 0);

    return half_switched;
}







































































