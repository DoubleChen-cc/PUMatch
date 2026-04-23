#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cstdint>

// 并行去重队列结构
struct ParallelDeduplicationQueue {
    int* d_queue;          // 队列存储
    int capacity;          // 队列容量
    int* d_size;           // 当前大小（原子操作）
    uint64_t* d_bitmap;    // 位图用于去重
    int bitmap_size;       // 位图大小（以uint64_t为单位）
};

// 初始化队列
__host__ void initQueue(ParallelDeduplicationQueue& queue, int capacity, int max_value) {
    cudaMalloc(&queue.d_queue, capacity * sizeof(int));
    cudaMalloc(&queue.d_size, sizeof(int));
    cudaMemset(queue.d_size, 0, sizeof(int));
    
    // 计算位图大小（每64个元素用一个uint64_t）
    queue.bitmap_size = (max_value + 63) / 64;
    cudaMalloc(&queue.d_bitmap, queue.bitmap_size * sizeof(uint64_t));
    cudaMemset(queue.d_bitmap, 0, queue.bitmap_size * sizeof(uint64_t));
    
    queue.capacity = capacity;
}

// 释放队列资源
__host__ void freeQueue(ParallelDeduplicationQueue& queue) {
    cudaFree(queue.d_queue);
    cudaFree(queue.d_size);
    cudaFree(queue.d_bitmap);
}

// 检查元素是否存在并尝试插入（原子操作）
__device__ bool tryInsert(ParallelDeduplicationQueue queue, int value) {
    // 计算位图中的位置
    int word_idx = value / 64;
    int bit_idx = value % 64;
    uint64_t mask = 1ULL << bit_idx;
    
    // 原子操作检查并设置位
    uint64_t old_value = atomicCAS(&queue.d_bitmap[word_idx], 
                                   queue.d_bitmap[word_idx], 
                                   queue.d_bitmap[word_idx] | mask);
    
    // 如果原来的位是0，表示元素不存在，可以插入
    bool inserted = !(old_value & mask);
    
    if (inserted) {
        // 获取队列中的位置
        int pos = atomicAdd(queue.d_size, 1);
        if (pos < queue.capacity) {
            queue.d_queue[pos] = value;
            return true;
        } else {
            // 队列已满，清除位图中的标记
            atomicAnd(&queue.d_bitmap[word_idx], ~mask);
            return false;
        }
    }
    
    return false;
}

// 多warp并行插入去重示例
__global__ void parallelInsertKernel(ParallelDeduplicationQueue queue, int* values, int num_values) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_values) return;
    
    int value = values[idx];
    tryInsert(queue, value);
}

// 主函数示例
int main() {
    // 初始化队列
    ParallelDeduplicationQueue queue;
    initQueue(queue, 1024, 10000); // 队列容量1024，最大元素值10000
    
    // 准备输入数据（假设由主机生成）
    std::vector<int> host_values = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    int num_values = host_values.size();
    
    // 分配并复制数据到GPU
    int* d_values;
    cudaMalloc(&d_values, num_values * sizeof(int));
    cudaMemcpy(d_values, host_values.data(), num_values * sizeof(int), cudaMemcpyHostToDevice);
    
    // 启动并行插入核函数
    int block_size = 256;
    int grid_size = (num_values + block_size - 1) / block_size;
    parallelInsertKernel<<<grid_size, block_size>>>(queue, d_values, num_values);
    
    // 同步并获取结果
    cudaDeviceSynchronize();
    
    int queue_size;
    cudaMemcpy(&queue_size, queue.d_size, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::vector<int> host_result(queue_size);
    cudaMemcpy(host_result.data(), queue.d_queue, queue_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 清理资源
    cudaFree(d_values);
    freeQueue(queue);
    
    return 0;
}

