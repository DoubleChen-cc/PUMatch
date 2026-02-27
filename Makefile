# 要求 CUDA >= 12.8, ARCH >= sm_70，用到 _nv_atomic_load_n
CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc

NVCC_FLAGS := -arch=sm_80 \
              -std=c++17 \
              -Xptxas -v\
	      -G\
              -Xcompiler -Wall \
              -rdc=true          # 必须开 relocatable device code

INCLUDES   := -I.

TARGET     := test
SRC        := test.cu
OBJ        := $(SRC:.cu=.o)

# 1. 把“实现缺失”的 cu 文件加进来（按你实际文件名写）
OTHER_OBJS := src/buffer.o \
              src/gpu_match.o \
              src/missing_tree.o   # <-- 新增，放你的定义

# 2. 默认目标
all: $(TARGET)

# 3. 链接：把 relocatable device code 链接成可执行文件
$(TARGET): $(OBJ) $(OTHER_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^

# 4. 生成 relocatable device code（-dc 而不是 -c）
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# 5. 清理
clean:
	rm -f $(TARGET) $(OBJ) $(OTHER_OBJS)

.PHONY: all clean
