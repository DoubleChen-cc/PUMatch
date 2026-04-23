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
MPI_TARGET := test_mpi
SRC        := test.cu
MPI_SRC    := test_mpi.cu
OBJ        := $(SRC:.cu=.o)
MPI_OBJ    := $(MPI_SRC:.cu=.o)

MPI_PREFIX ?= /usr/local/openmpi
MPI_CXX    := $(MPI_PREFIX)/bin/mpicxx
# nvcc 不接受顶层的 -pthread，须交给主机编译器 / 链接器一侧
MPI_RAW_COMPILE   := $(shell $(MPI_CXX) --showme:compile 2>/dev/null)
MPI_COMPILE_FLAGS := $(filter-out -pthread,$(MPI_RAW_COMPILE)) -Xcompiler -pthread
# nvcc 不能把 --showme:link 里的 -Wl,… 当作自身参数；-Wl 必须拆开成 -Xlinker 片段
MPI_LINK_FLAGS    := -L$(MPI_PREFIX)/lib -lmpi \
	-Xlinker -rpath -Xlinker $(MPI_PREFIX)/lib \
	-Xlinker --enable-new-dtags \
	-Xcompiler -pthread

# 1. 把“实现缺失”的 cu 文件加进来（按你实际文件名写）
OTHER_OBJS := src/buffer.o \
              src/gpu_match.o \
              src/missing_tree.o   # <-- 新增，放你的定义

# 2. 默认目标
all: $(TARGET)

# 3. 链接：把 relocatable device code 链接成可执行文件
$(TARGET): $(OBJ) $(OTHER_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^

# MPI 多节点入口：-rdc 目标文件必须由 nvcc 做最终链接（才会走 nvlink），再挂上 MPI 库
$(MPI_TARGET): $(MPI_OBJ) $(OTHER_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^ \
		$(MPI_LINK_FLAGS) \
		-L$(CUDA_PATH)/lib64 -lcudart -lcuda \
		-lstdc++fs

# 4. 生成 relocatable device code（-dc 而不是 -c）
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

$(MPI_OBJ): $(MPI_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(MPI_COMPILE_FLAGS) -dc $< -o $@

# 5. 清理
clean:
	rm -f $(TARGET) $(MPI_TARGET) $(OBJ) $(MPI_OBJ) $(OTHER_OBJS)

.PHONY: all clean
