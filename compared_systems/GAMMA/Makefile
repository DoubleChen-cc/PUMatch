NVCC=nvcc
NVCCFLAGS=-gencode arch=compute_80,code=sm_80 #--ptxas-options=-v
CXX=g++
CXXFLAGS=-O3 -std=c++11 #-fopemmp -Iinclude
DEBUGFLAGS=#-G -g
MPI_HOME=/usr/local/openmpi
MPI_CPPFLAGS=-I$(MPI_HOME)/include
# OpenMPI 4+ 通常不再提供 libmpi_cxx；本工程仅用 MPI C 接口，只需 -lmpi
# -rpath：运行时在各节点上能找到 libmpi.so（各节点需存在相同 MPI_HOME 路径）
MPI_LDFLAGS=-L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -Xlinker -rpath -Xlinker $(MPI_HOME)/lib -Xlinker -rpath -Xlinker $(MPI_HOME)/lib64 -lmpi
DEPENDENCY=accessMode.cuh graph.cuh utils.h embedding.cuh expand.cuh log.h clock.h
all: kcl fsm sm

sm : sm.cu ${DEPENDENCY} queryGraph.cuh log.o
	${NVCC} sm.cu log.o -O3 ${NVCCFLAGS} ${DEBUGFLAGS} -o sm -std=c++11
# 多机多 GPU + MPI（OpenMPI 头文件与库路径见 MPI_HOME）
sm_mpi : sm_mpi.cu ${DEPENDENCY} queryGraph.cuh log.o
	${NVCC} sm_mpi.cu log.o -O3 ${NVCCFLAGS} ${DEBUGFLAGS} -o sm_mpi -std=c++11 $(MPI_CPPFLAGS) $(MPI_LDFLAGS)
kcl : kcl.cu ${DEPENDENCY} log.o
	${NVCC} kcl.cu log.o -O3 ${NVCCFLAGS} ${DEBUGFLAGS} -o kcl -std=c++11
fsm : fsm.cu ${DEPENDENCY} aggregrate.cuh log.o
	${NVCC} fsm.cu log.o -O3 ${NVCCFLAGS} ${DEBUGFLAGS} -o fsm -std=c++11
multimerge: multimerge.cu ${DEPENDENCY} 
	${NVCC} multimerge.cu log.o -O3 ${NVCCFLAGS} ${DEBUGFLAGS} -o multimerge -std=c++11
log.o: log.cpp 
	$(CXX) $(CXXFLAGS) -c log.cpp -o log.o
clean:
	rm -f kcl sm sm_mpi fsm multimerge *.o
