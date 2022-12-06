# Distributed GPU Makefile
NVCC=nvcc
TARGET = distributed-gpu
CUDA_HOME ?= /opt/cuda
NVCC_FLAGS += -dc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt -std=c++14  -I$(MPI_HOME)/include
NVCC_LDFLAGS = -ccbin=mpic++ -L$(NVSHMEM_HOME)  -L$(MPI_HOME)/lib -lmpi -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvToolsExt
SRC = $(TARGET).cu ../common/cuda_bresenham.cu

build:
	$(NVCC) $(NVCC_FLAGS) $(TARGET).cu -o $(TARGET).o
	$(NVCC) $(TARGET).o -o $(TARGET) $(NVCC_LDFLAGS)

test:
	mpiexec -n 4 ./$(TARGET)

run: build test

.PHONY: build