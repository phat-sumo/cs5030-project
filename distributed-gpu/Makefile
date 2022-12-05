# Distributed GPU Makefile
export CUDA_HOME=/opt/cuda
export NCCL_DEBUG=info

TARGET = distributed-gpu
NVCC = nvcc
MPICXX = mpicxx
CUDA_HOME ?= /opt/cuda
NCCL_HOME ?= /usr/include
GENCODE_SM70 := -gencode arch=compute_70,code=sm_70
GENCODE_SM80 := -gencode arch=compute_80,code=sm_80
GENCODE_FLAGS := $(GENCODE_SM70) $(GENCODE_SM80)
NVCC_FLAGS += -lineinfo $(GENCODE_FLAGS) -std=c++17
MPICXX_FLAGS = -DUSE_NVTX -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -std=c++17
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt -lnccl

$(TARGET): Makefile $(TARGET).cpp bresenham.o
	$(MPICXX) $(MPICXX_FLAGS) $(TARGET).cpp bresenham.o $(LD_FLAGS) -o $(TARGET)

bresenham.o: Makefile ../common/bresenham.cu
	$(NVCC) $(NVCC_FLAGS) ../common/bresenham.cu -c

run:
	mpiexec -n 4 ./$(TARGET)