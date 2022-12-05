# Shared GPU Makefile
NVCC = nvcc
TARGET = shared-gpu
FLAGS = -O3 -dlto
SRC = $(TARGET).cu

build: 
	$(NVCC) $(FLAGS) -o $(TARGET) $(SRC)

test:
	./$(TARGET)

run: build test

.PHONY: build
