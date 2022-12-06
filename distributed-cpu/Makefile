# Distributed CPU Makefile
MPICC = mpicc
TARGET = distributed-cpu
FLAGS = -march=native -mtune=native -O3 -flto -fomit-frame-pointer -pipe
WARN = -Wall
SRC = $(TARGET).c ../common/bresenham.c ../common/bresenham.h

build:
	$(MPICC) -o $(TARGET) $(FLAGS) $(WARN) $(SRC)

debug:
	$(MPICC) -o $(TARGET) $(WARN) -Og $(SRC)

test:
	mpiexec --use-hwthread-cpus ./$(TARGET)

run: build test

.PHONY: build
