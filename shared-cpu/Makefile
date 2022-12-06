# Shared CPU Makefile
CC = gcc
TARGET = shared-cpu
FLAGS = -march=native -mtune=native -O3 -flto -fomit-frame-pointer -pipe -fopenmp
WARN = -Wall
SRC = $(TARGET).c ../common/bresenham.c ../common/bresenham.h

build:
	$(CC) -o $(TARGET) $(FLAGS) $(WARN) $(SRC)

test:
	./$(TARGET)

run: build test

.PHONY: build
