# cs5030 project

CC = gcc
TARGET = AwannaCU
FLAGS = -O3 -flto -march=native -mtune=native
WARN = -Wall
LIBS = -fopenmp
SRC = main.c bresenham.c bresenham.h

build: 
	$(CC) -o $(TARGET) $(FLAGS) $(WARN) $(LIBS) $(SRC)

test: 
	./$(TARGET)

run: build test

.PHONY: build
