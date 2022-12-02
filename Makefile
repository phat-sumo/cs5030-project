# cs5030 project

CC = gcc
TARGET = AwannaCU
WARN = -Wall
LIBS = -fopenmp
SRC = main.c bresenham.c bresenham.h

build: 
	$(CC) -o $(TARGET) $(WARN) $(LIBS) $(SRC)

test: 
	./$(TARGET)

run: build test

.PHONY: build
