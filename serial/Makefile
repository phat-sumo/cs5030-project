# Serial Makefile
CC = gcc
TARGET = serial
FLAGS = -march=native -mtune=native -O3 -flto -fomit-frame-pointer -pipe
WARN = -Wall
SRC = $(TARGET).c ../common/bresenham.c ../common/bresenham.h

build:
	$(CC) -o $(TARGET) $(FLAGS) $(WARN) $(SRC)

test:
	./$(TARGET)

run: build test

.PHONY: build
