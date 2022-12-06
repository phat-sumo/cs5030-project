# Serial Makefile
CC = gcc
TARGET = validate
FLAGS = -march=native -mtune=native -O3 -flto -fomit-frame-pointer -pipe -fopenmp
WARN = -Wall
SRC = $(TARGET).c 

build:
	$(CC) -o $(TARGET) $(FLAGS) $(WARN) $(SRC)

test:
	./$(TARGET)

run: build test

.PHONY: build
