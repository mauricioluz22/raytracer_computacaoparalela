CXX := g++
FLAGS := -std=c++11 -Wall -Wextra -O3 -DNDEBUG -ffast-math -fopenmp -march=native -mtune=native -ftree-vectorize -fopenmp-simd

all: main

main: main.cc
	$(CXX) -o main main.cc $(FLAGS)

.PHONY: clean
clean:
	rm -f main