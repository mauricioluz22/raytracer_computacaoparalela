CXX := g++
FLAGS := -std=c++11 -Wall -Wextra -O3 -DNDEBUG -ffast-math -fopenmp -march=native -mtune=native -ftree-vectorize -fopenmp-simd

BUILD := build
SRC := InOneWeekend/main.cc
OUTPUT := $(BUILD)/main

all: $(OUTPUT)

$(OUTPUT): $(SRC)
	mkdir -p $(BUILD)
	$(CXX) -o $(OUTPUT) $(SRC) $(FLAGS)

.PHONY: clean
clean:
	rm -f $(OUTPUT)