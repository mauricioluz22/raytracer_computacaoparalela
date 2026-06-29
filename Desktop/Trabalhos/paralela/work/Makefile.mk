CXX := ladcomp -env mpiCC
FLAGS := -std=c++11 -Wall -Wextra -O3 -DNDEBUG -ffast-math -ftree-vectorize # -march=native -mtune=native

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
