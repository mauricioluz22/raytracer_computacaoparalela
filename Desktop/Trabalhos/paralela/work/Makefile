CXX := mpic++
# -fopenmp habilita o paralelismo de memoria compartilhada dentro de cada no.
FLAGS := -std=c++11 -Wall -Wextra -O3 -DNDEBUG -ffast-math -ftree-vectorize -fopenmp # -march=native -mtune=native

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
