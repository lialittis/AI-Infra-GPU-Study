CUDA_HOME ?= /usr/local/cuda
NVCC  ?= $(CUDA_HOME)/bin/nvcc
ARCH  ?= native
FLAGS ?= -O2 -std=c++17 -lineinfo -arch=$(ARCH)

SRCS := $(wildcard *.cu)
BINS := $(patsubst %.cu,build/%,$(SRCS))

all: $(BINS)

build/%: %.cu common.cuh
	@mkdir -p build
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -rf build

.PHONY: all clean
