CUDA_DIR ?= /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
	   CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif

CC := nvcc
INCLUDE := $(CUDA_DIR)/include
GENCODE_FLAGS := -gencode arch=compute_80,code=compute_80 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_70,code=compute_70 -gencode arch=compute_62,code=compute_62 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_60,code=compute_60
OPTS := -O3 $(GENCODE_FLAGS) -rdc=true
OPTS_DEBUG := -G -g -O0 $(GENCODE_FLAGS) -rdc=true

SRC = main.cu

EXE = allSyncPrims-1kernel

$(EXE): release

release: $(SRC)
	$(CC) $(OPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

debug: $(SRC)
	$(CC) $(OPTS_DEBUG) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo *~
