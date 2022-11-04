HIPOPTS = --amdgpu-target=gfx906,gfx900

ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip

CC := $(HIP_PATH)/bin/hipcc

INCLUDE := $(ROCM_PATH)/include -Iinclude
LIB := $(ROCM_PATH)/lib -lhiprand -lrocrand

SRC = skel.hip.cc

EXE_GSRB = scan_gsrb
EXE_GCPUSRB = scan_gcpusrb

all: release-gsrb release-gcpusrb

release-gsrb:
	$(CC) -DGSRB $(HIPOPTS) $(SRC) -o $(EXE_GSRB) -I$(INCLUDE) -L$(LIB)

release-gcpusrb:
	$(CC) -DGCPUSRB $(HIPOPTS) $(SRC) -o $(EXE_GCPUSRB) -I$(INCLUDE) -L$(LIB)

clean:
	rm -f *~ *.o $(EXE_GSRB) $(EXE_GCPUSRB)
