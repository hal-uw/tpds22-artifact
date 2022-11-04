HIPOPTS = --amdgpu-target=gfx906,gfx900

HIP_PATH ?= /opt/rocm/hip

CC := $(HIP_PATH)/bin/hipcc

INCLUDE := 
LIB := 

SRC = main.hip.cc

EXE_GSRB = scan_gsrb
EXE_GCPUSRB = scan_gcpusrb

all: release-gsrb release-gcpusrb

release-gsrb:
	$(CC) -DGSRB $(HIPOPTS) $(SRC) -o $(EXE_GSRB) -I$(INCLUDE) -L$(LIB)

release-gcpusrb:
	$(CC) -DGCPUSRB $(HIPOPTS) $(SRC) -o $(EXE_GCPUSRB) -I$(INCLUDE) -L$(LIB)

clean:
	rm -f *~ *.o $(EXE_GSRB) $(EXE_GCPUSRB)
