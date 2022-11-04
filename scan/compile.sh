 #!/bin/bash
 nvcc -O3  -Xcompiler  -m64  -gencode arch=compute_75,code=compute_75 -gencode arch=compute_70,code=compute_70  -gencode arch=compute_61,code=compute_61  -w -o scan_gsrb scan_gsrb.cu main.cu
 nvcc -O3  -Xcompiler -m64 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_70,code=compute_70  -gencode arch=compute_61,code=compute_61 -w  -o scan_gcpusrb scan_gcpusrb.cu main.cu
 nvcc -O3 -Xcompiler -m64 -rdc=true -gencode arch=compute_75,code=compute_75 -gencode arch=compute_70,code=compute_70  -gencode arch=compute_61,code=compute_61 -w  -o scan_ccg scan_ccg.cu main.cu
