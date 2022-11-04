 #!/bin/bash
 nvcc -O3  -gencode arch=compute_75,code=compute_75 -gencode arch=compute_70,code=compute_70  -gencode arch=compute_61,code=compute_61  -o reduce_gsrb reduce_gsrb.cu test_reduce.cu
 nvcc -O3  -gencode arch=compute_75,code=compute_75 -gencode arch=compute_70,code=compute_70  -gencode arch=compute_61,code=compute_61  -o reduce_gcpusrb reduce_gcpusrb.cu test_reduce.cu
 nvcc -O3 -rdc=true -gencode arch=compute_75,code=compute_75 -gencode arch=compute_70,code=compute_70  -gencode arch=compute_61,code=compute_61  -o reduce_ccg reduce_ccg.cu test_reduce.cu
