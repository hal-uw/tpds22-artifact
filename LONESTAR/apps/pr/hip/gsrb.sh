nvcc -DTHRUST_IGNORE_CUB_VERSION_CHECK -dc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_70  -g -O3 -w  -I../../rt/include -I../../rt/include/mgpu/include  -I../../rt/include/cub ../../skelapp/skel.cu -o skel.o
nvcc -DTHRUST_IGNORE_CUB_VERSION_CHECK -dc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_70  -g -O3 -w  -I../../rt/include -I../../rt/include/mgpu/include  -I../../rt/include/cub kernel_gsrb.cu  -o kernel.o
nvcc -DTHRUST_IGNORE_CUB_VERSION_CHECK -dc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_70  -g -O3 -w  -I../../rt/include -I../../rt/include/mgpu/include  -I../../rt/include/cub support.cu  -o support.o
nvcc -DTHRUST_IGNORE_CUB_VERSION_CHECK -g -g -O3 -w -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=sm_70   -I../../rt/include -I../../rt/include/mgpu/include  -I../../rt/include/cub -L../../rt/lib  -o pr_gsrb skel.o kernel.o support.o ../../skelapp/mgpucontext.o ../../skelapp/mgpuutil.o -lggrt -lcurand -lcudadevrt -lz
cp pr_gsrb ../../bin

