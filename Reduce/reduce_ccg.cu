// Author: Nic Olsen

#include <iostream>
#include <stdio.h>
#include "reduce.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

 __global__ void reduce_kernel(int* g_idata, int* g_odata, unsigned int N, int* output /*, long long int* time */ ) {
    extern __shared__ int sdata[];
  
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int n = N; n > 1; n = (n + blockDim.x - 1) / blockDim.x){
    if (i < n) {
        sdata[tid] = g_idata[i];
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    // Sequential addressing alteration
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write out reduced portion of the output
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
    
  // __syncthreads();
   
    cg::grid_group grid = cg::this_grid(); 
    grid.sync();
 
   
    //__threadfence();
    
    int* tmp = g_idata;
    g_idata = g_odata;
    g_odata = tmp;
}

*output = g_idata[0];
}

__host__ int reduce(const int* arr, unsigned int N, unsigned int threads_per_block) {
    // Workspace NOTE: Could be smaller
    int* a;
    int* b;
    int* output;
    //long long int* time;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&output, sizeof(int));
    //cudaMallocManaged(&time, sizeof(long long int ));
    cudaMemcpy(a, arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    

    void *kernelArgs[] = {
        (void *)&a,  (void *)&b, (void *)&N, (void *)&output /*, (void *)&time */
    };
    cudaEventRecord(start);
      cudaLaunchCooperativeKernel((void*)reduce_kernel, ((N + threads_per_block - 1) / threads_per_block), threads_per_block,  kernelArgs, threads_per_block * sizeof(int), 0);
      cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("time (ms) is %f\n", ms) ;
    int sum = *output;

    cudaFree(a);
    cudaFree(b);

    return sum;
}
