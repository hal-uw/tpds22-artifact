// Author: Nic Olsen

#include <iostream>
#include "reduce.cuh"

__global__ void reduce_kernel(const int* g_idata, int* g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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

}

__host__ int reduce(const int* arr, unsigned int N, unsigned int threads_per_block) {
    // Workspace NOTE: Could be smaller
    int* a;
    int* b;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMemcpy(a, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    for (unsigned int n = N; n > 1; n = (n + threads_per_block - 1) / threads_per_block) {
        reduce_kernel<<<(n + threads_per_block - 1) / threads_per_block, threads_per_block,
                        threads_per_block * sizeof(int)>>>(a, b, n);

        // Swap input and output arrays
        int* tmp = a;
        a = b;
        b = tmp;
  }  
    cudaDeviceSynchronize();

    int sum = a[0];

    cudaFree(a);
    cudaFree(b);

    return sum;
}
