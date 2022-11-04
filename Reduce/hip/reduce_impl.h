#include "hip/hip_runtime.h"
// Author: Nic Olsen

#include <iostream>
#include "reduce.h"

__global__ void reduce_kernel(const int* g_idata, int* g_odata, unsigned int n) {
  HIP_DYNAMIC_SHARED( int, sdata)

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
  hipMallocManaged(&a, N * sizeof(int));
  hipMallocManaged(&b, N * sizeof(int));
  hipMemcpy(a, arr, N * sizeof(int), hipMemcpyHostToDevice);

  for (unsigned int n = N; n > 1; n = (n + threads_per_block - 1) / threads_per_block) {
    hipLaunchKernelGGL(reduce_kernel, dim3((n + threads_per_block - 1) / threads_per_block), dim3(threads_per_block), threads_per_block * sizeof(int), 0, a, b, n);

    // Swap input and output arrays
    int* tmp = a;
    a = b;
    b = tmp;
  }  
  hipDeviceSynchronize();

  int sum = a[0];

  hipFree(a);
  hipFree(b);

  return sum;
}
