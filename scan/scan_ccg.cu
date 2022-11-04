#include "scan.h"
#include <stdio.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MAX_BLOCK_SZ 64
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif




__global__
void gpu_sum_scan_naive(unsigned int* const d_out,
	const unsigned int* const d_in,
	const size_t numElems)
{
	// Using naive scan where each thread calculates a separate partial sum
	// Step complexity is still O(n) as the last thread will calculate the global sum

	unsigned int d_hist_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (d_hist_idx == 0 || d_hist_idx >= numElems)
	{
		return;
	}
	unsigned int cdf_val = 0;
	for (int i = 0; i < d_hist_idx; ++i)
	{
		cdf_val = cdf_val + d_in[i];
	}
	d_out[d_hist_idx] = cdf_val;
}

__global__
void gpu_sum_scan_blelloch(unsigned int* const d_out,
	const unsigned int* const d_in,
	unsigned int* const d_block_sums,
	const size_t numElems)
{
	extern __shared__ unsigned int s_out[];

	unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Zero out shared memory
	// Especially important when padding shmem for
	//  non-power of 2 sized input
	//s_out[2 * threadIdx.x] = 0;
	//s_out[2 * threadIdx.x + 1] = 0;
	s_out[threadIdx.x] = 0;
	s_out[threadIdx.x + blockDim.x] = 0;

	__syncthreads();

	// Copy d_in to shared memory per block
	//if (2 * glbl_tid < numElems)
	//{
	//	s_out[2 * threadIdx.x] = d_in[2 * glbl_tid];
	//	if (2 * glbl_tid + 1 < numElems)
	//		s_out[2 * threadIdx.x + 1] = d_in[2 * glbl_tid + 1];
	//}
	unsigned int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (cpy_idx < numElems)
	{
		s_out[threadIdx.x] = d_in[cpy_idx];
		if (cpy_idx + blockDim.x < numElems)
			s_out[threadIdx.x + blockDim.x] = d_in[cpy_idx + blockDim.x];
	}

	__syncthreads();

	// Reduce/Upsweep step

	// 2^11 = 2048, the max amount of data a block can blelloch scan
	unsigned int max_steps = 11; 

	unsigned int r_idx = 0;
	unsigned int l_idx = 0;
	unsigned int sum = 0; // global sum can be passed to host if needed
	unsigned int t_active = 0;
	for (int s = 0; s < max_steps; ++s)
	{
		t_active = 0;

		// calculate necessary indexes
		// right index must be (t+1) * 2^(s+1)) - 1
		r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
		if (r_idx >= 0 && r_idx < 2048)
			t_active = 1;

		if (t_active)
		{
			// left index must be r_idx - 2^s
			l_idx = r_idx - (1 << s);

			// do the actual add operation
			sum = s_out[l_idx] + s_out[r_idx];
		}
		__syncthreads();

		if (t_active)
			s_out[r_idx] = sum;
		__syncthreads();
	}

	// Copy last element (total sum of block) to block sums array
	// Then, reset last element to operation's identity (sum, 0)
	if (threadIdx.x == 0)
	{
		d_block_sums[blockIdx.x] = s_out[r_idx];
		s_out[r_idx] = 0;
	}

	__syncthreads();

	// Downsweep step

	for (int s = max_steps - 1; s >= 0; --s)
	{
		// calculate necessary indexes
		// right index must be (t+1) * 2^(s+1)) - 1
		r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
		if (r_idx >= 0 && r_idx < 2048)
		{
			t_active = 1;
		}

		unsigned int r_cpy = 0;
		unsigned int lr_sum = 0;
		if (t_active)
		{
			// left index must be r_idx - 2^s
			l_idx = r_idx - (1 << s);

			// do the downsweep operation
			r_cpy = s_out[r_idx];
			lr_sum = s_out[l_idx] + s_out[r_idx];
		}
		__syncthreads();

		if (t_active)
		{
			s_out[l_idx] = r_cpy;
			s_out[r_idx] = lr_sum;
		}
		__syncthreads();
	}

	// Copy the results to global memory
	//if (2 * glbl_tid < numElems)
	//{
	//	d_out[2 * glbl_tid] = s_out[2 * threadIdx.x];
	//	if (2 * glbl_tid + 1 < numElems)
	//		d_out[2 * glbl_tid + 1] = s_out[2 * threadIdx.x + 1];
	//}
	if (cpy_idx < numElems)
	{
		d_out[cpy_idx] = s_out[threadIdx.x];
		if (cpy_idx + blockDim.x < numElems)
			d_out[cpy_idx + blockDim.x] = s_out[threadIdx.x + blockDim.x];
	}



}

__global__
void gpu_add_block_sums(unsigned int* const d_out,
	const unsigned int* const d_in,
	unsigned int* const d_block_sums,
	const size_t numElems)
{
	//unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int d_block_sum_val = d_block_sums[blockIdx.x];

	//unsigned int d_in_val_0 = 0;
	//unsigned int d_in_val_1 = 0;

	// Simple implementation's performance is not significantly (if at all)
	//  better than previous verbose implementation
	unsigned int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (cpy_idx < numElems)
	{
		d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
		if (cpy_idx + blockDim.x < numElems)
			d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
	}

	//if (2 * glbl_t_idx < numElems)
	//{
	//	d_out[2 * glbl_t_idx] = d_in[2 * glbl_t_idx] + d_block_sum_val;
	//	if (2 * glbl_t_idx + 1 < numElems)
	//		d_out[2 * glbl_t_idx + 1] = d_in[2 * glbl_t_idx + 1] + d_block_sum_val;
	//}

	//if (2 * glbl_t_idx < numElems)
	//{
	//	d_in_val_0 = d_in[2 * glbl_t_idx];
	//	if (2 * glbl_t_idx + 1 < numElems)
	//		d_in_val_1 = d_in[2 * glbl_t_idx + 1];
	//}
	//else
	//	return;
	//__syncthreads();

	//d_out[2 * glbl_t_idx] = d_in_val_0 + d_block_sum_val;
	//if (2 * glbl_t_idx + 1 < numElems)
	//	d_out[2 * glbl_t_idx + 1] = d_in_val_1 + d_block_sum_val;
}

// Modified version of Mark Harris' implementation of the Blelloch scan
//  according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
__global__
void gpu_prescan(unsigned int* d_out,
	unsigned int*  d_in,
	unsigned int*  d_block_sums,
	unsigned int*  d_block_sums_dummy,
	unsigned int*  d_block_sums_dummy_2, 
	const unsigned int len,
	const unsigned int shmem_sz,
	const unsigned int max_elems_per_block)
{
	// Allocated on invocation
	extern __shared__ unsigned int s_out[];
	unsigned int* temp1;
	unsigned int* temp2;
	int thid;
	cg::grid_group grid = cg::this_grid();
	//int id = blockIdx.x * blockDim.x + threadIdx.x; 
	int ai;
	int bi;
    for(int a = len; a > 1; a = ((a+max_elems_per_block-1)/max_elems_per_block)){
	thid = threadIdx.x;
	ai = thid;
	bi = thid + blockDim.x;
	// Zero out the shared memory
	// Helpful especially when input size is not power of two
	
	s_out[thid] = 0;
	s_out[thid + blockDim.x] = 0;
	// If CONFLICT_FREE_OFFSET is used, shared memory
	//  must be a few more than 2 * blockDim.x
	if (thid + max_elems_per_block < shmem_sz)
		s_out[thid + max_elems_per_block] = 0;

	__syncthreads();

	if(blockIdx.x < ((a+max_elems_per_block-1)/max_elems_per_block)){
	// Copy d_in to shared memory
	// Note that d_in's elements are scattered into shared memory
	//  in light of avoiding bank conflicts
	unsigned int cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
	if (cpy_idx < a)
	{
		s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
	
		if( a == 2560){
		//printf("s[out] ai is %d and a is %d\n", d_in[cpy_idx], a);
		}
		
		if (cpy_idx + blockDim.x < a){
			s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
		
		if( a == 2560){
		//printf("s[out] bi is %d and a is %d\n", d_in[cpy_idx + blockDim.x], a);
		}
		
		}
		}
	__syncthreads();	

	// For both upsweep and downsweep:
	// Sequential indices with conflict free padding
	//  Amount of padding = target index / num banks
	//  This "shifts" the target indices by one every multiple
	//   of the num banks
	// offset controls the stride and starting index of 
	//  target elems at every iteration
	// d just controls which threads are active
	// Sweeps are pivoted on the last element of shared memory

	// Upsweep/Reduce step
	int offset = 1;
	for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			s_out[bi] += s_out[ai];
			if(a < len && threadIdx.x + blockDim.x*blockIdx.x == 0){
			//	printf("s[out] is %d and a is %d\n", s_out[bi], bi);	
			}
		}
		offset <<= 1;
	}

	__syncthreads();	

	// Save the total sum on the global block sums array
	// Then clear the last element on the shared memory
	if (thid == 0) 
	{ 
		d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
		s_out[max_elems_per_block - 1 
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
		//if( a < len){
		//printf("s[finalout] is %d and index is %d and a is %d\n", d_block_sums[blockIdx.x], max_elems_per_block - 1 
		//+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1), a);
		//s}
	}

	// Downsweep step
	for (int d = 1; d < max_elems_per_block; d <<= 1)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned int temp = s_out[ai];
			s_out[ai] = s_out[bi];
			s_out[bi] += temp;
		}
	}
	__syncthreads();

	// Copy contents of shared memory to global memory
	if (cpy_idx < a)
	{
		d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
		if( a < len){
			//printf("d[out] is %d and index is %d and a is %d \n", d_out[cpy_idx], cpy_idx, a );
		}
		if (cpy_idx + blockDim.x < a){
			d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
			if( a < len){
		//printf("d[out] is %d and index is %d nad a is %d\n", d_out[cpy_idx + blockDim.x], cpy_idx + blockDim.x, a);
			}
	}
}
    __syncthreads();
}
	grid.sync();
	//if(a==len && id < gridDim.x){
	//d_block_sums_2[id] = d_block_sums[id];
	//printf("block sum is  %d and a is %d\n",d_block_sums[id], a);
	//}
	if(a == len){
	temp1 = d_out;
	d_out = d_block_sums;
	d_in = d_block_sums;
	d_block_sums = d_block_sums_dummy;
	}
	
	else if ( a<len && a >1){
	temp2 = d_out;
	d_out = d_block_sums_dummy;
	d_in = d_block_sums_dummy;
	d_block_sums = d_block_sums_dummy_2;
	}
	
	__syncthreads();
}
 
 
/*
//unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
if(blockIdx.x < ((len+max_elems_per_block-1)/max_elems_per_block)){
unsigned int d_block_sum_val = d_block_sums_dummy[blockIdx.x];

//unsigned int d_in_val_0 = 0;
//unsigned int d_in_val_1 = 0;

// Simple implementation's performance is not significantly (if at all)
//  better than previous verbose implementation
unsigned int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
if (cpy_idx < ((len+max_elems_per_block-1)/max_elems_per_block))
{
	temp2[cpy_idx] = temp2[cpy_idx] + d_block_sum_val;
	if (cpy_idx + blockDim.x < ((len+max_elems_per_block-1)/max_elems_per_block))
		temp2[cpy_idx + blockDim.x] = temp2[cpy_idx + blockDim.x] + d_block_sum_val;
}
}
grid.sync();
*/
//d_out = temp1;
//d_block_sums =  temp2;

}

void sum_scan_naive(unsigned int* const d_out,
	const unsigned int* const d_in,
	const size_t numElems)
{
	unsigned int blockSz = MAX_BLOCK_SZ;
	unsigned int gridSz = numElems / blockSz;
	if (numElems % blockSz != 0)
		gridSz += 1;
	checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(unsigned int)));
	gpu_sum_scan_naive << <gridSz, blockSz >> >(d_out, d_in, numElems);
}
 
void sum_scan_blelloch(unsigned int* d_out,
	unsigned int*  d_in,
	const size_t numElems)
{
	// Zero out d_out
	checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(unsigned int)));

	// Set up number of threads and blocks
	
	unsigned int block_sz = MAX_BLOCK_SZ / 2;
	unsigned int max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
	//unsigned int grid_sz = (unsigned int) std::ceil((double) numElems / (double) max_elems_per_block);
	// UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically  
	//  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
	unsigned int grid_sz = (numElems + max_elems_per_block -1) / max_elems_per_block;
	// Take advantage of the fact that integer division drops the decimals
	if (numElems % max_elems_per_block != 0) 
		grid_sz += 1;

	// Conflict free padding requires that shared memory be more than 2 * block_sz
	unsigned int shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks
	unsigned int* d_block_sums;
	unsigned int* d_block_sums_dummy_2;
	unsigned int* d_block_sums_dummy;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMalloc(&d_block_sums_dummy, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMemset(d_block_sums_dummy, 0, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMalloc(&d_block_sums_dummy_2, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMemset(d_block_sums_dummy_2, 0, sizeof(unsigned int) * grid_sz));

	// Sum scan data allocated to each block
	//gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
	void *kernelArgs[] = {
        (void *)&d_out,  (void *)&d_in, (void *)&d_block_sums,  (void *)&d_block_sums_dummy, (void *)&d_block_sums_dummy_2,  (void *)&numElems, (void *)&shmem_sz, (void *)&max_elems_per_block
	};
	int numBlocksPerSm;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, gpu_prescan, block_sz, 0);

	cudaEvent_t start;
		cudaEvent_t stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	
		cudaEventRecord(start);
		cudaLaunchCooperativeKernel((void*)gpu_prescan, grid_sz, block_sz,  kernelArgs, sizeof(unsigned int) * shmem_sz, 0);
		cudaEventRecord(stop);
		cudaDeviceSynchronize();
		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		std::cout << "Kernel time (ms) " << ms << std::endl;


		 
	/*
	gpu_prescan<<<grid_sz, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_out, 
																	d_in, 
																	d_block_sums,
																	d_block_sums_2,
																	d_block_sums_dummy, 
																	d_block_sums_dummy_2,
																	numElems, 
																	shmem_sz,
																	max_elems_per_block);
*/
	// Sum scan total sums produced by each block
	// Use basic implementation if number of total sums is <= 2 * block_sz
	//  (This requires only one block to do the scan)
	// Else, recurse on this same function as you'll need the full-blown scan
	//  for the block sums
	
	//// Uncomment to examine block sums

	int max_elems = (grid_sz + max_elems_per_block -1)/max_elems_per_block;

	//checkCudaErrors(cudaMemcpy(h_block_sums_out, d_out, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToHost));

	// Add each block's total sum to its scan output
	// in order to get the final, global scanned array
    gpu_add_block_sums<<<max_elems, block_sz>>>(d_block_sums, d_block_sums, d_block_sums_dummy, grid_sz);
	gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

	 checkCudaErrors(cudaFree(d_block_sums));
	 checkCudaErrors(cudaFree(d_block_sums_dummy));
	 checkCudaErrors(cudaFree(d_block_sums_dummy_2));
	
}
