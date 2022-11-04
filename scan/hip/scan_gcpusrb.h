#include "hip/hip_runtime.h"
#include "scan.h"
#include <stdio.h>

#define MAX_BLOCK_SZ 64
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n)                         \
  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif
__device__ __forceinline__ bool ld_gbl_cg (const bool *addr)
{
  uint64_t out;
  // use GLC modifier to cache the load in L2 and bypass L1
  asm volatile (
		"flat_load_dwordx2 %0, %1 glc\n"
		"s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
		: "=v"(out) : "v"(addr): "memory"
		);
  return (bool)out;
}

inline __device__ void hipBarrierAtomicNaiveSRB(unsigned int *globalBarr,
                                                // numBarr represents the number
                                                // of TBs going to the barrier
                                                const unsigned int numBarr,
                                                int backoff,
                                                const bool isMasterThread,
                                                bool * volatile global_sense) {
  __syncthreads();
  __shared__ bool s;
  if (isMasterThread) {
    s = !(ld_gbl_cg(global_sense));
    __threadfence();
    // atomicInc effectively adds 1 to atomic for each TB that's part of the
    // global barrier.
    atomicInc(globalBarr, 0x7FFFFFFF);
    //printf("Global barr is %d and numbarr is %d\n", *globalBarr, numBarr);
  }
  __syncthreads();
  
  while (ld_gbl_cg(global_sense) != s) {
    if (isMasterThread) {
      /*
        Once the atomic's value == numBarr, then reset the value to 0 and
        proceed because all of the TBs have reached the global barrier.
      */
      if (atomicCAS(globalBarr, numBarr, 0) == numBarr) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        *global_sense = s;
      } else { // increase backoff to avoid repeatedly hammering global barrier
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (1024 - 1));
      }
    }
    __syncthreads();
  
    // do exponential backoff to reduce the number of times we pound the global
    // barrier
    if (ld_gbl_cg(global_sense) != s) {
      for (int i = 0; i < backoff; ++i) {
        ;
      }
      __syncthreads();
    }
  }
}
  
inline __device__ void hipBarrierAtomicSubSRB(unsigned int * globalBarr,
                                              // numBarr represents the number of
                                              // TBs going to the barrier
                                              const unsigned int numBarr,
                                              int backoff,
                                              const bool isMasterThread,
                                              bool * volatile sense,
                                              bool * volatile global_sense)
{
  __syncthreads();
  if (isMasterThread)
    {
      //printf("Inside global Barrier for blockID %d and sense is %d and global sense is %d\n", blockIdx.x, *sense, *global_sense);
      // atomicInc acts as a store release, need TF to enforce ordering
      __threadfence();
      // atomicInc effectively adds 1 to atomic for each TB that's part of the
      // global barrier.
      atomicInc(globalBarr, 0x7FFFFFFF);
      //printf("Global barr is %d and numBarr is %d\n", *globalBarr, numBarr);
    }
  __syncthreads();

  while (*global_sense != ld_gbl_cg(sense))
    {
      if (isMasterThread)
        {
          //printf("Global sense hili\n");
          /*
            For the tree barrier we expect only 1 TB from each SM to enter the
            global barrier.  Since we are assuming an equal amount of work for all
            SMs, we can use the # of TBs reaching the barrier for the compare value
            here.  Once the atomic's value == numBarr, then reset the value to 0 and
            proceed because all of the TBs have reached the global barrier.
          */
          if (atomicCAS(globalBarr, numBarr, 0) == numBarr) {
            // atomicCAS acts as a load acquire, need TF to enforce ordering
            __threadfence();
            *global_sense = ld_gbl_cg(sense);
          }
          else { // increase backoff to avoid repeatedly hammering global barrier
            // (capped) exponential backoff
            backoff = (((backoff << 1) + 1) & (1024-1));
          }
        }
      __syncthreads();

      // do exponential backoff to reduce the number of times we pound the global
      // barrier
      if (isMasterThread) {
        //for (int i = 0; i < backoff; ++i) { ; }
      }
      __syncthreads();
    }
}

inline __device__ void hipBarrierAtomicSRB(unsigned int * barrierBuffers,
                                           // numBarr represents the number of
                                           // TBs going to the barrier
                                           const unsigned int numBarr,
                                           const bool isMasterThread,
                                           bool * volatile sense,
                                           bool * volatile global_sense)
{
  __shared__ int backoff;

  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();

  hipBarrierAtomicSubSRB(barrierBuffers, numBarr, backoff, isMasterThread, sense, global_sense);
}

inline __device__ void hipBarrierAtomicSubLocalSRB(unsigned int * perSMBarr,
                                                   const unsigned int numTBs_thisSM,
                                                   const bool isMasterThread,
                                                   bool * sense,
                                                   const int smID,
                                                   unsigned int* last_block)

{
  __syncthreads();
  __shared__ bool s;
  if (isMasterThread)
    {
      s = !(ld_gbl_cg(sense));
      // atomicInc acts as a store release, need TF to enforce ordering locally
      __threadfence_block();
      /*
        atomicInc effectively adds 1 to atomic for each TB that's part of the
        barrier.  For the local barrier, this requires using the per-CU
        locations.
      */
      atomicInc(perSMBarr, 0x7FFFFFFF);
    }
  __syncthreads();

  while (ld_gbl_cg(sense) != s)
    {
      if (isMasterThread)
        {
          /*
            Once all of the TBs on this SM have incremented the value at atomic,
            then the value (for the local barrier) should be equal to the # of TBs
            on this SM.  Once that is true, then we want to reset the atomic to 0
            and proceed because all of the TBs on this SM have reached the local
            barrier.
          */
          if (atomicCAS(perSMBarr, numTBs_thisSM, 0) == numTBs_thisSM) {
            // atomicCAS acts as a load acquire, need TF to enforce ordering
            // locally
            __threadfence_block();
            *sense = s;
            *last_block = blockIdx.x;
          }
        }
      __syncthreads();
    }
}

//Implements PerSM sense reversing barrier
inline __device__ void hipBarrierAtomicLocalSRB(unsigned int * perSMBarrierBuffers,
                                                unsigned int * last_block,
                                                const unsigned int smID,
                                                const unsigned int numTBs_thisSM,
                                                const bool isMasterThread,
                                                bool* sense)
{
  // each SM has MAX_BLOCKS locations in barrierBuffers, so my SM's locations
  // start at barrierBuffers[smID*MAX_BLOCKS]
  hipBarrierAtomicSubLocalSRB(perSMBarrierBuffers, numTBs_thisSM, isMasterThread, sense, smID, last_block);
}

/*
  Helper function for joining the barrier with the atomic tree barrier.
*/
__device__ void joinBarrier_helperSRB(bool * volatile global_sense,
                                      bool * volatile perSMsense,
                                      bool * done,
                                      unsigned int* global_count,
                                      unsigned int* local_count,
                                      unsigned int* last_block,
                                      const unsigned int numBlocksAtBarr,
                                      const int smID,
                                      const int perSM_blockID,
                                      const int numTBs_perSM,
                                      const bool isMasterThread,
                                      bool naive) {                                 
  //*done = 0;
  __syncthreads();
  if (numTBs_perSM > 1 && naive == false) {
    hipBarrierAtomicLocalSRB(&local_count[smID], &last_block[smID], smID, numTBs_perSM, isMasterThread, &perSMsense[smID]);

    // only 1 TB per SM needs to do the global barrier since we synchronized
    // the TBs locally first
    if (blockIdx.x == last_block[smID]) {
      hipBarrierAtomicSRB(global_count, numBlocksAtBarr, isMasterThread , &perSMsense[smID], global_sense);  

    }
    else {
      if(isMasterThread){
        while (*global_sense != ld_gbl_cg(&perSMsense[smID])){  
          __threadfence();
        }
      }
      __syncthreads();
    }    
  } else { // if only 1 TB on the SM, no need for the local barriers
    __shared__ int backoff;
    if (isMasterThread) {
      backoff = 1;
    }
    __syncthreads();
    hipBarrierAtomicNaiveSRB(global_count, (numBlocksAtBarr*numTBs_perSM), backoff,  isMasterThread,  global_sense);
  }
}


__device__ void kernelAtomicTreeBarrierUniqSRB( bool * volatile global_sense,
                                                bool * volatile perSMsense,
                                                bool * done,
                                                unsigned int* global_count,
                                                unsigned int* local_count,
                                                unsigned int* last_block,
                                                const int NUM_SM,
                                                bool naive)
{

  // local variables
  // thread 0 is master thread
  const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) &&
                               (threadIdx.z == 0));
  // represents the number of TBs going to the barrier (max NUM_SM, gridDim.x if
  // fewer TBs than SMs).
  const unsigned int numBlocksAtBarr = ((gridDim.x < NUM_SM) ? gridDim.x :
                                        NUM_SM);
  const int smID = (blockIdx.x % numBlocksAtBarr); // mod by # SMs to get SM ID
  // all thread blocks on the same SM access unique locations because the
  // barrier can't ensure DRF between TBs
  const int perSM_blockID = (blockIdx.x / numBlocksAtBarr);
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs

  int numTBs_perSM = (int)ceil((float)gridDim.x / numBlocksAtBarr);


  joinBarrier_helperSRB(global_sense, perSMsense, done, global_count, local_count, last_block,
                        numBlocksAtBarr, smID, perSM_blockID, numTBs_perSM,
                        isMasterThread, naive);

}





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
  HIP_DYNAMIC_SHARED( unsigned int, s_out)

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
  //    s_out[2 * threadIdx.x] = d_in[2 * glbl_tid];
  //    if (2 * glbl_tid + 1 < numElems)
  //            s_out[2 * threadIdx.x + 1] = d_in[2 * glbl_tid + 1];
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
  //    d_out[2 * glbl_tid] = s_out[2 * threadIdx.x];
  //    if (2 * glbl_tid + 1 < numElems)
  //            d_out[2 * glbl_tid + 1] = s_out[2 * threadIdx.x + 1];
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
  //    d_out[2 * glbl_t_idx] = d_in[2 * glbl_t_idx] + d_block_sum_val;
  //    if (2 * glbl_t_idx + 1 < numElems)
  //            d_out[2 * glbl_t_idx + 1] = d_in[2 * glbl_t_idx + 1] + d_block_sum_val;
  //}

  //if (2 * glbl_t_idx < numElems)
  //{
  //    d_in_val_0 = d_in[2 * glbl_t_idx];
  //    if (2 * glbl_t_idx + 1 < numElems)
  //            d_in_val_1 = d_in[2 * glbl_t_idx + 1];
  //}
  //else
  //    return;
  //__syncthreads();

  //d_out[2 * glbl_t_idx] = d_in_val_0 + d_block_sum_val;
  //if (2 * glbl_t_idx + 1 < numElems)
  //    d_out[2 * glbl_t_idx + 1] = d_in_val_1 + d_block_sum_val;
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
                 const unsigned int max_elems_per_block,
                 bool * volatile global_sense,
                 bool * volatile perSMsense,
                 bool * done,
                 unsigned int* global_count,
                 unsigned int* local_count,
                 unsigned int* last_block,
                 const int NUM_SM,
                 bool naive)
{
  // Allocated on invocation
  HIP_DYNAMIC_SHARED( unsigned int, s_out)
    unsigned int* temp1;
  unsigned int* temp2;
  int thid;
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
                
          if (cpy_idx + blockDim.x < a){
            s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
                
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
                //      printf("s[out] is %d and a is %d\n", s_out[bi], bi);    
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
    kernelAtomicTreeBarrierUniqSRB(global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM, naive);
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
  checkHipErrors(hipMemset(d_out, 0, numElems * sizeof(unsigned int)));
  hipLaunchKernelGGL(gpu_sum_scan_naive, dim3(gridSz), dim3(blockSz), 0, 0, d_out, d_in, numElems);
}
 
void sum_scan_blelloch(unsigned int* d_out,
                       unsigned int*  d_in,
                       const size_t numElems)
{
  // Zero out d_out
  checkHipErrors(hipMemset(d_out, 0, numElems * sizeof(unsigned int)));

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
  //if (numElems % max_elems_per_block != 0) 
  //grid_sz += 1;

  // Conflict free padding requires that shared memory be more than 2 * block_sz
  unsigned int shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

  // Allocate memory for array of total sums produced by each block
  // Array length must be the same as number of blocks
  unsigned int* d_block_sums;
  unsigned int* d_block_sums_dummy_2;
  unsigned int* d_block_sums_dummy;
  checkHipErrors(hipMalloc(&d_block_sums, sizeof(unsigned int) * grid_sz));
  checkHipErrors(hipMemset(d_block_sums, 0, sizeof(unsigned int) * grid_sz));
  checkHipErrors(hipMalloc(&d_block_sums_dummy, sizeof(unsigned int) * grid_sz));
  checkHipErrors(hipMemset(d_block_sums_dummy, 0, sizeof(unsigned int) * grid_sz));
  checkHipErrors(hipMalloc(&d_block_sums_dummy_2, sizeof(unsigned int) * grid_sz));
  checkHipErrors(hipMemset(d_block_sums_dummy_2, 0, sizeof(unsigned int) * grid_sz));
    
  // Sum scan data allocated to each block
  //hipLaunchKernelGGL(gpu_sum_scan_blelloch, dim3(grid_sz), dim3(block_sz), sizeof(unsigned int) * max_elems_per_block , 0, d_out, d_in, d_block_sums, numElems);
  //Barrier Stuff
        
  unsigned int* global_count;
  unsigned int* local_count; 
  unsigned int *last_block;
  bool * volatile global_sense;
  bool* volatile perSMsense;
  bool * done;
  bool naive = true;
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  int NUM_SM = deviceProp.multiProcessorCount;
  hipMallocManaged((void **)&global_sense,sizeof(bool));
  hipMallocManaged((void **)&done,sizeof(bool));
  hipMallocManaged((void **)&perSMsense,NUM_SM*sizeof(bool));
  hipMallocManaged((void **)&last_block,sizeof(unsigned int)*(NUM_SM));
  hipMallocManaged((void **)&local_count,  NUM_SM*sizeof(unsigned int));
  hipMallocManaged((void **)&global_count,sizeof(unsigned int));
    
  hipMemset(global_sense, false, sizeof(bool));
  hipMemset(done, false, sizeof(bool));
  hipMemset(global_count, 0, sizeof(unsigned int));

  for (int i = 0; i < NUM_SM; ++i) {
    hipMemset(&perSMsense[i], false, sizeof(bool));
    hipMemset(&local_count[i], 0, sizeof(unsigned int));
    hipMemset(&last_block[i], 0, sizeof(unsigned int));
  }
  //hipLaunchCooperativeKernel((void*)gpu_prescan, grid_sz, block_sz,  kernelArgs, sizeof(unsigned int) * shmem_sz, 0);
  hipEvent_t start;
  hipEvent_t stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
        
  hipEventRecord(start);
  hipLaunchKernelGGL(gpu_prescan, dim3(grid_sz), dim3(block_sz), sizeof(unsigned int) * shmem_sz, 0, d_out, 
                     d_in, 
                     d_block_sums,
                     d_block_sums_dummy, 
                     d_block_sums_dummy_2,
                     numElems, 
                     shmem_sz,
                     max_elems_per_block,
                     global_sense, 
                     perSMsense, 
                     done, 
                     global_count, 
                     local_count, 
                     last_block, 
                     NUM_SM,
                     naive
                     );

  hipEventRecord(stop);
  hipDeviceSynchronize();
  float ms;
  hipEventElapsedTime(&ms, start, stop);
  std::cout << "Kernel time (ms) " << ms << std::endl;
  // Sum scan total sums produced by each block
  // Use basic implementation if number of total sums is <= 2 * block_sz
  //  (This requires only one block to do the scan)
  // Else, recurse on this same function as you'll need the full-blown scan
  //  for the block sums
        
  //// Uncomment to examine block sums

  int max_elems = (grid_sz + max_elems_per_block -1)/max_elems_per_block;

  //checkHipErrors(hipMemcpy(h_block_sums_out, d_out, sizeof(unsigned int) * grid_sz, hipMemcpyDeviceToHost));

  // Add each block's total sum to its scan output
  // in order to get the final, global scanned array
  hipLaunchKernelGGL(gpu_add_block_sums, dim3(max_elems), dim3(block_sz), 0, 0, d_block_sums, d_block_sums, d_block_sums_dummy, grid_sz);
  hipLaunchKernelGGL(gpu_add_block_sums, dim3(grid_sz), dim3(block_sz), 0, 0, d_out, d_out, d_block_sums, numElems);

  checkHipErrors(hipFree(d_block_sums));
  checkHipErrors(hipFree(d_block_sums_dummy));
  checkHipErrors(hipFree(d_block_sums_dummy_2));
}
