#ifndef __CUDALOCKSBARRIERATOMICSRB_CU__

#define __CUDALOCKSBARRIERATOMICSRB_CU__

#include "cudaLocks.h"
__device__ __forceinline__ bool ld_gbl_cg (const bool *addr)
{
    short t;
#if defined(__LP64__) || defined(_WIN64)
    asm ("ld.global.cg.u8 %0, [%1];" : "=h"(t) : "l"(addr));
#else
    asm ("ld.global.cg.u8 %0, [%1];" : "=h"(t) : "r"(addr));
#endif
    return (bool)t;
}

inline __device__ void cudaBarrierAtomicNaiveSRB(unsigned int *globalBarr,
                                               // numBarr represents the number
                                               // of TBs going to the barrier
                                               const unsigned int numBarr,
                                               int backoff,
                                               const bool isMasterThread,
                                               bool *volatile global_sense) {
  __syncthreads();
  __shared__ bool s;
  if (isMasterThread) {
    s = !(ld_gbl_cg(global_sense));
    __threadfence();
    // atomicInc effectively adds 1 to atomic for each TB that's part of the
    // global barrier.
    atomicInc(globalBarr, 0x7FFFFFFF);
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
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF - 1));
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

inline __device__ void cudaBarrierAtomicSubSRB(unsigned int * globalBarr,
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
    // atomicInc acts as a store release, need TF to enforce ordering
    __threadfence();
    // atomicInc effectively adds 1 to atomic for each TB that's part of the
    // global barrier.
    atomicInc(globalBarr, 0x7FFFFFFF);
  }
  __syncthreads();

  while (*global_sense != *sense)
  {
    if (isMasterThread)
    {
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
        *global_sense = *sense;
      }
      else { // increase backoff to avoid repeatedly hammering global barrier
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
      }
    }
    __syncthreads();

    // do exponential backoff to reduce the number of times we pound the global
    // barrier
    if (*global_sense != *sense) {
      for (int i = 0; i < backoff; ++i) { ; }
      __syncthreads();
    }
  }
}

inline __device__ void cudaBarrierAtomicSRB(unsigned int * barrierBuffers,
                                            // numBarr represents the number of
                                            // TBs going to the barrier
                                            const unsigned int numBarr,
                                            const bool isMasterThread,
                                            bool * volatile sense,
                                            bool * volatile global_sense)
{
  unsigned int * atomic1 = barrierBuffers;
  __shared__ int backoff;

  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();
  cudaBarrierAtomicSubSRB(atomic1, numBarr, backoff, isMasterThread, sense, global_sense);
}

inline __device__ void cudaBarrierAtomicSubLocalSRB(unsigned int * perSMBarr,
                                                    const unsigned int numTBs_thisSM,
                                                    const bool isMasterThread,
                                                    bool * sense,
                                                    const int smID)

{
  __syncthreads();
  __shared__ bool s;
  if (isMasterThread)
  {
    s = !(*sense);
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

  while (*sense != s)
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
      }
    }
    __syncthreads();
  }
}

//Implements PerSM sense reversing barrier
inline __device__ void cudaBarrierAtomicLocalSRB(unsigned int * perSMBarrierBuffers,
                                                          const unsigned int smID,
                                                          const unsigned int numTBs_thisSM,
                                                          const bool isMasterThread,
                                                          const int MAX_BLOCKS)
{
  // each SM has MAX_BLOCKS locations in barrierBuffers, so my SM's locations
  // start at barrierBuffers[smID*MAX_BLOCKS]
  unsigned int * atomic1 = perSMBarrierBuffers + (smID * MAX_BLOCKS);
  bool *   sense = (bool *)(perSMBarrierBuffers + (smID * MAX_BLOCKS) + 2);

  cudaBarrierAtomicSubLocalSRB(atomic1, numTBs_thisSM, isMasterThread, sense, smID);
}

/*
  Helper function for joining the barrier with the atomic tree barrier.
*/
__device__ void joinBarrier_helperSRB(unsigned int * barrierBuffers,
                                      unsigned int * perSMBarrierBuffers,
                                      const unsigned int numBlocksAtBarr,
                                      const int smID,
                                      const int perSM_blockID,
                                      const int numTBs_perSM,
                                      const bool isMasterThread,
                                      const int MAX_BLOCKS) {
  bool * volatile  global_sense = (bool *)(barrierBuffers + 2);
  if (numTBs_perSM > 4) {
    bool * volatile sense = (bool *)(perSMBarrierBuffers + (smID * MAX_BLOCKS) + 2);
    bool * volatile done = (bool *)(barrierBuffers + 1);
    *done = 0;
    __syncthreads();
    cudaBarrierAtomicLocalSRB(perSMBarrierBuffers, smID, numTBs_perSM, isMasterThread, MAX_BLOCKS);
    // only 1 TB per SM needs to do the global barrier since we synchronized
    // the TBs locally first
    if (perSM_blockID == 0) {
      cudaBarrierAtomicSRB(barrierBuffers, numBlocksAtBarr, isMasterThread, sense, global_sense);  
      if(isMasterThread){
        *done = 1;
      }
      __syncthreads();
    }
    else {
      if(isMasterThread){
        while(ld_gbl_cg(done)) {;}
        __threadfence();
        while(ld_gbl_cg(global_sense) != ld_gbl_cg(sense)) {;}
      }
      
      __syncthreads();
    }
  } else { // For low contention just call 1 level barrier
    __shared__ int backoff;
    if (isMasterThread) {
      backoff = 1;
    }
    cudaBarrierAtomicNaiveSRB(barrierBuffers, gridDim.x , backoff, isMasterThread, global_sense);
  }
}

/*
  Helper function for joining the barrier with the naive atomic tree barrier 
  where all threads join the barrier.
*/
__device__ void joinBarrier_helperNaiveAllSRB(unsigned int * barrierBuffers,
					      unsigned int * perSMBarrierBuffers,
                                              const unsigned int numThreadsAtBarr,
					      const int smID,
                                              const int MAX_BLOCKS) {
  bool * volatile global_sense = (bool *)(barrierBuffers + 2);
  bool * volatile sense = (bool *)(perSMBarrierBuffers + (smID * MAX_BLOCKS) + 2);
  *sense = !(*global_sense);
  __syncthreads();
  // since all threads are joining, isMasterThread is "true" for all threads
  cudaBarrierAtomicSRB(barrierBuffers, numThreadsAtBarr, true, sense, global_sense);
}

/*
  Helper function for joining the barrier with the naive atomic tree barrier.
*/
__device__ void joinBarrier_helperNaiveSRB(unsigned int * barrierBuffers,
                                           const unsigned int numBlocksAtBarr,
					                                 const int smID,
                                           const bool isMasterThread,
                                           const int MAX_BLOCKS) {
  bool * volatile global_sense = (bool *)(barrierBuffers + 2);
  __shared__ int backoff;
  if (isMasterThread) {
    backoff = 1;
  }
  cudaBarrierAtomicNaiveSRB(barrierBuffers, numBlocksAtBarr, backoff, isMasterThread, global_sense);
}

#endif
