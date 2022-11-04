#ifndef __CUDASEMAPHOREEBO_CU__
#define __CUDASEMAPHOREEBO_CU__

inline __host__ cudaError_t cudaSemaphoreCreateEBO(cudaSemaphore_t * const handle,
                                                   const int semaphoreNumber,
                                                   const unsigned int count,
                                                   const int NUM_SM)
{
  // Here we set the initial value to be count+1, this allows us to do an
  // atomicExch(sem, 0) and basically use the semaphore value as both a
  // lock and a semaphore.
  unsigned int initialValue = (count + 1), zero = 0;
  *handle = semaphoreNumber;
  for (int id = 0; id < NUM_SM; ++id) { // need to set these values for all SMs
    // Current count of the semaphore hence initialized to count+1
    cudaMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 5 * NUM_SM) + (id * 5))]), &initialValue, sizeof(initialValue), cudaMemcpyHostToDevice);
    // Lock variable initialized to 0
    cudaMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 5 * NUM_SM) + (id * 5)) + 1]), &zero, sizeof(zero), cudaMemcpyHostToDevice);
    // Writer waiting flag initialized to 0
    cudaMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 5 * NUM_SM) + (id * 5)) + 2]), &zero, sizeof(zero), cudaMemcpyHostToDevice);
    // Max count for the semaphore hence initialized it to count+1
    cudaMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 5 * NUM_SM) + (id * 5)) + 3]), &initialValue, sizeof(initialValue), cudaMemcpyHostToDevice);
     // Priority count initialized to 0
    cudaMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 5 * NUM_SM) + (id * 5)) + 4]), &zero, sizeof(zero), cudaMemcpyHostToDevice);  
  }
  return cudaSuccess;
}


inline __device__ bool cudaSemaphoreEBOTryWait(const cudaSemaphore_t sem,
                                               const bool isWriter,
                                               const unsigned int maxSemCount,
                                               unsigned int * semaphoreBuffers,
                                               const int NUM_SM)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_SM * 4 locations in the buffer.  Of these locations, each
    SM uses 4 of them (current count, head, tail, max count).  For the global 
    semaphore all SMs use semaphoreBuffers[sem * 4 * NUM_SM].
  */
  unsigned int * const currCount = semaphoreBuffers + (sem * 4 * NUM_SM);
  unsigned int * const lock = currCount + 1;
  /*
    Reuse the tail for the "writers are waiting" flag since tail is unused.
    For now just use to indicate that at least 1 writer is waiting instead of
    a count to make sure that readers aren't totally starved out until all the
    writers are done.
  */
  unsigned int * const writerWaiting = currCount + 2;
  __shared__ bool acq1, acq2;

  __syncthreads();
  if (isMasterThread)
  {
    acq1 = false;
    // try to acquire the sem head "lock"
    if (atomicCAS(lock, 0, 1) == 0) {
      // atomicCAS acts as a load acquire, need TF to enforce ordering
      __threadfence();
      acq1 = true;
    }
  }
  __syncthreads();

  if (!acq1) { return false; } // return if we couldn't acquire the lock
  if (isMasterThread)
  {
    acq2 = false;
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    unsigned int currSemCount = currCount[0];

    if (isWriter) {
      // writer needs the count to be == maxSemCount to enter the critical
      // section (otherwise there are readers in the critical section)
      if (currSemCount == maxSemCount) { acq2 = true; }
    } else {
      // if there is a writer waiting, readers aren't allowed to enter the
      // critical section
      if (writerWaiting[0] == 0) {
        // readers need count > 1 to enter critical section (otherwise semaphore
        // is full)
        if (currSemCount > 1) { acq2 = true; }
      }
    }
  }
  __syncthreads();

  if (!acq2) // release the sem head "lock" since the semaphore was full
  {
    // writers set a flag to note that they are waiting so more readers don't
    // join after the writer started waiting
    if (isWriter) { writerWaiting[0] = 1; /* if already 1, just reset to 1 */ }

    if (isMasterThread) {
      // atomicExch acts as a store release, need TF to enforce ordering
      __threadfence();
      atomicExch(lock, 0);
    }
    __syncthreads();
    return false;
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      /*
        writer decrements the current count of the semaphore by the max to
        ensure that no one else can enter the critical section while it's
        writing.
      */
      currCount[0] -= maxSemCount;

      // writers also need to unset the "writer is waiting" flag
      writerWaiting[0] = 0;
    } else {
      /*
        readers decrement the current count of the semaphore by 1 so other
        readers can also read the data (but not the writers since they needs
        the entire CS).
      */
      --currCount[0]; //atomicSub(currCount, 1);
    }

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();

  return true;
}

inline __device__ void cudaSemaphoreEBOWait(const cudaSemaphore_t sem,
                                            const bool isWriter,
                                            const unsigned int maxSemCount,
                                            unsigned int * semaphoreBuffers,
                                            const int NUM_SM)
{
  __shared__ int backoff;
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  volatile __shared__ int dummySum;

  if (isMasterThread)
  {
    backoff = 1;
    dummySum = 0;
  }
  __syncthreads();

  while (!cudaSemaphoreEBOTryWait(sem, isWriter, maxSemCount, semaphoreBuffers, NUM_SM))
  {
    __syncthreads();
    if (isMasterThread)
    {
      // if we failed to enter the semaphore, wait for a little while before
      // trying again
      for (int j = 0; j < backoff; ++j) { dummySum += j; }
      /*
        for writers increse backoff a lot because failing means readers are in
        the CS currently -- most important for non-unique because all TBs on
        all SMs are going for the same semaphore.
      */
      if (isWriter) {
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
      }
      else { backoff += 5; /* small, linear backoff increase for readers */ }
    }
    __syncthreads();
  }
  __syncthreads();
}

inline __device__ void cudaSemaphoreEBOPost(const cudaSemaphore_t sem,
                                            const bool isWriter,
                                            const unsigned int maxSemCount,
                                            unsigned int * semaphoreBuffers,
                                            const int NUM_SM)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_SM * 4 locations in the buffer.  Of these locations, each
    SM uses 4 of them (current count, head, tail, max count).  For the global
    semaphore use semaphoreBuffers[sem * 4 * NUM_SM].
  */
  unsigned int * const currCount = semaphoreBuffers + (sem * 4 * NUM_SM);
  unsigned int * const lock = currCount + 1;
  __shared__ bool acquired;

  if (isMasterThread) { acquired = false; }
  __syncthreads();

  while (!acquired)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // try to acquire sem head "lock"
      if (atomicCAS(lock, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        acquired = true;
      }
      else                            { acquired = false; }
    }
    __syncthreads();
  }

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      // writers add the max value to the semaphore to allow the readers to
      // start accessing the critical section.
      currCount[0] += maxSemCount;
    } else {
      ++currCount[0]; // readers add 1 to the semaphore
    }

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();
}

// same wait algorithm but with local scope and per-SM synchronization
inline __device__ bool cudaSemaphoreEBOTryWaitLocal(const cudaSemaphore_t sem,
                                                    const unsigned int smID,
                                                    const bool isWriter,
                                                    const unsigned int maxSemCount,
                                                    unsigned int * semaphoreBuffers,
                                                    const int NUM_SM)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_SM * 4 locations in the buffer.  Of these locations, each
    SM gets 4 of them (current count, head, tail, max count).  So SM 0 starts
    at semaphoreBuffers[sem * 4 * NUM_SM].
  */
  unsigned int * const currCount = semaphoreBuffers + ((sem * 4 * NUM_SM) +
                                                       (smID * 4));
  unsigned int * const lock = currCount + 1;
  /*
    Reuse the tail for the "writers are waiting" flag since tail is unused.
    For now just use to indicate that at least 1 writer is waiting instead of
    a count to make sure that readers aren't totally starved out until all the
    writers are done.
  */
  unsigned int * const writerWaiting = currCount + 2;
  __shared__ bool acq1, acq2;

  __syncthreads();
  if (isMasterThread)
  {
    acq1 = false;
    // try to acquire the sem head "lock"
    if (atomicCAS(lock, 0, 1) == 0) {
      // atomicCAS acts as a load acquire, need TF to enforce ordering locally
      __threadfence_block();
      acq1 = true;
    }
  }
  __syncthreads();

  if (!acq1) { return false; } // return if we couldn't acquire the lock
  if (isMasterThread)
  {
    acq2 = false;
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    unsigned int currSemCount = currCount[0];

    if (isWriter) {
      // writer needs the count to be == maxSemCount to enter the critical
      // section (otherwise there are readers in the critical section)
      if (currSemCount == maxSemCount) { acq2 = true; }
    } else {
      // if there is a writer waiting, readers aren't allowed to enter the
      // critical section
      if (writerWaiting[0] == 0) {
        // readers need count > 1 to enter critical section (otherwise semaphore
        // is full)
        if (currSemCount > 1) { acq2 = true; }
      }
    }
  }
  __syncthreads();

  if (!acq2) // release the sem head "lock" since the semaphore was full
  {
    // writers set a flag to note that they are waiting so more readers don't
    // join after the writer started waiting
    if (isWriter) { writerWaiting[0] = 1; /* if already 1, just reset to 1 */ }

    if (isMasterThread) {
      // atomicExch acts as a store release, need TF to enforce ordering locally
      __threadfence_block();
      atomicExch(lock, 0);
    }
    __syncthreads();
    return false;
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      /*
        writer decrements the current count of the semaphore by the max to
        ensure that no one else can enter the critical section while it's
        writing.
      */
      currCount[0] -= maxSemCount;

      // writers also need to unset the "writer is waiting" flag
      writerWaiting[0] = 0;
    } else {
      /*
        readers decrement the current count of the semaphore by 1 so other
        readers can also read the data (but not the writers since they needs
        the entire CS).
      */
      --currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();

  return true;
}

// same algorithm but with local scope
inline __device__ void cudaSemaphoreEBOWaitLocal(const cudaSemaphore_t sem,
                                                 const unsigned int smID,
                                                 const bool isWriter,
                                                 const unsigned int maxSemCount,
                                                 unsigned int * semaphoreBuffers,
                                                 const int NUM_SM)
{
  __shared__ int backoff;
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  volatile __shared__ int dummySum;

  if (isMasterThread)
  {
    backoff = 1;
    dummySum = 0;
  }
  __syncthreads();

  while (!cudaSemaphoreEBOTryWaitLocal(sem, smID, isWriter, maxSemCount, semaphoreBuffers, NUM_SM))
  {
    __syncthreads();
    if (isMasterThread)
    {
      // if we failed to enter the semaphore, wait for a little while before
      // trying again
      for (int j = 0; j < backoff; ++j) { dummySum += j; }
      // (capped) exponential backoff
      backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
    }
    __syncthreads();
  }
  __syncthreads();
}

inline __device__ void cudaSemaphoreEBOPostLocal(const cudaSemaphore_t sem,
                                                 const unsigned int smID,
                                                 const bool isWriter,
                                                 const unsigned int maxSemCount,
                                                 unsigned int * semaphoreBuffers,
                                                 const int NUM_SM)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  // Each sem has NUM_SM * 4 locations in the buffer.  Of these locations, each
  // SM gets 4 of them.  So SM 0 starts at semaphoreBuffers[sem * 4 * NUM_SM].
  unsigned int * const currCount = semaphoreBuffers + ((sem * 4 * NUM_SM) +
                                                       (smID * 4));
  unsigned int * const lock = currCount + 1;
  __shared__ bool acquired;

  if (isMasterThread) { acquired = false; }
  __syncthreads();

  while (!acquired)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // try to acquire sem head "lock"
      if (atomicCAS(lock, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();
        acquired = true;
      }
      else                            { acquired = false; }
    }
    __syncthreads();
  }

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      // writers add the max value to the semaphore to allow the readers to
      // start accessing the critical section.
      currCount[0] += maxSemCount;
    } else {
      ++currCount[0]; // readers add 1 to the semaphore
    }

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();
}

inline __device__ bool cudaSemaphoreEBOTryWaitPriority(const cudaSemaphore_t sem,
                                               const bool isWriter,
                                               const unsigned int maxSemCount,
                                               unsigned int * semaphoreBuffers,
                                               const int NUM_SM)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_SM * 5 locations in the buffer.  Of these locations, each
    SM uses 5 of them (current count, head, tail, max count, priority).  For the global 
    semaphore all SMs use semaphoreBuffers[sem * 5 * NUM_SM].
  */
  unsigned int * const currCount = semaphoreBuffers + (sem * 5 * NUM_SM);
  unsigned int * const lock = currCount + 1;
  /*
    Reuse the tail for the "writers are waiting" flag since tail is unused.

    For now just use to indicate that at least 1 writer is waiting instead of
    a count to make sure that readers aren't totally starved out until all the
    writers are done.
  */
  unsigned int * const writerWaiting = currCount + 2;
  unsigned int * const priority = currCount + 4;
  __shared__ int backoff;
  __shared__ bool acq1, acq2;

  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();
  if (isMasterThread)
  {
    acq1 = false;
    while(atomicCAS(priority, 0, 0) !=0){
      // Spinning until all blocks wanting to exit the semaphore have exited
      for (int i = 0; i < backoff; ++i) { ; }
      // Increase backoff to avoid repeatedly hammering priority flag
      backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
    }
     // try to acquire the sem head "lock"
    if (atomicCAS(lock, 0, 1) == 0) {
      // atomicCAS acts as a load acquire, need TF to enforce ordering
      __threadfence();
      acq1 = true;
    }
  }
  __syncthreads();

  if (!acq1) { return false; } // return if we couldn't acquire the lock
  if (isMasterThread)
  {
    acq2 = false;
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    unsigned int currSemCount = currCount[0];

    if (isWriter) {
      // writer needs the count to be == maxSemCount to enter the critical
      // section (otherwise there are readers in the critical section)
      if (currSemCount == maxSemCount) { acq2 = true; }
    } else {
      // if there is a writer waiting, readers aren't allowed to enter the
      // critical section
      if (writerWaiting[0] == 0) {
        // readers need count > 1 to enter critical section (otherwise semaphore
        // is full)
        if (currSemCount > 1) { acq2 = true; }
      }
    }
  }
  __syncthreads();

  if (!acq2) // release the sem head "lock" since the semaphore was full
  {
    // writers set a flag to note that they are waiting so more readers don't
    // join after the writer started waiting
    if (isWriter) { writerWaiting[0] = 1; /* if already 1, just reset to 1 */ }

    if (isMasterThread) {
      // atomicExch acts as a store release, need TF to enforce ordering
      __threadfence();
      atomicExch(lock, 0);
    }
    __syncthreads();
    return false;
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      /*
        writer decrements the current count of the semaphore by the max to
        ensure that no one else can enter the critical section while it's
        writing.
      */
      currCount[0] -= maxSemCount;

      // writers also need to unset the "writer is waiting" flag
      writerWaiting[0] = 0;
    } else {
      /*
        readers decrement the current count of the semaphore by 1 so other
        readers can also read the data (but not the writers since they needs
        the entire CS).
      */
      --currCount[0]; //atomicSub(currCount, 1);
    }

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();

  return true;
}

inline __device__ void cudaSemaphoreEBOWaitPriority(const cudaSemaphore_t sem,
                                            const bool isWriter,
                                            const unsigned int maxSemCount,
                                            unsigned int * semaphoreBuffers,
                                            const int NUM_SM)
{
  __shared__ int backoff;
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  volatile __shared__ int dummySum;

  if (isMasterThread)
  {
    backoff = 1;
    dummySum = 0;
  }
  __syncthreads();

  while (!cudaSemaphoreEBOTryWaitPriority(sem, isWriter, maxSemCount, semaphoreBuffers, NUM_SM))
  {
    __syncthreads();
    if (isMasterThread)
    {
      // if we failed to enter the semaphore, wait for a little while before
      // trying again
      for (int j = 0; j < backoff; ++j) { dummySum += j; }
      /*
        for writers increse backoff a lot because failing means readers are in
        the CS currently -- most important for non-unique because all TBs on
        all SMs are going for the same semaphore.
      */
      if (isWriter) {
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
      }
      else { backoff += 5; /* small, linear backoff increase for readers */ }
    }
    __syncthreads();
  }
  __syncthreads();
}

inline __device__ void cudaSemaphoreEBOPostPriority(const cudaSemaphore_t sem,
                                            const bool isWriter,
                                            const unsigned int maxSemCount,
                                            unsigned int * semaphoreBuffers,
                                            const int NUM_SM)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_SM * 5 locations in the buffer.  Of these locations, each
    SM uses 5 of them (current count, head, tail, max count, priority).  For the global
    semaphore use semaphoreBuffers[sem * 5 * NUM_SM].
  */
  unsigned int * const currCount = semaphoreBuffers + (sem * 5 * NUM_SM);
  unsigned int * const lock = currCount + 1;
  unsigned int * const priority = currCount + 4;
  __shared__ bool acquired;

  if (isMasterThread) 
  { 
    acquired = false;
    /*
    Incrementing priority count whenever a thread block wants to exit 
    the Semaphore. A priority count of > 0 will stop blocks trying to enter 
    the semaphore from making an attempt to acquire the lock, reducing contention
    */
    atomicAdd(priority, 1); 
  }
  __syncthreads();

  while (!acquired)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // try to acquire sem head "lock"
      if (atomicCAS(lock, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        acquired = true;
      }
      else                            { acquired = false; }
    }
    __syncthreads();
  }

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      // writers add the max value to the semaphore to allow the readers to
      // start accessing the critical section.
      currCount[0] += maxSemCount;
    } else {
      ++currCount[0]; // readers add 1 to the semaphore
    }

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
    // Decrement priority as thread block which wanted to exit has relenquished the lock
    atomicSub(priority, 1);
  }
  __syncthreads();
}

// same wait algorithm but with local scope and per-SM synchronization
inline __device__ bool cudaSemaphoreEBOTryWaitLocalPriority(const cudaSemaphore_t sem,
                                                    const unsigned int smID,
                                                    const bool isWriter,
                                                    const unsigned int maxSemCount,
                                                    unsigned int * semaphoreBuffers,
                                                    const int NUM_SM)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_SM * 5 locations in the buffer.  Of these locations, each
    SM gets 5 of them (current count, head, tail, max count, priority).  So SM 0 starts
    at semaphoreBuffers[sem * 5 * NUM_SM].
  */
  unsigned int * const currCount = semaphoreBuffers + ((sem * 5 * NUM_SM) +
                                                       (smID * 5));
  unsigned int * const lock = currCount + 1;
  /*
    Reuse the tail for the "writers are waiting" flag since tail is unused.

    For now just use to indicate that at least 1 writer is waiting instead of
    a count to make sure that readers aren't totally starved out until all the
    writers are done.
  */
  unsigned int * const writerWaiting = currCount + 2;
  unsigned int * const priority = currCount + 4;
  __shared__ int backoff;
  __shared__ bool acq1, acq2;

  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();
  if (isMasterThread)
  {
    acq1 = false;
    while(atomicCAS(priority, 0, 0) !=0){
      // Spinning until all blocks wanting to exit the semaphore have exited
      for (int i = 0; i < backoff; ++i) { ; }
      // Increase backoff to avoid repeatedly hammering priority flag
      backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
    }
    // try to acquire the sem head "lock"
    if (atomicCAS(lock, 0, 1) == 0) {
      // atomicCAS acts as a load acquire, need TF to enforce ordering locally
      __threadfence_block();
      acq1 = true;
    }
  }
  __syncthreads();

  if (!acq1) { return false; } // return if we couldn't acquire the lock
  if (isMasterThread)
  {
    acq2 = false;
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    unsigned int currSemCount = currCount[0];

    if (isWriter) {
      // writer needs the count to be == maxSemCount to enter the critical
      // section (otherwise there are readers in the critical section)
      if (currSemCount == maxSemCount) { acq2 = true; }
    } else {
      // if there is a writer waiting, readers aren't allowed to enter the
      // critical section
      if (writerWaiting[0] == 0) {
        // readers need count > 1 to enter critical section (otherwise semaphore
        // is full)
        if (currSemCount > 1) { acq2 = true; }
      }
    }
  }
  __syncthreads();

  if (!acq2) // release the sem head "lock" since the semaphore was full
  {
    // writers set a flag to note that they are waiting so more readers don't
    // join after the writer started waiting
    if (isWriter) { writerWaiting[0] = 1; /* if already 1, just reset to 1 */ }

    if (isMasterThread) {
      // atomicExch acts as a store release, need TF to enforce ordering locally
      __threadfence_block();
      atomicExch(lock, 0);
    }
    __syncthreads();
    return false;
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      /*
        writer decrements the current count of the semaphore by the max to
        ensure that no one else can enter the critical section while it's
        writing.
      */
      currCount[0] -= maxSemCount;

      // writers also need to unset the "writer is waiting" flag
      writerWaiting[0] = 0;
    } else {
      /*
        readers decrement the current count of the semaphore by 1 so other
        readers can also read the data (but not the writers since they needs
        the entire CS).
      */
      --currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();

  return true;
}

// same algorithm but with local scope
inline __device__ void cudaSemaphoreEBOWaitLocalPriority(const cudaSemaphore_t sem,
                                                 const unsigned int smID,
                                                 const bool isWriter,
                                                 const unsigned int maxSemCount,
                                                 unsigned int * semaphoreBuffers,
                                                 const int NUM_SM)
{
  __shared__ int backoff;
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  volatile __shared__ int dummySum;

  if (isMasterThread)
  {
    backoff = 1;
    dummySum = 0;
  }
  __syncthreads();

  while (!cudaSemaphoreEBOTryWaitLocalPriority(sem, smID, isWriter, maxSemCount, semaphoreBuffers, NUM_SM))
  {
    __syncthreads();
    if (isMasterThread)
    {
      // if we failed to enter the semaphore, wait for a little while before
      // trying again
      for (int j = 0; j < backoff; ++j) { dummySum += j; }
      // (capped) exponential backoff
      backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
    }
    __syncthreads();
  }
  __syncthreads();
}

inline __device__ void cudaSemaphoreEBOPostLocalPriority(const cudaSemaphore_t sem,
                                                 const unsigned int smID,
                                                 const bool isWriter,
                                                 const unsigned int maxSemCount,
                                                 unsigned int * semaphoreBuffers,
                                                 const int NUM_SM)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  // Each sem has NUM_SM * 5 locations in the buffer.  Of these locations, each
  // SM gets 5 of them.  So SM 0 starts at semaphoreBuffers[sem * 5 * NUM_SM].
  unsigned int * const currCount = semaphoreBuffers + ((sem * 5 * NUM_SM) +
                                                       (smID * 5));
  unsigned int * const lock = currCount + 1;
  unsigned int * const priority = currCount + 4;
  __shared__ bool acquired;

  if (isMasterThread) 
  { 
    acquired = false;
     /*
    Incrementing priority count whenever a thread block wants to exit 
    the Semaphore. A priority count of > 0 will stop blocks trying to enter 
    the semaphore from making an attempt to acquire the lock, reducing contention.
    */ 
    atomicAdd(priority, 1); 
  }
  __syncthreads();

  while (!acquired)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // try to acquire sem head "lock"
      if (atomicCAS(lock, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();
        acquired = true;
      }
      else                            { acquired = false; }
    }
    __syncthreads();
  }

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 TB at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      // writers add the max value to the semaphore to allow the readers to
      // start accessing the critical section.
      currCount[0] += maxSemCount;
    } else {
      ++currCount[0]; // readers add 1 to the semaphore
    }

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
    // Decrement priority as thread block which wanted to exit has relenquished the lock
    atomicSub(priority, 1);
  }
  __syncthreads();
}

#endif // #ifndef __CUDASEMAPHOREEBO_CU__
