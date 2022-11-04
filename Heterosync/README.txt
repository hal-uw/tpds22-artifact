BACKGROUND INFORMATION
----------------------

Structure: All of the HeteroSync microbenchmarks are run from a single main function.  Each of the microbenchmarks has a separate .cu (CUDA) file that contains the code for its lock and unlock functions.

Contents: The following Synchronization Primitives (SyncPrims) microbenchmarks are included in HeteroSync:

- Centralized Mutexes:
	1.  Spin Mutex Lock: A fairly standard spin-lock implementation.  It repeatedly tries to obtain the lock.  This version has high contention and a lot of atomic accesses since all TBs are spinning on the same lock variable.
	2.  Spin Mutex Lock with Backoff: Standard backoff version of a spin lock where they “sleep” for a short period of time between each unsuccessful acquire.  They use a linear backoff instead of exponential backoff.  On the first failed acquire they will “sleep” for I_min; every subsequent failed read will increase the “sleep” time (up to I_max).
	3.  Fetch-and-Add (FA) Mutex Lock (similar to Ticket/Queue-style Locks): To make their spin lock fair and have a deterministic number of atomic accesses per operation they also implement this queue-style spin lock.  Every TB uses an atomic to get a "ticket" for when they'll get the lock.  The TBs poll the “current ticket” location until their turn arrives (when it does they acquire the lock).  FAMutex uses backoff in the polling section of this lock to reduce contention.
	4.  Ring Buffer-based Sleeping Mutex Lock: Each TB places itself on the end of the buffer and repeatedly checks if is now at the front of the buffer.  To unlock they increment the head pointer.  In the original paper they found that performance is bad for this one because it requires more reads and writes to the head pointer are serialized.
- Centralized Semaphores:
	1.  Spin Lock Semaphore: To approximate the "perform OP if > 0" feature of semaphores (on CPUs) they use atomicExch's to block the TB until the exchange returns true.  Requires more reads and writes on a GPU than a mutex.  Each TB sets the semaphore to the appropriate new values in the post and wait phases depending on the current capacity of the semaphore.
	2.  Spin Lock Semaphore with Backoff: As with the mutexes, they add a linear backoff to decrease contention.  The backoff is only in the wait() phase because usually more TBs are waiting, not posting.
- Barriers:
	1.  Atomic Barrier: a two-stage atomic counter barrier.  There are several versions of this barrier: a tree barrier and a second version that exchanges data locally on a CU before joining the global tree barrier.
	2.  Lock-Free Barrier: a decentralized, sleeping based approach that doesn't require atomics.  Each TB sets a flag in a distinct memory location.  Once all TBs have set their flag, then each TB does an intra-block barrier between its warps.  Like the atomic barrier, there are two versions.

All microbenchmarks access shared data that requires synchronization.

A subsequent commit will add the Relaxed Atomics microbenchmarks discussed in our paper.

USAGE
-----

Compilation:

Since all of the microbenchmarks run from a single main function, users only need to compile the entire suite once in order to use any of the microbenchmarks.  You will need to set CUDA_DIR in the Makefile in order to properly compile the code.
Please use "make" to compile. Before compiling please set CUDA_DIR in the makefile to CUDA 11.0

Running:

The usage of the microbenchmarks is as follows:

./allSyncPrims-1kernel <syncPrim> <numLdSt> <numTBs> <numCSIters>

        <syncPrim>: a string that represents which synchronization primitive to run.
                atomicTreeBarrUniq - Atomic Tree Barrier
                atomicTreeBarrUniqSRB - True Sense Reversing Barrier
                atomicTreeBarrUniqNaiveSRB - True Sense Reversing Barrier (naive)
                atomicTreeBarrUniqNaiveAllSRB - True Sense Reversing Barrier (naive all)
                atomicTreeBarrUniqHybridSRB - Hybrid Sense Reversing Barrier
                atomicTreeBarrUniqCCG - NVIDIA's CUDA Cooperative Groups Barrier
                atomicTreeBarrUniqLocalExch - Atomic Tree Barrier with local exchange
                atomicTreeBarrUniqLocalExchSRB - True Sense Reversing Barrier with local exchange
                atomicTreeBarrUniqLocalExchNaiveSRB - True Sense Reversing Barrier with local exchange (naive)
                atomicTreeBarrUniqLocalExchNaiveAllSRB - True Sense Reversing Barrier with local exchange (all threads join naively)
                atomicTreeBarrUniqLocalExchHybridSRB - Hybrid CCG-Sense Reversing Barrier with local exchange
                atomicTreeBarrUniqLocalExchCCG - NVIDIA's CCG Barrier with local exchange
                lfTreeBarrUniq - Lock-Free Tree Barrier
                lfTreeBarrUniqLocalExch - Lock-Free Tree Barrier with local exchange
                spinMutex - Spin Lock Mutex
                spinMutexEBO - Spin Lock Mutex with Backoff
                sleepMutex - Sleep Mutex
                faMutex - Fetch-and-Add Mutex
                spinSem1 - Spin Semaphore (Max: 1)
                spinSem2 - Spin Semaphore (Max: 2)
                spinSem10 - Spin Semaphore (Max: 10)
                spinSem120 - Spin Semaphore (Max: 120)
                spinSemEBO1 - Spin Semaphore with Backoff (Max: 1)
                spinSemEBO2 - Spin Semaphore with Backoff (Max: 2)
                spinSemEBO10 - Spin Semaphore with Backoff (Max: 10)
                spinSemEBO120 - Spin Semaphore with Backoff (Max: 120)
                PRspinSem1 - Priority Spin Semaphore (Max: 1)
                PRspinSem2 - Priority Spin Semaphore (Max: 2)
                PRspinSem10 - Priority Spin Semaphore (Max: 10)
                PRspinSem120 - Priority Spin Semaphore (Max: 120)
                PRspinSemEBO1 - Priority Spin Semaphore with Backoff (Max: 1)
                PRspinSemEBO2 - Priority Spin Semaphore with Backoff (Max: 2)
                PRspinSemEBO10 - Priority  Spin Semaphore with Backoff (Max: 10)
                PRspinSemEBO120 - Priority Spin Semaphore with Backoff (Max: 120)
                spinMutexUniq - Spin Lock Mutex -- accesses to unique locations per TB
                spinMutexEBOUniq - Spin Lock Mutex with Backoff -- accesses to unique locations per TB
                sleepMutexUniq - Sleep Mutex -- accesses to unique locations per TB
                faMutexUniq - Fetch-and-Add Mutex -- accesses to unique locations per TB
                spinSemUniq1 - Spin Semaphore (Max: 1) -- accesses to unique locations per TB
                spinSemUniq2 - Spin Semaphore (Max: 2) -- accesses to unique locations per TB
                spinSemUniq10 - Spin Semaphore (Max: 10) -- accesses to unique locations per TB
                spinSemUniq120 - Spin Semaphore (Max: 120) -- accesses to unique locations per TB
                spinSemEBOUniq1 - Spin Semaphore with Backoff (Max: 1) -- accesses to unique locations per TB
                spinSemEBOUniq2 - Spin Semaphore with Backoff (Max: 2) -- accesses to unique locations per TB
                spinSemEBOUniq10 - Spin Semaphore with Backoff (Max: 10) -- accesses to unique locations per TB
                spinSemEBOUniq120 - Spin Semaphore with Backoff (Max: 120) -- accesses to unique locations per TB
        <numLdSt>: the # of LDs and STs to do for each thread in the critical section.
        <numTBs>: the # of TBs to execute (want to be divisible by the number of SMs).
        <numCSIters>: number of iterations of the critical section.

The microbenchmarks used in the paper include: 
Barrier: atomicTreeBarrUniq(Heterosync Baseline), atomicTreeBarrUniqSRB(G-SRB), atomicTreeBarrUniqNaiveSRB(G-CPUSRB), atomicTreeBarrUniqCCG(CCG) and atomicTreeBarrUniqHybridSRB(Hybrid barrier)

All numbers in the paper were genereted with 10 as the number of iterations of the critical section. 
Usage e.g for the barrier (Assuming 80 SMs)-> ./allSyncPrims-1kernel atomicTreeBarrUniqSRB 100 80 10

Semaphore: spinSem1, spinSem10 , spinSem120, spinSemEBO10, spinSemEBO120, spinSemEBO1, PRspinSem1, PRspinSem10 , PRspinSem120, PRspinSemEBO10, PRspinSemEBO120, PRspinSemEBO1

The number at the end of the name of the semaphore benchmark gives you the semaphore size.
All numbers in the paper were genereted with 10 as the number of iterations of the critical section

Usage e.g for the semaphore ./allSyncPrims-1kernel PRspinSem1 100 80 10


CUDA VERSION: 11.2


