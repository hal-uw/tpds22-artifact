# tpds22-artifact

The benchmarks and artifacts associated with our TPDS '22 paper on GPU synchronization primitives

## Artifact Identification

**Title**: Improving the Scalability of GPU Synchronization Primitives

**Authors**: Preyesh Dalmia, Rohan Mahapatra, Jeremy Intan, Dan Negrut, Matthew D. Sinclair

**Abstract**: General-purpose GPU applications increasingly use synchronization to enforce ordering between many threads accessing shared data. Accordingly, recently there has been a push to establish a common set of GPU synchronization primitives. However, the expressiveness of existing GPU synchronization primitives is limited. In particular the expensive GPU atomics often used to implement fine-grained synchronization make it challenging to implement efficient algorithms. Consequently, as GPU algorithms scale to millions or billions of threads, existing GPU synchronization primitives either scale poorly or suffer from livelock or deadlock issues because of heavy contention between threads accessing shared synchronization objects. We seek to overcome these inefficiencies by designing more efficient, scalable GPU barriers and semaphores. In particular, we show how multi-level sens e reversing barriers and priority mechanisms for semaphores can be designed with the GPUs unique processing model in mind to improve performance and scalability of GPU synchronization primitives. Our results show that the proposed designs significantly improve performance compared to state-of-the-art solutions like CUDA Cooperative Groups and optimized CPU-style synchronization algorithms at medium and high contention levels, scale to an order of magnitude more threads, and avoid livelock in these situations unlike prior open source algorithms. Overall, across three modern GPUs the proposed barrier algorithm improves performance by an average of 33% over a GPU tree barrier algorithm and improves performance by an average of 34% over CUDA Cooperative Groups for five full-sized benchmarks at high contention levels; the new semaphore algorithm improves performance by an average of 83% compared to prior GPU semaphores.

## Artifact Dependencies and Requirements

**Hardware resources required**: A system with a CPU (e.g., x86 CPU) and NVIDIA GPU (e.g., NVIDIA Titan V)

**Operating systems required**: GNU/Linux

**Software libraries needed**: CUDA, g++, gcc

Specifically, we tested with the following configuration:

- GCC version: 9.3
- NVCC version: 11.2 (note: this is updated in revised version, was 11.1 previously)
- OS: Ubuntu 20 (note: this is updated in revised version, was Ubuntu 16.04 previously)
- Dependencies: No dependency except CUDA and its associated libraries required (for Lonestar we have included a local version of the dependencies to avoid the need to get this working on the reviewer’s end)
- Hardware: same GPUs as listed in paper (Titan V, GTX 2080Ti)
- For HIP versions, we tested the benchmarks with HIP v4.1.  Newer versions may work but could require additional tweaks.

**Input dataset(s) needed**: For the microbenchmarks, Reduce, and Scan no external datasets are needed, this repo contains what is required to reproduce results.  For the LoneStarGPU (BFS, PR, SSSP) benchmarks, the dataset is downloaded from the original repo.  To get this dataset, you will need to run `make inputs` in $HEAD/LONESTAR, which will take ~10 minutes.

## Artifact Installation and Deployment Process

### How to install and compile the libraries and the code

### Use git to clone the repository

$ git clone git@github.com:https://github.com/hal-uw/tpds22-artifact

Cloning the repo should not take more than a minute.

The repository is organized as follows
HEAD
Lonestar (BFS, SSSP and PR)
Reduce
Heterosync
Scan

The respective benchmark folders contain their own README’s with instructions on how to compile and run them, they also contain information about what algorithm they implement and the version number of the benchmark. More specifically, go to the following locations to compile and run:

**BFS**: $HEAD/LONESTAR/apps/bfs/
**PR**: $HEAD/LONESTAR/apps/pr
**SSSP**: $HEAD/LONESTAR/apps/sssp
**Reduce**: $HEAD/Reduce
**Scan**: $HEAD/scan

### Compilation

For all benchmarks to compile properly, you will need to update your PATH in your bashrc/cshrc to include whatever directory nvcc is in (e.g., export PATH=/usr/local/cuda/bin:$PATH if /usr/local/cuda/bin has nvcc in it).
Each benchmark in this repo has a README file with it that provides additional details on running different variants of the benchmark (e.g., how to run the CCG, GSRB, and GCPUSRB variants).
With the exception of HeteroSync, all benchmarks also contain a compile.sh script that can be used to compile the corresponding benchmark -- for HeteroSync simply run `make` in $HEAD/Heterosync.
Moreover, for LONESTAR, $HEAD/LONESTAR/run.sh will compile and run all LONESTAR benchmarks.

### Time to run:

Each experiment should take < 5 minutes, in total < 1 hour

Step-by-step:
Benchmarks:
0. Make sure whatever directory nvcc is in, is in your `PATH` in your bashrc/cshrc
1. Run `compile.sh` in the appropriate benchmark folder (see above)
2. See *README* for each application for how to run
3. (For BFS, PR, SSSP only) run `make inputs` in `$HEAD/LONESTAR/` before running application
4. Run application

Microbenchmarks:
0. Make sure whatever directory nvcc is in, is in your PATH in your bashrc/cshrc
1. Run `make`
2. See *README* for how to run each microbenchmark
3. Run microbenchmark

CITATION
--------

If you publish work that uses these benchmarks, please cite the following papers:

0.  P. Dalmia, R. Mahapatra, J. Intan, D. Negrut, and M. D. Sinclair.  Improving the Scalability of GPU Synchronization Primitives, in IEEE Transactions on Parallel and Distributed Computing (TPDS), 2022.

Depending on how you use the benchmarks, you may also consider citing:

1.  M. D. Sinclair, J. Alsop, and S. V. Adve, HeteroSync: A Benchmark Suite for Fine-Grained Synchronization on Tightly Coupled GPUs, in the IEEE International Symposium on Workload Characterization (IISWC), October 2017

2.  J. A. Stuart and J. D. Owens, “Efficient Synchronization Primitives for GPUs,” CoRR, vol. abs/1110.4623, 2011
