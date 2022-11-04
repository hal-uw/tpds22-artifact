# tpds22-artifact
The benchmarks and artifacts associated with our TPDS '22 paper on GPU synchronization primitives

The repository is organized as follows
HEAD
Lonestar (BFS, SSSP and PR)
Reduce
Heterosync
Scan

The respective benchmark folders contain their own README’s with instructions on how to compile and run them, they also contain information about what algorithm they implement and the version number of the benchmark. More specifically, go to the following locations to compile and run:

BFS: $HEAD/LONESTAR/apps/bfs/
PR: $HEAD/LONESTAR/apps/pr
SSSP: $HEAD/LONESTAR/apps/sssp
Reduce: $HEAD/Reduce
Scan: $HEAD/scan

Requested Info:
- GCC version: 9.3
- NVCC version: 11.2 (note: this is updated in revised version, was 11.1 previously)
- OS: Ubuntu 20 (note: this is updated in revised version, was Ubuntu 16.04 previously)
- Dependencies: No dependency except CUDA and its associated libraries required (for Lonestar we have included a local version of the dependencies to avoid the need to get this working on the reviewer’s end)
- Hardware: same GPUs as listed in paper (Titan V, GTX 2080Ti)
- Inputs/Datasets: The microbenchmarks, Reduce, and Scan have no datasets needed. For the - Lonestar benchmarks (BFS, PR, SSSP), please run “make inputs” in $HEAD/LONESTAR/ (Note: this will take a ~10 minutes).
- How to run: Please refer to appropriate READMEs in GitHub
- Scripts: Each application’s folder contains scripts to compile the program (compile.sh), except for HeteroSync. For HeteroSync simply run “make” in $HEAD/Heterosync.
- Time to run: each experiment should take < 5 minutes, in total < 1 hour
- Components: see READMEs
- For HIP versions, we tested the benchmarks with HIP v4.1.  Newer versions may work but could require additional tweaks.

Step-by-step:
Benchmarks:
0. Make sure whatever directory nvcc is in, is in your PATH in your bashrc/cshrc
1. Run compile.sh in the appropriate benchmark folder (see above)
2. See README for each application for how to run
3. (For BFS, PR, SSSP only) run “make inputs” in $HEAD/LONESTAR/ before running application
4. Run application

Microbenchmarks:
0. Make sure whatever directory nvcc is in, is in your PATH in your bashrc/cshrc
1. Run “make”
2. See README for how to run each microbenchmark
3. Run microbenchmark

CITATION
--------

If you publish work that uses these benchmarks, please cite the following papers:

0.  P. Dalmia, R. Mahapatra, J. Intan, D. Negrut, and M. D. Sinclair.  Improving the Scalability of GPU Synchronization Primitives, in IEEE Transactions on Parallell and Distributed Computing (TPDS), 2022.

Depending on how you use the benchmarks, you may also consider citing:

1.  M. D. Sinclair, J. Alsop, and S. V. Adve, HeteroSync: A Benchmark Suite for Fine-Grained Synchronization on Tightly Coupled GPUs, in the IEEE International Symposium on Workload Characterization (IISWC), October 2017

2.  J. A. Stuart and J. D. Owens, “Efficient Synchronization Primitives for GPUs,” CoRR, vol. abs/1110.4623, 2011
