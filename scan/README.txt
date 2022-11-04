Version:

- Uses **Blelloch's Algorithm** (exclusive scan)
- Not limited by 2048 items (a former restriction on the initial implementation of the algorithm due to the maximum threads that can run in a thread block on current GPUs)
- Not limited by input sizes that are powers of 2 (a former restriction due to inherent binary tree-approach of the algorithm)
- Free of shared memory bank conflicts using the index padding method in this [paper](https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf).


Compilation:

Run ./compile.sh to create binaries for Scan benchmark using G-SRB(scan_gsrb), Nvidia's CCG(scan_ccg) and G-CPUSRB (scan_gcpusrb)

Parameters:

To run any benchmarks you have to pass one argument ./scan_gsrb ARRAY_SIZE 
To Create different levels of contention, adjust array size according to the underlying architecture for e.g the TITANV has 80 SMs so 1 thread block on each SM  requires an array of size 80*64 = 5120, 32 thread blocks on each SM will require an array size of 80*32*64 = 163840

USAGE:
Note: these examples are assuming 80 SMs
e.g 
./scan_gsrb 5120 ( min contention)
./scan_ccg 163840  (max contention)
