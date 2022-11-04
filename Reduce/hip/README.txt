Compilation:

Run ./compile.sh to create binaries for Reduce benchmark using G-SRB(reduce_gsrb) , Nvidia's CCG(reduce_ccg) and G-CPUSRB (reduce_gcpusrb)

Parameters:

To run any benchmarks you have to pass two arguments ./reduce_gsrb ARRAY_SIZE WORKGROUP_SIZE
To Create different levels of contention, adjust array size according to the underlying architecture for e.g the TITANV has 80 SMs so 80 thread blocks with block size of 32 requires an array of size 80*32 = 2560

USAGE:
Note: these examples are assuming 80 SMs
e.g 
./reduce_gsrb 2560  32 ( min contention)
./reduce_ccg 81920 32 (max contention)
