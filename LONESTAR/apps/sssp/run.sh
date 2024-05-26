#!/bin/bash

# Run GPU SRB (sense-reversing barrier) variant
../../bin/sssp_gsrb -s 0 -o outfile.txt ../../inputs/USA-road-d.W.gr
# Run GPU SRB variant that uses CPU-style SRBs
../../bin/sssp_gcpusrb -s 0 -o outfile.txt ../../inputs/USA-road-d.W.gr
# Run CUDA Cooperative Graphs (CCG) variant
../../bin/sssp_ccg -s 0 -o outfile.txt ../../inputs/USA-road-d.W.gr
