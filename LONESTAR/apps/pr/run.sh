#!/bin/bash

# Run GPU SRB (sense-reversing barrier) variant
../../bin/pr_gsrb -b 4 -o outfile.txt -x 1000000 ../../inputs/USA-road-d.W.gr
# Run GPU SRB variant that uses CPU-style SRBs
../../bin/pr_gcpusrb -b 4 -o outfile.txt -x 1000000 ../../inputs/USA-road-d.W.gr
# Run CUDA Cooperative Graphs (CCG) variant
../../bin/pr_ccg -b 4 -o outfile.txt -x 1000000 ../../inputs/USA-road-d.W.gr
