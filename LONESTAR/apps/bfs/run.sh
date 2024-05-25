#!/bin/bash

# Run GPU SRB (sense-reversing barrier) variant
./bin/bfs_gsrb -s 0 -o outfile.txt ../../inputs/USA-road-d.W.gr
# Run GPU SRB variant that uses CPU-style SRBs
./bin/bfs_gcpusrb -s 0 -o outfile.txt ../../inputs/USA-road-d.W.gr
# Run CUDA Cooperative Graphs (CCG) variant
./bin/bfs_ccg -s 0 -o outfile.txt ../../inputs/USA-road-d.W.gr
