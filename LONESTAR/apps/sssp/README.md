## DESCRIPTION

This benchmark computes the shortest path from a source node to all nodes in a directed graph with non-negative edge weights by using a modified near-far algorithm [1].

[1] https://people.csail.mit.edu/jshun/6886-s18/papers/DBGO14.pdf


## BUILD

Run ./compile.sh to create binaries for SSSP benchmark using G-SRB (sssp_gsrb), Nvidia's CCG (sssp_ccg) and G-CPUSRB (sssp_gcpusrb)

## RUN

Execute as: ./sssp [-o output-file] [-l] [-s startNode] graph-file 


The option -l  enables thread block load balancer. Enable this option for power-law graphs to improve the performance. It is recommended to disable this option for high diameter graphs, such as road-networks. 

e.g., ./sssp_gsrb -s 0 -o outfile.txt ../../inputs/USA-road-d.W.gr

After the application runs it will ask you for a block factor, this basically corresponds to NumTBs/SM for the volta can enter any value between 1 and 16

Graphs Corresponding to these block factors are
1 - bgg.gr
2 - USA-road-d.NY.gr
4 - USA-road-d.FLA.gr
8 - USA-road-d.W.gr
16 - USA-road-d.USA.gr

The run.sh script contains an example of how to run all 3 variants back-to-back-to-back for the USA-road-d.W.gr input graph (scale factor 8).