## DESCRIPTION

This benchmark computes the level of each node from a source node in an unweighted graph. It starts at node 0 and explores all the nodes on the same level and move on to nodes at the next depth level. 

## BUILD

Run ./compile.sh to create binaries for BFS benchmark using G-SRB(bfs_gsrb), Nvidia's CCG(bfs_ccg) and G-CPUSRB (bfs_gcpusrb)

## RUN

Execute as: ./bfs [-o output-file] [-l] [-s startNode] graph-file[appropriately chosen according to block factor(defined below)


The option -l  enables thread block load balancer. Enable this option for power-law graphs to improve the performance. It is recommneded to disable this option for high diameter graphs, such as road-networks. 

e.g., ./bfs_gsrb -s 0 -o outfile.txt ../../inputs/USA-road-d.W.gr

After the application runs it wil ask you for a block factor, this basically corresponds to NumTBs/SM for the volta can enter any value between 1 and 16

Graphs Corresponding to these block factors are
1 - bgg.gr
2 - USA-road-d.NY.gr  
4 - USA-road-d.FLA.gr
8 - USA-road-d.W.gr
16 - USA-road-d.USA.gr  
