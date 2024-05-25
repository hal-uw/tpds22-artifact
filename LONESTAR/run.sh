#!/bin/bash

# download dataset
make inputs

for appl in bfs pr sssp
do
    echo "Running $appl"
    cd apps/$appl
    ./compile.sh
    ./run.sh
    cd -
done
