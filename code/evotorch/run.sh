#!/bin/bash
# Date: 2023-04-24
# Author: Jesse Wood
#
# The task spooler (ts) command allows for parallel execution of python scripts.
 
DATASET="part"; 
# DATASET="species";
# Directory to save results to.
NAME="tmp";

# If the directory does not already exsit.
if [ ! -d "logs/${DATASET}/${NAME}" ];
then
    # Make a directory for the output logs.
    mkdir "logs/${DATASET}/${NAME}";
fi 

# The server has 3 GPUs to use in parallel.
ts -S 3

for i in {1..10}; 
do 
    # Run the experiments using the ts command.
    ts -G 1 python3 main.py \
	    --dataset "${DATASET}" \
	    --file-path "checkpoints/run_${i}.pth" \
	    --output "logs/${DATASET}/${NAME}/run" --run $i \
	    --generations 50 --beta 3;
done
