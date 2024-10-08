#!/bin/bash
# Date: 2023-04-24
# Author: Jesse Wood
#
# The task spooler (ts) command allows for parallel execution of python scripts.
 

# The server has 3 GPUs to use in parallel.
ts -S 3

for i in {1..10}; 
do 
    # Run the experiments using the ts command.
    ts -G 1 python3 vae.py
done
