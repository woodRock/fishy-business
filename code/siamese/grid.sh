#!/bin/bash
# Date: 2023-04-24
# Author: Jesse Wood
# 
# This script is for submission on the ECS grid computing system.
#
# Mail me at the b(eginning) and e(nd) of the job
#
#$ -M jesse.wood@ecs.vuw.ac.nz
#$ -m be
#$ -wd /vol/grid-solar/sgeusers/woodjess3 
#
# Help: https://ecs.wgtn.ac.nz/Support/TechNoteEcsGrid#A_basic_job_submission_script

# Ensure the grid commands work.
need sgegrid

# Change directory to desktop fishy-business
cd /vol/ecrg-solar/woodj4/fishy-business/code/siamese

DATASET="instance-recognition"; 
# DATASET="species";
# Directory to save results to.
NAME="tmp_3";

# If the directory does not already exsit.
if [ ! -d "logs/${DATASET}/${NAME}" ];
then
    # Make a directory for the output logs.
    mkdir "logs/${DATASET}/${NAME}";
fi 

# $SGE_TASK_ID, is provided so as to differentiate between the tasks. 
i=$SGE_TASK_ID

# Run the experiments using the ts command.
python3 gp.py \
    --dataset "${DATASET}" \
    --file-path "checkpoints/run_${i}.pth" \
    --output "logs/${DATASET}/${NAME}/run" --run "${i}" \
    --generations 50 --population 200 --num-trees 50;
