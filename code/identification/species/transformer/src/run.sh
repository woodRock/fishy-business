#!/bin/bash

for i in {1..10}; 
do 
    python3 main.py -d "part" -o "logs/part/data_augmentation" -r $i -da; 
done
