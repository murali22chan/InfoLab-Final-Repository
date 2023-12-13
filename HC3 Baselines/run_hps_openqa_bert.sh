#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Define list of parameters
LR=(0.00001 0.000001 0.000002 0.000005)
Epochs=(5 10 15 20)


#Loop over Alpha values
for lr in "${LR[@]}"
do
    # Loop over Beta values
    for epochs in "${Epochs[@]}"
    do
        python run_open_qa.py --epochs $epochs --lr $lr
    done
done