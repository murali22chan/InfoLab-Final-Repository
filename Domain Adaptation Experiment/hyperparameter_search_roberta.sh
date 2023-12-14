#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 #Running on GTX 1080

# Define list of parameters
LR=(0.00001 0.00005 0.000001 0.000005)
Epochs=(3 5 10)


#Loop over Alpha values
for lr in "${LR[@]}"
do
    # Loop over Beta values
    for epochs in "${Epochs[@]}"
    do
        python run_roberta.py --epochs $epochs --lr $lr
    done
done