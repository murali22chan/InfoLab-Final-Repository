#!/bin/bash
#export CUDA_VISIBLE_DEVICES=1 #Running on GTX 1080

# Define list of parameters
LR=(0.01 0.1 0.001)
Epochs=(25 50 75 100)


#Loop over Alpha values
for lr in "${LR[@]}"
do
    # Loop over Beta values
    for epochs in "${Epochs[@]}"
    do
        python run_with_load_data.py --epochs $epochs --lr $lr
    done
done


