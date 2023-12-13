#!/bin/bash
#export CUDA_VISIBLE_DEVICES=1 #Running on GTX 1080

# Define list of parameters
Alphas=(0.5 0.005 0.005)
Betas=(0.5 0.05 0.005)


#Loop over Alpha values
for alpha in "${Alphas[@]}"
do
    # Loop over Beta values
    for beta in "${Betas[@]}"
    do
        python main_v2.py --alpha $alpha --beta $beta
    done
done
