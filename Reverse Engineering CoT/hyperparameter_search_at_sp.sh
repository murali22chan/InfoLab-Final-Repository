#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 #Running on GTX 1080

# Define list of parameters
AH=(4 6 8 10)
SL=(5 7 10)



for sl in "${SL[@]}"
do
  for ah in "${AH[@]}"
  do
    python run_cross_domain.py --attentionHeads $ah --noOfSplits $sl

  done
done



