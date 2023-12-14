#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 #Running on GTX 1080
# Define list of parameters
DOMAINS=("finance" "medicine" "open_qa" "wiki_csai" "reddit_eli5")

# Loop over training data domains
for train_domain in "${DOMAINS[@]}"
do
  # Loop over the testing data domains
  for test_domain in "${DOMAINS[@]}"
  do
    # Run the Python script with specified parameters
    python run_roberta.py --trainDomain $train_domain --testDomain $test_domain
  done
done