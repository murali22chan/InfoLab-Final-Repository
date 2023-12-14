# InfoLab-Final-Repository

## DetectGPTHC3
This folder contains the modified code to run detect GPT on HC3 dataset. Load the HC3 dataset to the "data" folder from the drive and directly run the Python file. The result analysis of the experiment is in the drive link.

## Domain Adaptation (Unsupervised DA through backpropagation)
This file contains the code to run Unsupervised DA through backpropagation on the HC3 dataset. Place the dataset from the drive to this folder. User "run_experiment" to run the experiment and "hyperparameters" to run hyperparameter search for the experiment.

## HC3 Baselines
This file contains the code to run supervised fine-tuning of BERT and RoBERTa on HC3 dataset in cross-domain setting. Load the HC3 dataset files to the "/data" folder. Use "run_cross_domain" and "run_cross_domain_roberta" to run the experiment. And use "run_hps_domain_bert" for domain-wise hyperparameter search. 

## Hypergraph and LLM
This file contains the code to run the Hypergraph + LLM method on the HC3 dataset. Load the HC3 dataset files to the "/data" folder. Use "run_experiment" to run the experiment.

## Reverse Engineering CoT
This file contains the code to run GAT + BERT method. Load the HC3 dataset files to the "/data" folder. Use "hyperparameter_search" to perform hyperparater seach for LR and Epochs. Use "hyperparameter_search_at_sp" to run hyperparameter search for the number of splits and the number of attention heads. 

Load Data.pkl file directly here from drive to reduce the run time by utilizing the saved BERT embeddings as a graph.

## Domain Adaptation Experiment
The file contains initial experiments with BERT and Roberta. It includes code that used to split HC3 data, initial analysis, code to test HC3 dataset on low resource and combination settings.
