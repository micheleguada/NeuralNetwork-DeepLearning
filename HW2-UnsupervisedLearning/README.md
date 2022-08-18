# Neural Networks and Deep Learning 

## Homework 2 - Unsupervised Learning

Solution of the second homework of the course: NEURAL NETWORKS AND DEEP LEARNING 2021/2022 

The code is structured into three folders:  
> 1. **autoencoder**, which contains the files where the models classes are implemented using `PyTorch` and `PyTorch-Lightning` libraries.  
> 2. **data_management**, where data tools and the `datamodule` class for the `PyTorch-Lightning` framework are defined.  
> 3. **utilities**, which contains some tools useful to optimize the hyper-parameters (using `Optuna` library), to train the model and also to plot and analyze results.  

The main script is implemented inside a jupyter-notebook (`Homework2-main.ipynb`), located in the main folder together with a *settings* file, `settings.py`. 
This file contains the global seed number, GPU availability check and the path settings for all the output files.
