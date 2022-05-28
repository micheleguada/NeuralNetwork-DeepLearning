# PyTorch imports
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

# python imports
import os
import numpy as np

# additional libraries
import pytorch_lightning as pl


### data utilities ------------------------------------------------------------------------------------------

class DefaultDataset(Dataset):
    """Class that build a proper "Dataset" object."""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.data)



### custom transformations ----------------------------------------------------------------------------------

class AddGaussianNoise():
    """ Transform that add a gaussian noise with given mean and std with a certain probability.
            p    : occurring probability of the transformation
            mean : mean of the gaussian noise
            std  : std of the gaussian noise
    """
    def __init__(self, p=0.5, mean=0., std=1.):
        self.prob = p
        self.mean = mean
        self.std  = std
        
    def __call__(self, tensor):
        if torch.rand(1) < self.prob:
            tensor = tensor.clone()
            
            # generating and adding noise
            tensor += torch.randn(tensor.size())*self.std + self.mean
            
            # returning pixels values in range [0,1]
            tensor = torch.clip(tensor, min=0., max=1.)

            return tensor
        else:
            return tensor
        
        
