# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import random_split
import torchvision
from torchvision import transforms

# python imports
import os
import numpy as np

# additional libraries
import pytorch_lightning as pl

# custom imports
from data_management.data_tools import DefaultDataset

    

class FashionMNISTDataModule(pl.LightningDataModule):
    """PyTorch-Lightning datamodule class for the FashionMNIST dataset."""
    def __init__(self, data_dir, batch_size, Nsamples=None, valid_frac=None, 
                 random_state=42, transform=None, test_transform=None ):
        super().__init__()
        self.data_dir   = data_dir      # directory to contain this and other eventual datasets
        self.batch_size = batch_size    # training batch size
        self.Nsamples   = Nsamples      # number of samples to load from the train dataset
        self.frac       = valid_frac    # fraction of samples to use as validation set
        self.seed       = random_state
        
        # set up transformations
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        
        if test_transform is None:
            self.test_transform = transforms.ToTensor()
        else:
            self.test_transform = test_transform
            
        ### label names
        self.label_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat'      ,
                            'Sandal'     ,'Shirt'  ,'Sneaker' ,'Bag'  ,'Ankle boot']
        
    def prepare_data(self):
        ### train dataset
        self.full = torchvision.datasets.FashionMNIST(self.data_dir, 
                                                      train    = True, 
                                                      download = True, ##
                                                     )
        ### test dataset
        self.test_data = torchvision.datasets.FashionMNIST(self.data_dir,
                                                           train     = False,
                                                           download  = True, ##
                                                           transform = self.test_transform,
                                                          )

    def setup(self, stage=None):
        # eventually restricting to Nsamples 
        if self.Nsamples is not None:
            self.full = Subset(self.full, np.random.choice(range(self.full.__len__()), size=self.Nsamples))
            
        # split into train and validation 
        split_tr  = round( self.full.__len__()*(1-self.frac) )
        split_val = round( self.full.__len__()*self.frac )
        
        train, val = random_split(self.full,
                                  [split_tr, split_val], 
                                  generator=torch.Generator().manual_seed(self.seed),
                                 )
        # add transforms
        self.train_data = DefaultDataset(data=train, transform=self.transform     )
        self.val_data   = DefaultDataset(data=val  , transform=self.test_transform)  

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size, 
                          shuffle    = True, 
                          pin_memory = True,
                         )

    def val_dataloader(self):
        return DataLoader(self.val_data, 
                          batch_size = self.batch_size, 
                          shuffle    = False, 
                          pin_memory = True,
                         ) 
    
    def test_dataloader(self):
        return DataLoader(self.test_data, 
                          batch_size = 500, #self.batch_size, 
                          shuffle    = False, 
                          pin_memory = True,
                         ) 
    
    def predict_dataloader(self):
        return DataLoader(self.test_data, 
                          batch_size = 500, #self.batch_size, 
                          shuffle    = False, 
                          pin_memory = True,
                         ) 
    
    def get_label_names(self):
        return self.label_names
    
    def get_sample_size(self):
        return torch.Size([1,28,28])

    
