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

    

class CelebADataModule(pl.LightningDataModule):
    """PyTorch-Lightning datamodule class for the CelebA dataset."""
    def __init__(self, data_dir, batch_size, Ntrain=None, Nvalid=None, Ntest=None,
                 random_state=42, transform=None, test_transform=None ):
        super().__init__()
        self.data_dir   = data_dir      # directory to contain this and other eventual datasets
        self.batch_size = batch_size    # training batch size
        self.Ntrain     = Ntrain        # number of samples to load from the train dataset
        self.Nvalid     = Nvalid        # number of samples to use of the validation set
        self.Ntest      = Ntest         # number of samples to use of the test set
        #self.seed       = random_state  # DANGER
        
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
        self.label_names = []    #DANGER
        
        
    def prepare_data(self):
        ### train dataset
        self.train_data = torchvision.datasets.CelebA(self.data_dir, 
                                                      split    = "train", 
                                                      download = True, ##
                                                      transform = self.transform
                                                     )
        ### valid dataset
        self.valid_data = torchvision.datasets.CelebA(self.data_dir,
                                                      split     = "valid",
                                                      download  = True, ##
                                                      transform = self.test_transform,
                                                     )
        ### test dataset
        self.test_data = torchvision.datasets.CelebA(self.data_dir,
                                                     split     = "test",
                                                     download  = True, ##
                                                     transform = self.test_transform,
                                                    )

    def setup(self, stage=None):
        # eventually restricting to less samples 
        if self.Ntrain is not None:
            self.train_data = Subset(self.train_data, np.random.choice(range(self.train_data.__len__()), size=self.Ntrain))
            
        if self.Ntrain is not None:
            self.train_data = Subset(self.train_data, np.random.choice(range(self.train_data.__len__()), size=self.Ntrain))
            
        if self.Ntrain is not None:
            self.train_data = Subset(self.train_data, np.random.choice(range(self.train_data.__len__()), size=self.Ntrain))

        ## add transforms   # DANGER
        #self.train_data = DefaultDataset(data=train, transform=self.transform     )
        #self.val_data   = DefaultDataset(data=val  , transform=self.test_transform)  

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
    
    #def get_sample_size(self):   # DANGER
        #return torch.Size([1,28,28])

    
 
