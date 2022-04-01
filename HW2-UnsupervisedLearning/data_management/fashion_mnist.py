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


## DANGER cose commentate sotto da sistemare DANGER


class DatasetFromSubset(Dataset):
    """Class that build a proper "Dataset" object from a "Subset"."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


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
        self.mnist_test = torchvision.datasets.FashionMNIST(self.data_dir,
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
        self.mnist_train = DatasetFromSubset(subset=train, transform=self.transform     )
        self.mnist_val   = DatasetFromSubset(subset=val  , transform=self.test_transform)  

    def train_dataloader(self):
        return DataLoader(self.mnist_train, 
                          batch_size = self.batch_size, 
                          shuffle    = True, 
                          pin_memory = True,
                         )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, 
                          batch_size = self.batch_size, 
                          shuffle    = False, 
                          pin_memory = True,
                         ) 
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, 
                          batch_size = self.batch_size, 
                          shuffle    = False, 
                          pin_memory = True,
                         ) 
    
    def get_label_names(self):
        return self.label_names
    
    def get_sample_size(self):
        return torch.Size([1,28,28])
    
    
    
    
### custom transformations ----------------------------------------------------------------------------------

class AddGaussianNoise():
    """ Transform that add a gaussian noise with given mean and std with a certain probability.
            prob : occurring probability of the transformation
            mean : mean of the gaussian noise
            std  : std of the gaussian noise
    """
    def __init__(self, prob=0.5, mean=0., std=1.):
        self.prob = prob
        self.mean = mean
        self.std  = std
        
    def __call__(self, tensor):
        if torch.rand(1) < self.prob:
            # generating and adding noise
            tensor += torch.randn(tensor.size())*self.std + self.mean
            
            # returning pixels values in range [0,1]
            min_val = torch.min(tensor)
            tensor -= min_val
            max_val = torch.max(tensor)
            tensor /= max_val
            return tensor
        else:
            return tensor
    
    
    
    
## transform that add a rectangular occlusion to image with a certain probability
#class AddOcclusion():  # BUG BUG FORSE CE N'È UNA DENTRO A torchvision #-> RandomErasing() is better
    #def __init__(self, max_area=0.5, prob=0.5):
        #"""
        #max_area : maximum fraction of image area that is allowed to be covered by occlusion
        #prob     : occurring probability of the transformation
        #"""
        #self.max_area = max_area
        #self.prob     = prob
        
    #def __call__(self, tensor):
        #if torch.rand(1) < self.prob:
            ## taking random box vertices
            #xs = np.rint( np.random.rand(2)*tensor.size()[1] )
            #ys = np.rint( np.random.rand(2)*tensor.size()[2] ) 
            
            ## ordering the vertices
            #xs = np.sort(xs.astype(int))
            #ys = np.sort(ys.astype(int))
            
            ## checking if occluded area is greater than max_area
            #max_area_pxs = tensor.size()[1]*tensor.size()[2] * self.max_area
            #if (xs[1]-xs[0])*(ys[1]-ys[0]) > max_area_pxs:
                #xs[1] = xs[1]//2
                #ys[1] = ys[1]//2
            
            #tensor[:,xs[0]:xs[1],ys[0]:ys[1]] = 0.
            #return tensor
        #else:
            #return tensor


#def test_transform(img, transform):   ## DA SISTEMARE forse non serve a niente
    #to_tensor = transforms.ToTensor()
    #img = to_tensor(train_dataset[0][0])

    ## add gaussian noise
    #noiser = AddGaussianNoise(0., 0.1, prob=1.)
    #img_noisy = noiser(img.detach().clone())

    ## add occlusion
    #occluder = AddOcclusion(max_area=0.5, prob=1.)
    #img_occluded = occluder(img.detach().clone())

    ## add horizontal flipping
    #flipper = transforms.RandomVerticalFlip(p=1.)
    #img_flipped = flipper(img.detach().clone())



    
    
