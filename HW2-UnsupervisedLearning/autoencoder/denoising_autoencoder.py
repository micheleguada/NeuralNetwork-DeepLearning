# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# python imports
import numpy as np

# additional libraries
import pytorch_lightning as pl

# custom modules 
from autoencoder.symmetric_autoencoder import SymmetricAutoencoder, SymmetricAutoencoderHPS
from autoencoder.components import Encoder, Decoder, BaseHPS, BasePLModule
from autoencoder.components import get_activation, conv_output_shape, conv_transpose_output_shape

from data_management.data_tools import AddGaussianNoise  # custom



### Denoising Autoencoder class with symmetric Encoder/Decoder pair -----------------------------------------
class DenoisingAutoencoder(SymmetricAutoencoder):
    """
    Denoising Autoencoder with symmetric Encoder/Decoder pair.
    """    
    def __init__(self, 
                 input_size    : tuple, 
                 params        : dict   = None,
                 optimizer     : str    = "adam",
                 learning_rate : float  = 0.001,
                 L2_penalty    : float  = 0.,
                 momentum      : float  = None,   #ignored if optimizer is optim.Adam
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 corruption    : object = None,   #composed transformation to be used to corrupt train images
                 corruption_p  : float  = 0.5,    #probability of single transformations to be applied (used if "corruption" is None)
                ):
        super(DenoisingAutoencoder, self).__init__(input_size    = input_size   , 
                                                   params        = params       , 
                                                   optimizer     = optimizer    , 
                                                   learning_rate = learning_rate, 
                                                   L2_penalty    = L2_penalty   , 
                                                   momentum      = momentum     , 
                                                   encoder_class = encoder_class,
                                                   decoder_class = decoder_class,
                                                  ) 
        self.save_hyperparameters() # save hyperparameters when checkpointing
        
        # set corruption transformation
        self.configure_corruption(corruption=corruption, prob=corruption_p)
        
        
    def training_step(self, batch, batch_idx=None):
        orig, _ = batch
        
        # corrupt the input images with noise/other transforms
        corrupted = self.corrupt_transform(orig)
        gen       = self(corrupted) 
        
        # compute the loss w.r.t. the original images
        train_loss = self.compute_loss(input=gen, target=orig)
        self.log("train_loss", train_loss.item(), on_step=False, on_epoch=True, prog_bar=True) 
        return train_loss
    
    def configure_corruption(self, transforms_list=None, prob=0.5, corruption=None):
        
        if corruption is not None:
            self.corrupt_transform = corruption
            return
        
        # single transform probability
        self.prob = prob
        
        if transforms_list is None: # use default transform
            self.corrupt_transform = transforms.Compose([transforms.RandomHorizontalFlip( p=self.prob ),
                                                         transforms.RandomVerticalFlip( p=self.prob ),
                                                         AddGaussianNoise( p=self.prob, mean=0., std=0.2 ),     # custom
                                                         transforms.RandomErasing( p=self.prob ),
                                                       ])
        else: #transforms_list is a list of transformation objects
            to_compose = []
            for tr in transforms_list:
                if type(tr) is list:
                    to_compose.append( transforms.RandomApply(transforms=tr, p=self.prob) )
                else:
                    to_compose.append( tr( p=self.prob ) )
            
            self.corrupt_transform = transforms.Compose(to_compose)
            
        return




### hyper-parameters space class ----------------------------------------------------------------------------
class DenoisingAutoencoderHPS(SymmetricAutoencoderHPS):
    """ 
    Class that receives in input the hyper-parameters space as a dict and provide a sampling function to be
    used within a Optuna study. This class is for the DenoisingAutoencoder model.
    """        
    def __init__(self, hp_space):
        
        super().__init__(hp_space)
        self.model_class = DenoisingAutoencoder
 
