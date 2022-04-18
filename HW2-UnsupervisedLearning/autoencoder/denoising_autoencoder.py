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
from autoencoder.symmetric_autoencoder import SymmetricAutoencoder
from autoencoder.components import Encoder, Decoder, BaseHPS, BasePLModule
from autoencoder.components import get_activation, conv_output_shape, conv_transpose_output_shape

from data_management.custom_transforms import AddGaussianNoise



### Denoising Autoencoder class with symmetric Encoder/Decoder pair -----------------------------------------
class DenoisingAutoencoder(SymmetricAutoencoder):
    """
    Denoising Autoencoder BUG BUG
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
                 corruption_p  : float  = 0.5,    #probability of single transformations to be applied
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
        
        if corruption is None:
            randomcrop = transforms.RandomCrop( size=(self.input_size[1], self.input_size[2]) )
            
            # default transform
            corruption = transforms.Compose([transforms.RandomApply(transforms=[randomcrop)], p=corruption_p),
                                             transforms.RandomHorizontalFlip( p=corruption_p ),
                                             transforms.RandomVerticalFlip( p=corruption_p ),
                                             AddGaussianNoise( prob=corruption_p ),     # custom
                                             transforms.RandomErasing( p=corruption_p ),
                                            ])
        self.corrupt_transform = corruption
        
        
    def training_step(self, batch, batch_idx=None):
        orig, _ = batch
        
        # corrupt the input images with noise/other transforms
        corrupted = self.corrupt_transform(orig)
        gen       = self(corrupted) 
        
        # compute the loss w.r.t. the original images
        train_loss = self.compute_loss(input=gen, target=orig)
        self.log("train_loss", train_loss.item(), on_step=False, on_epoch=True, prog_bar=True) 
        return train_loss





### hyper-parameters space class ----------------------------------------------------------------------------
class DenoisingAutoencoderHPS(BaseHPS):
    """ 
    Class that receives in input the hyper-parameters space as a dict and provide a sampling function to be
    used within a Optuna study. This class is for the DenoisingAutoencoder model.
    """        
    def __init__(self, hp_space):
        
        super().__init__(hp_space)
        self.model_class = DenoisingAutoencoder
    
    
    def _sample_model_params(self, trial):
        ### convolutional parameters
        # layers
        n_conv_configs = len(self.hp_space["conv_configs"])
        conv_config_id = trial.suggest_categorical( "conv_config_id", list(range(n_conv_configs)) )
        conv_config = self.hp_space["conv_configs"][conv_config_id]
        
        # channels
        n_ch_configs = len(self.hp_space["channels_configs"])
        channels_config_id = trial.suggest_categorical( "channels_config_id", list(range(n_ch_configs)) )
        
        n_conv_layers = len(conv_config)
        channels = self.hp_space["channels_configs"][channels_config_id][:n_conv_layers]

        ### linear layers parameters        
        n_linear = trial.suggest_categorical("n_linear", self.hp_space["n_linear"])
        linear_config = [trial.suggest_int(f"linear_units_{kk}", 
                                           self.hp_space["linear_units_range"][0], 
                                           self.hp_space["linear_units_range"][1],
                                           step = self.hp_space["linear_units_range"][2]) 
                         for kk in range(n_linear)
                        ]
        
        ### latent space dimension
        latent_space_dim = trial.suggest_int("latent_space_dim",
                                             self.hp_space["latent_space_range"][0], 
                                             self.hp_space["latent_space_range"][1],
                                             step = self.hp_space["latent_space_range"][2],
                                            )
        ### others 
        instance_norm = trial.suggest_categorical("instance_norm", self.hp_space["instance_norm"])
        Pdropout      = trial.suggest_float("Pdropout", 
                                            self.hp_space["Pdropout_range"][0],
                                            self.hp_space["Pdropout_range"][1],
                                           )
        activation = trial.suggest_categorical("activation", self.hp_space["activations"])
        
        # build model hyperparameters dictionary
        params = {"channels"        : channels,
                  "conv_config"     : conv_config,
                  "linear_config"   : linear_config,
                  "latent_space_dim": latent_space_dim,
                  "instance_norm"   : instance_norm,
                  "Pdropout"        : Pdropout,
                  "activation"      : activation,
                 } 
        
        return params

 
