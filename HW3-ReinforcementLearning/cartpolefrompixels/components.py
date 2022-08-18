# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset, IterableDataset

# python imports
import numpy as np
import random
from collections import deque
from abc import ABC, abstractmethod

# additional libraries
import pytorch_lightning as pl


# custom modules
from .utilities import get_activation, conv_output_shape


######################## POLICY NETWORK #####################################################################

### Base blocks for model architecture ----------------------------------------------------------------------
# convolutional block
class ConvBlock(nn.Module):
    """
    Base convolutional block composed of:
        - a convolutional layer
        - eventual instance normalization
        - activation function
    """
    def __init__(self, input_channels, out_channels, config, batch_norm=False, activation="relu"):

        super().__init__()
        
        self.conv = nn.Conv2d(in_channels  = input_channels, 
                              out_channels = out_channels, 
                              kernel_size  = config[0], 
                              stride       = config[1],
                              padding      = config[2],
                             )
        # eventually add instance normalization
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)
        
    def forward(self, x, additional_out=False):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.act(x)
        return x
    
# deconvolutional block
class ConvTransposeBlock(nn.Module):
    """
    Base deconvolutional block composed of:
        - a deconvolutional layer
        - eventual instance normalization
        - activation function
    """
    def __init__(self, input_channels, out_channels, config, out_pad, batch_norm=False, activation="relu"):

        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels    = input_channels, 
                                         out_channels   = out_channels, 
                                         kernel_size    = config[0], 
                                         stride         = config[1],
                                         padding        = config[2],
                                         output_padding = out_pad,
                                        )
        # eventually add instance normalization
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)
        
    def forward(self, x, additional_out=False):
        x = self.deconv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.act(x)
        return x
    
# linear block
class LinearBlock(nn.Module):
    """
    Base linear block composed of:
        - a linear layer
        - eventual dropout
        - activation function
    """
    def __init__(self, input_dim, out_dim, Pdropout=0., activation="relu"):

        super().__init__()
        
        self.lin = nn.Linear(input_dim, out_dim)
        
        # eventually add a dropout layer
        self.Pdropout = Pdropout
        if self.Pdropout > 0.:
            self.drp = nn.Dropout(p=Pdropout) 
        self.act = get_activation(activation)
        
    def forward(self, x, additional_out=False):
        x = self.lin(x)
        if self.Pdropout > 0.:
            x = self.drp(x)
        x = self.act(x)
        return x
    

### policy network class ------------------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    
    def __init__(self, 
                 state_space_dim, 
                 action_space_dim,
                 params = None, 
                ):
        super().__init__()
        
        if params is None:
            conv_channels = [16,32,32]
            linear_units  = [256]
            activation    = "tanh"
            batch_norm    = True
            dropout       = 0.
            conv_config   = [[5,2,0],  
                             [5,2,0],
                             [3,2,1],
                            ]
        else:
            conv_channels = params["conv_channels"]
            linear_units  = params["linear_units"]
            activation    = params["activation"]
            batch_norm    = params["batch_norm"]
            dropout       = params["dropout"]
            conv_config   = params["conv_config"]
        
        self.state_space_dim = state_space_dim        
        
        # activation
        self.act = get_activation(activation)
        
        ### convolutional layers
        self.conv_channels = conv_channels
        # layers configurations
        self.conv_config = conv_config
                           
        channels = state_space_dim[0] # first layer input channels (1 or 3)
        conv_list = []
        for idx, out_channels in enumerate(conv_channels):
            conv_list.append( ConvBlock(channels, out_channels, self.conv_config[idx], 
                                        batch_norm, activation,
                            )          )
            channels = out_channels # redefine the number of input channels for the next layer
            
        self.policy_cnn = nn.Sequential(*conv_list)
        
        ### flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### linear layers
        in_units = self._compute_flatten_dim(self.state_space_dim,  #number of input units needed after the flatten layer
                                             self.conv_config, 
                                             self.conv_channels,
                                            )
        linear_list = []
        for value in linear_units:
            linear_list.append( LinearBlock(in_units, value, dropout, activation) )
            in_units = value # redefine the number of input units for the next layer
            
        # append last linear layer
        linear_list.append( nn.Linear(in_units, action_space_dim) )
        
        self.policy_lin = nn.Sequential(*linear_list)

    def forward(self, x):
        
        x = self.policy_cnn(x)  # convolutions
        x = self.flatten(x)
        x = self.policy_lin(x)  # linear layers
        return x
    
    def _compute_flatten_dim(self, shape, conv_config, conv_channels):
        """Keep trace of the image size through conv. layers """
        intermediate_shapes = [(shape[1], shape[2])]        
        for kk, values in enumerate(conv_config):
            intermediate_shapes.append(conv_output_shape(intermediate_shapes[kk],   # (W,H)
                                                         values[0], # kernel size
                                                         values[1], # stride
                                                         values[2], # padding
                                      )                 )
        channels   = conv_channels[-1]
        to_flatten = intermediate_shapes[-1]
        
        return np.prod( (channels, to_flatten[0], to_flatten[1]) )


### Abstract classes ----------------------------------------------------------------------------------------
# Base abstract module     
class BasePLModule(ABC, pl.LightningModule):
    """
    Base PyTorch Lightning module to be derived.
    """
    def __init__(self,
                 optimizer     : str    = "adam",
                 learning_rate : float  = 0.001,
                 L2_penalty    : float  = 0.,
                 momentum      : float  = None,   #ignored if optimizer is optim.Adam
                ):
        super().__init__()
        self.optim_name = optimizer
        self.momentum   = momentum
        self.lr         = learning_rate
        self.L2         = L2_penalty
        
    @abstractmethod
    def forward(self, x, additional_out=False):
        pass
    
    @abstractmethod
    def compute_loss(self, input, target):
        pass
        
    def configure_optimizers(self):
        if   (self.optim_name == "adam"):
            return optim.Adam(self.parameters(), self.lr, weight_decay=self.L2)
        elif (self.optim_name == "sgd"):
            return optim.SGD(self.parameters(), self.lr, momentum=self.momentum, weight_decay=self.L2)
        elif (self.optim_name == "adamax"):
            return optim.Adamax(self.parameters(), self.lr, weight_decay=self.L2)
        elif (self.optim_name == "rmsprop"):
            return optim.RMSprop(self.parameters(), self.lr, momentum=self.momentum, weight_decay=self.L2)
        else:
            raise ValueError("Optimizer "+self.optim_name+" not supported.")
    

# Base hyper-parameters space class -------------------------------------------------------------------------
class BaseHPS(ABC):
    """ 
    Class that receives in input the hyper-parameters space as a dict and provide a sampling function to be
    used within a Optuna study. This is the base class to be extended for every actual model with its own
    parameters names and ranges.
    """
    def __init__(self, hp_space):
        
        self.model_class = None
        self.hp_space = hp_space
     
    def sample_configuration(self, trial, auto_lr_find=False, use_gpu=False, datamodule=None):
        
        params                                         = self._sample_model_params(trial)
        optimizer, learning_rate, L2_penalty, momentum = self._sample_optim_params(trial, params, auto_lr_find, use_gpu, datamodule)
        
        return params, optimizer, learning_rate, L2_penalty, momentum
     
    @abstractmethod
    def _sample_model_params(self, trial):
        """
        Function that returns a dict with the sampled values from the hyper-parameters space.
           - trial: is the 'trial' object provided by the optuna framework.
        """
        pass
    
    def _sample_optim_params(self, trial, params, auto_lr_find=False, use_gpu=False, datamodule=None):        
        ### optimizer parameters        
        optimizer     = trial.suggest_categorical("optimizer", self.hp_space["optimizers"])
        L2_penalty    = trial.suggest_float("L2_penalty", 
                                            self.hp_space["L2_penalty_range"][0], 
                                            self.hp_space["L2_penalty_range"][1],
                                            log=True,
                                           )
        if optimizer in ["adam","adamax"]:
            momentum = None
        else:
            momentum = trial.suggest_float("momentum", 
                                           self.hp_space["momentum_range"][0], 
                                           self.hp_space["momentum_range"][1],
                                          )           
            
        # learning_rate
        if auto_lr_find:  # auto tune learning rate from Pytorch-Lightning
            learning_rate = self._estimate_lr(params, optimizer, L2_penalty, momentum, datamodule, use_gpu)
            print("Selected learning rate: ", learning_rate)
            
            # fake sample learning rate value
            learning_rate = trial.suggest_float("learning_rate", learning_rate, learning_rate)
        else:             # sample with optuna
            learning_rate = trial.suggest_float("learning_rate", 
                                                self.hp_space["learning_rate_range"][0], 
                                                self.hp_space["learning_rate_range"][1],
                                                log=True,
                                               )        
        return optimizer, learning_rate, L2_penalty, momentum 

    
    def _estimate_lr(self, params, optimizer, L2_penalty, momentum, datamodule, use_gpu=False):        
        ### create model
        model = self.model_class(input_size = datamodule.get_sample_size(),
                                 params     = params,
                                 optimizer     = optimizer,
                                 learning_rate = 1.,       # will be optimized by pytorch lightning
                                 L2_penalty    = L2_penalty,
                                 momentum      = momentum,
                                )
        ### create trainer object       
        trainer = pl.Trainer(logger     = False,
                             max_epochs = 50,
                             gpus       = 1 if use_gpu else None,
                             enable_checkpointing = False,
                             enable_model_summary = False,
                             deterministic  = True,  # use deterministic algorithms for reproducibility
                             auto_lr_find   = True,  # if True, at beggining estimate a good learning rate
                             detect_anomaly = True,
                            )
        ### find learning rate
        lr_finder = trainer.tune(model, datamodule=datamodule)["lr_find"]
        learning_rate = lr_finder.results["lr"][lr_finder._optimal_idx]
               
        return learning_rate
        
  
