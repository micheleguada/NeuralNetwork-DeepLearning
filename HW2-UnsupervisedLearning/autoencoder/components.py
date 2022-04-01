# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# python imports
import numpy as np

# additional libraries
import pytorch_lightning as pl

# docs da completare ########

### Base blocks for model architecture ----------------------------------------------------------------------
# convolutional block
class ConvBlock(nn.Module):
    """
    Base convolutional block composed of:
        - a convolutional layer
        - eventual batch normalization
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
        # eventually add batch normalization
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)
        
    def forward(self, x, additional_out=False):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.act(x)
        return x
    
# deconvolutional block
class ConvTransposeBlock(nn.Module):
    """
    Base deconvolutional block composed of:
        - a deconvolutional layer
        - eventual batch normalization
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
        # eventually add batch normalization
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)
        
    def forward(self, x, additional_out=False):
        x = self.deconv(x)
        if self.batch_norm:
            x = self.bn(x)
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
        
        
        
        
### encoder/decoder classes ---------------------------------------------------------------------------------
# Encoder
class Encoder(nn.Module):
    """
    Convolutional Encoder:
        input_size : size of input image (channels, width, height)
        params     : dictionary containing hyper-parameters of the model
        flatten_dim: units of the flatten layer
    """
    def __init__(self, input_size, params, flatten_dim):
        super().__init__()

        self.input_size = input_size
        self.hp = params # hyper-parameters
            
        ### convolutional layers
        channels = input_size[0] # first layer input channels (1 or 3)
        
        conv_list = []
        for idx,values in enumerate(self.hp["conv_config"]):
            out_channels = self.hp["channels"][idx]
            conv_list.append( ConvBlock(channels, out_channels, values, 
                                        self.hp["batch_norm"], self.hp["activation"],
                            )          )
            channels = out_channels # redefine the number of input channels for the next layer
            
        self.encoder_cnn = nn.Sequential(*conv_list)
        
        ### flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ### linear layers
        in_units = flatten_dim #number of input units needed after the flatten layer
            
        linear_list = []
        for value in self.hp["linear_config"]:
            linear_list.append( LinearBlock(in_units, value, self.hp["Pdropout"], self.hp["activation"]) )
            
            in_units = value # redefine the number of input units for the next layer
        
        # append latent space layer
        linear_list.append( nn.Linear(in_units, self.hp["latent_space_dim"]) )
        
        self.encoder_lin = nn.Sequential(*linear_list)
            
    def forward(self, x, additional_out=False):
        x = self.encoder_cnn(x) #convolutions
        x = self.flatten(x)
        x = self.encoder_lin(x) #linear layers
        return x
        
# Decoder
class Decoder(nn.Module):
    """
    Convolutional Decoder:
        out_size     : size of decoded image (equal to input image size) (width, height, channels)
        params       : dictionary containing hyper-parameters of the model.
        unflatten_dim: input shape at the first deconvolutional layer (tuple)
    """
    def __init__(self, out_size, params, unflatten_dim, out_padding):

        super().__init__()

        self.out_size = out_size
        self.hp = params # hyper-parameters
        
        ### linear layers
        in_units = self.hp["latent_space_dim"]
            
        linear_list = [get_activation(self.hp["activation"])] #append an activation after the latent space
        for value in self.hp["linear_config"]:
            linear_list.append( LinearBlock(in_units, value, self.hp["Pdropout"], self.hp["activation"]) )
            
            in_units = value # redefine the number of input units for the next layer
        
        # append flatten linear block
        linear_list.append( LinearBlock(in_units, 
                                        np.prod(unflatten_dim), 
                                        self.hp["Pdropout"], 
                                        self.hp["activation"],
                          )            )
        
        self.decoder_lin = nn.Sequential(*linear_list)
        
        ### unflatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=unflatten_dim)
        
        ### deconvolutional layers        
        deconv_list = []       
        for idx,values in enumerate(self.hp["conv_config"][:-1]):
            in_channels  = self.hp["channels"][idx]
            out_channels = self.hp["channels"][idx+1]
            deconv_list.append( ConvTransposeBlock(in_channels, out_channels, values, out_padding[idx],
                                                   self.hp["batch_norm"], self.hp["activation"],
                              )                   )
        # append last deconvolutional layer
        deconv_list.append( nn.ConvTranspose2d(in_channels  = self.hp["channels"][-1],
                                               out_channels = self.out_size[0], # image channels (1 or 3) 
                                               kernel_size  = self.hp["conv_config"][-1][0], 
                                               stride       = self.hp["conv_config"][-1][1], 
                                               padding      = self.hp["conv_config"][-1][2], 
                                               output_padding = out_padding[-1],
                          )                   )
        # append sigmoid layer at the end to ensure proper range for pixel values
        deconv_list.append( nn.Sigmoid() )
        
        self.decoder_cnn = nn.Sequential(*deconv_list)

    def forward(self, x, additional_out=False):
        
        x = self.decoder_lin(x) #linear layers
        x = self.unflatten(x)
        x = self.decoder_cnn(x) #deconvolutional layers
        
        return x
    
    
    
### Base hyper-parameters space class -----------------------------------------------------------------------
class BaseHPS(object):
    """ 
    Class that receives in input the hyper-parameters space as a dict and provide a sampling function to be
    used within a Optuna study. This is the base class to be extended for every actual model with its own
    parameters names and ranges.
    """
    def __init__(self, hp_space):
        
        self.hp_space = hp_space
        
    def sample_model_params(self, trial):
        """
        Function that returns a dict with the sampled values from the hyper-parameters space.
           - trial: is the 'trial' object provided by the optuna framework.
        """
        
        pass
    
    def sample_optim_params(self, trial):        
        ### optimizer parameters        
        optimizer     = trial.suggest_categorical("optimizer", self.hp_space["optimizers"])
        learning_rate = trial.suggest_float("learning_rate", 
                                            self.hp_space["learning_rate_range"][0], 
                                            self.hp_space["learning_rate_range"][1],
                                            log=True,
                                           )
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
        
        return optimizer, learning_rate, L2_penalty, momentum 
        
        

### some utility functions ----------------------------------------------------------------------------------
# activation function from name
def get_activation(act):
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "tanh":
        return nn.Tanh()
    elif act == "sigmoid":
        return nn.Sigmoid()
    else: 
        raise ValueError(f"Activation {act} is currently not supported.")
        
# convolutional layer output shape
def conv_output_shape(input_shape, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function that computes the output of a convolutional or pooling layer for 
    a given rectangular input shape (tuple of integers with 'Width' and 'Height' in pixels). 
    Note that:    
        - shape of kernel is assumed to be square.
        - stride, padding and dilation are assumed to be symmetric.
    """
    dim1 = int( ((input_shape[0] + (2*pad) - (dilation*(kernel_size-1)) -1 )/ stride) +1 )
    dim2 = int( ((input_shape[1] + (2*pad) - (dilation*(kernel_size-1)) -1 )/ stride) +1 )
    return (dim1, dim2)

# transpose convolutional layer output shape
def conv_transpose_output_shape(input_shape, kernel_size=1, stride=1, pad=0, dilation=1, out_padding=(0,0)):
    """
    Utility function that computes the output of a transpose convolutional layer for 
    a given rectangular input shape (tuple of integers with 'Width' and 'Height' in pixels). 
    Note that:    
        - shape of kernel is assumed to be square.
        - stride, padding and dilation are assumed to be symmetric.
    """
    dim1 = int( (input_shape[0]-1)*stride - 2*pad + dilation*(kernel_size-1) + out_padding[0] +1 )
    dim2 = int( (input_shape[1]-1)*stride - 2*pad + dilation*(kernel_size-1) + out_padding[1] +1 )
    return (dim1, dim2)
 
