# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset, IterableDataset

# python imports
import numpy as np
import random
from collections import deque

# additional libraries
import pytorch_lightning as pl




### some utility functions ----------------------------------------------------------------------------------
# activation function from name
def get_activation(act):
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif act == "silu":
        return nn.SiLU(inplace=True)   #sigmoid linear unit
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


# print shape of image through convolutional layers
def print_conv_shapes(input_shape, config):
    """
    Utility function that computes and prints the shape of image through the convolutional layers.
    """    
    shape = input_shape
    shapes = [shape]
    for layer in config:
        shape = conv_output_shape(shape, layer[0], layer[1], layer[2])
        shapes.append( shape )
        
    print("  ", " -> ".join([str(item) for item in shapes]))
    
    return shapes
