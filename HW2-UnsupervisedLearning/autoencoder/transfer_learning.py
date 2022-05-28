# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

# python imports
import numpy as np

# additional libraries
import pytorch_lightning as pl

# custom modules 
from autoencoder.components import LinearBlock, Encoder, BasePLModule, get_activation


               
        
class EncoderClassifier(BasePLModule):
    """
    Transfer learning model made of some linear layers that uses a preprocessed input from a pretrained 
    autoencoder to classify images.
    """
    def __init__(self, 
                 input_dim       : int,             # last pretrained layer size
                 num_classes     : int    = 10,
                 linear_config   : list   = [],     # additional layers of the classifier
                 activation      : str    = "relu",
                 optimizer       : str    = "adam",
                 learning_rate   : float  = 0.001,
                 L2_penalty      : float  = 0.,
                 momentum        : float  = None,   
                ):
        super(EncoderClassifier, self).__init__(optimizer     = optimizer, 
                                                learning_rate = learning_rate, 
                                                L2_penalty    = L2_penalty, 
                                                momentum      = momentum,
                                               )      
        self.save_hyperparameters() # save hyperparameters when checkpointing
        
        # activation function
        self.act_name = activation
        
        # create classifier
        self.num_classes = num_classes
        self.input_dim   = input_dim
        
        layers = []
        units_list = [self.input_dim] + linear_config
        for idx in range(1, len(units_list)):
            layers.append( LinearBlock(units_list[idx-1],
                                       units_list[idx],
                                       activation = activation,
                         )            ) 
        layers.append( nn.Linear(units_list[-1], self.num_classes) )
                  
        self.classifier = nn.Sequential(*layers) 
        
    def forward(self, preprocessed_input, additional_out=False):
        x = self.classifier(preprocessed_input)
        return x
    
    def compute_loss(self, output, target):
        return nn.functional.cross_entropy(output, target)
    
    
    def training_step(self, batch, batch_idx=None):        
        data, target = batch
        output = self(data)  
        
        train_loss = self.compute_loss(output, target)
        self.log("train_loss", train_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return train_loss  

    def validation_step(self, batch, batch_idx=None):
        data, target = batch
        output = self(data)
        
        val_loss = self.compute_loss(output, target)
        self.log("val_loss", val_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return val_loss
    
    def predict_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        
        test_loss = self.compute_loss(output, target)
        return {"outputs"    : output, 
                "labels"     : target, 
                "test_loss"  : test_loss.item(), 
                "batch_size" : target.size(),    # can be useful if batch size are not always the same
               }
    
