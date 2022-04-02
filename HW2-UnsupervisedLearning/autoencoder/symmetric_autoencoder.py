# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# python imports
import numpy as np

# additional libraries
import pytorch_lightning as pl

# custom modules 
from autoencoder.components import Encoder, Decoder, BaseHPS
from autoencoder.components import get_activation, conv_output_shape, conv_transpose_output_shape


# docs da completare ########  
    

### Autoencoder class with symmetric Encoder/Decoder pair ---------------------------------------------------
class SymmetricAutoencoder(pl.LightningModule):    
    """
    Autoencoder BUG BUG
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
                ):
        super().__init__()
        self.save_hyperparameters() # save hyperparameters when checkpointing
        
        self.input_size = input_size
        
        if params is None:
            # default hyper-parameters
            params = {"channels"        : [16, 32, 64], # number of feature maps 
                      "conv_config"     : [[3, 2, 0],   # conv. layer settings: kernel size, stride, padding 
                                           [3, 2, 0],
                                           [3, 1, 0],
                                          ],
                      "linear_config"   : [128, 64],
                      "latent_space_dim": 8,
                      "batch_norm"      : False,
                      "Pdropout"        : 0.,
                      "activation"      : "relu",
                     } 
                      
        self.enc_hp = params
        self.dec_hp = self._build_decoder_config(params)
        
        self.optim_name = optimizer
        self.momentum   = momentum
        self.lr         = learning_rate
        self.L2         = L2_penalty
        
        # compute shape after_convolution/before_deconvolution and output padding
        conv_shape, out_padding = self._compute_shapes()
        
        self.encoder = encoder_class(input_size, 
                                     params      = self.enc_hp, 
                                     flatten_dim = np.prod(conv_shape),
                                    )
        self.decoder = decoder_class(input_size, 
                                     params        = self.dec_hp, 
                                     unflatten_dim = conv_shape, 
                                     out_padding   = out_padding,
                                    ) 
        self.min_val_loss = None  # to store minimum validation loss
           
    def forward(self, x, additional_out=False):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def training_step(self, batch, batch_idx=None):
        orig, _ = batch
        gen     = self.forward(orig)  
        
        train_loss = nn.functional.mse_loss(input=gen, target=orig)
        self.log("train_loss", train_loss.item(), on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx=None):
        orig, _ = batch
        gen     = self.forward(orig)
        
        val_loss = nn.functional.mse_loss(input=gen, target=orig)
        self.log("val_loss", val_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return val_loss
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack(outputs).mean()
        
        # store minimum reached validation loss
        if (self.min_val_loss is None) or (self.min_val_loss > val_loss.item()):
            self.min_val_loss = val_loss.item()
            self.log("min_val_loss", self.min_val_loss, on_step=False, on_epoch=True)
        return    
    
    def test_step(self, batch, batch_idx):
        # test loss
        orig, _ = batch
        gen     = self.forward(orig)
        
        test_loss = nn.functional.mse_loss(input=gen, target=orig)
        self.log("test_loss", test_loss.item(), on_step=False, on_epoch=True)
        
        return
        
    
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
            
    
    def _compute_shapes(self):
        """Keep trace of the image size through conv. layers and compute proper output padding"""
        intermediate_shapes = [(self.input_size[1], self.input_size[2])]        
        for kk,values in enumerate(self.enc_hp["conv_config"]):
            intermediate_shapes.append(conv_output_shape(intermediate_shapes[kk],   # (W,H)
                                                         values[0], # kernel size
                                                         values[1], # stride
                                                         values[2], # padding
                                      )                 )
        channels   = self.enc_hp["channels"][-1]
        to_flatten = intermediate_shapes[-1]
        
        # compute proper output padding to ensure symmetry of intermediate shapes
        out_padding = []
        reversed_shapes = intermediate_shapes[::-1]
        for kk,values in enumerate(self.dec_hp["conv_config"]):
            upsampled = conv_transpose_output_shape(reversed_shapes[kk],
                                                    values[0],
                                                    values[1],
                                                    values[2],
                                                   )
            diff = map(lambda x,y: x-y, reversed_shapes[kk+1], upsampled)
            out_padding.append( tuple(diff) )
        
        return (channels, to_flatten[0], to_flatten[1]), out_padding
    
    def _build_decoder_config(self, params):
        """Generate dictionary of decoder hyperparameters from the encoder ones"""
        dec_hp = params.copy()
        dec_hp["channels"]      = dec_hp["channels"][::-1]      # reverse order of convolutional layers
        dec_hp["conv_config"]   = dec_hp["conv_config"][::-1]   #  ...
        dec_hp["linear_config"] = dec_hp["linear_config"][::-1] #  and of linear layers
        
        return dec_hp
    
        
        

### hyper-parameters space class ----------------------------------------------------------------------------
class SymmetricAutoencoderHPS(BaseHPS):
    """ 
    Class that receives in input the hyper-parameters space as a dict and provide a sampling function to be
    used within a Optuna study. This class is for the SymmetricAutoencoder model.
    """        
    def sample_model_params(self, trial):
        ### convolutional parameters
        n_configs = len(self.hp_space["conv_configs"])
        conv_config_id = trial.suggest_categorical( "conv_config_id", list(range(n_configs)) )
        conv_config = self.hp_space["conv_configs"][conv_config_id]
        
        # DANGER qui voglio modificare il modo in cui faccio il sampling, ma non so come ????? DANGER
        #n_conv_layers = len(conv_config)
        #channel_1st = trial.suggest_int("channels_0", 
                                        #self.hp_space["channels_range"][0], 
                                        #self.hp_space["channels_range"][1],
                                        #step = self.hp_space["channels_range"][2],
                                       #) 
        
        #channel_f = []
        #for layer_id in range(1,n_conv_layers):
            #channel_f.append( trial.suggest_int(f"channel_factor_{layer_id}", 1, 2) )
        
        channels = [trial.suggest_int(f"channels_{ii}", 
                                      self.hp_space["channels_range"][0], 
                                      self.hp_space["channels_range"][1],
                                      step = self.hp_space["channels_range"][2],
                                     )
                    for ii in range(len(conv_config))
                   ]

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
        batch_norm = trial.suggest_categorical("batch_norm", self.hp_space["batch_norm"])
        Pdropout   = trial.suggest_float("Pdropout", 
                                         self.hp_space["Pdropout_range"][0],
                                         self.hp_space["Pdropout_range"][1],
                                        )
        activation = trial.suggest_categorical("activation", self.hp_space["activations"])
        
        # build model hyperparameters dictionary
        params = {"channels"        : channels,
                  "conv_config"     : conv_config,
                  "linear_config"   : linear_config,
                  "latent_space_dim": latent_space_dim,
                  "batch_norm"      : batch_norm,
                  "Pdropout"        : Pdropout,
                  "activation"      : activation,
                 } 
        
        return params


