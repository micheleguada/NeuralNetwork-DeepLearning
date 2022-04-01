# PyTorch imports
import torch
import torch.optim as optim

# python imports
import os
import numpy as np
import time
import datetime
import json

# additional libraries
import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback, EarlyStopping
import plotly.express as px

from optuna.visualization import plot_optimization_history, plot_contour, plot_intermediate_values
from optuna.visualization import plot_parallel_coordinate, plot_param_importances

# DA SISTEMARE


##### Optuna optimization tools #####------------------------------------------------------------------------

class Objective(object):
    """ Objective class for the optuna study optimization to be used with PyTorch-Lightning. """
    def __init__(self, model_class, datamodule, hp_space,
                 max_epochs=50, early_stop_patience=5, use_gpu=False,
                ):  
        self.model_class = model_class
        self.datamodule  = datamodule
        self.hp_space    = hp_space     # object of class: 'HyperparameterSpace'
            
        self.max_epochs  = max_epochs  
        self.patience    = early_stop_patience
        self.use_gpu     = use_gpu
        
    def __call__(self, trial):
        
        print(f"Trial [{trial.number}] started at:", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        ### sample hyperparameters
        params                                         = self.hp_space.sample_model_params(trial)
        optimizer, learning_rate, L2_penalty, momentum = self.hp_space.sample_optim_params(trial)
        
        ### create model
        model = self.model_class(input_size = self.datamodule.get_sample_size(),
                                 params     = params,
                                 optimizer  = optimizer,
                                 learning_rate = learning_rate,
                                 L2_penalty    = L2_penalty,
                                 momentum      = momentum,
                                )
        
        ### create trainer object
        early_stop = EarlyStopping(monitor   = "val_loss", 
                                   min_delta = 0.001, 
                                   patience  = self.patience,
                                   verbose   = False, 
                                   check_on_train_epoch_end=False, # check early_stop at end of validation
                                  )
        pl_pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        
        trainer = pl.Trainer(logger     = False,
                             max_epochs = self.max_epochs,
                             gpus       = 1 if self.use_gpu else None,   #trainer will take care of moving model and datamodule to GPU
                             callbacks  = [pl_pruning, early_stop],
                             enable_checkpointing = False,
                             enable_model_summary = False,
                             deterministic = True,  #use deterministic algorithms to ensure reproducibility
                            )
        
        ### fit the model
        trainer.fit(model, datamodule=self.datamodule)
        
        # storing hyper-parameters as user attribute of trial object for convenience
        hypers = {"params"       : params,
                  "optimizer"    : optimizer,
                  "learning_rate": learning_rate,
                  "L2_penalty"   : L2_penalty,
                  "momentum"     : momentum,
                 }
        trial.set_user_attr("hypers", hypers)
        
        final_valid_loss = trainer.callback_metrics["val_loss"].item()
        
        # getting minimum reached validation loss
        try:
            min_valid_loss = trainer.callback_metrics["min_val_loss"].item()
        except:
            print("INFO: No 'min_val_loss' value logged")
            min_valid_loss = final_valid_loss

        print(f"Trial [{trial.number}] ended at:", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")) 
        print(f"    Valid. loss: {final_valid_loss}; min valid. loss: {min_valid_loss}\n")
        
        return min_valid_loss
    

def inspect_optimization(study, parallel_sets=[], contour_sets=[], 
                         to_show          = "1110000", 
                         save_path        = "Results_test",
                         best_hypers_file = None,
                         fig_shape        = (800,500),
                        ):
    """
    Function that produces a set of useful plots to analyze the hyper-parameters optimization study.
    """
    px.defaults.width  = fig_shape[0]
    px.defaults.height = fig_shape[1]
    
    # ensure folder existence
    os.makedirs(save_path, exist_ok=True)
    
    # save best hyperparameters to file (json)
    if best_hypers_file is not None:
        best_hypers = study.best_trial.user_attrs["hypers"]
        with open(best_hypers_file, 'w') as fp:
            json.dump(best_hypers, fp)
        print("Best hyper-parameters saved to: '"+best_hypers_file+"'.")
    
    # function to plot/save images
    def _deal_with_image(fig, show, filename):
        if show=="1":
            fig.show()
        fig.write_image(filename)   
        print("Image saved at: ", filename)
        return

    ##### scatter plot (time vs value)
    study_df = study.trials_dataframe()
    
    # compute time in minutes and the name for the hover functionality
    study_df["time"] = study_df.apply(lambda row: row['duration'].total_seconds()/60, axis=1)
    study_df["name"] = study_df.apply(lambda row: "Trial "+str(row['number']), axis=1)

    # plot picture
    fig0 = px.scatter(study_df, 
                      x="time", y="value",
                      labels     = {"time":"Training Time [min]", "value":"Min Validation Loss"},
                      color      = "state",
                      symbol     = "state",
                      hover_name = "name", 
                      hover_data = {"time":True,"value":True,"state":False},
                      log_y      = True,
                     )
    fig0.update_traces(marker={'size': 12})

    # plot and save image
    filename = save_path +'/' + "time_vs_value.pdf"
    _deal_with_image(fig0, to_show[0], filename)
    
    ##### optimization history
    fig1 = plot_optimization_history(study)
    filename = save_path +'/' + "optimization_history.pdf"
    _deal_with_image(fig1, to_show[1], filename)
    
    ##### intermediate values
    fig2 = plot_intermediate_values(study)
    filename = save_path +'/' + "intermediate_values.pdf"
    _deal_with_image(fig2, to_show[2], filename)
    
    ##### parameters importance
    fig3 = plot_param_importances(study)
    filename = save_path +'/' + "importances.pdf"
    _deal_with_image(fig3, to_show[3], filename)
    
    ##### parallel coordinate plots
    for conf in parallel_sets:        
        # build suffix 
        suffix = "_" + conf[0]   # first is the suffix for the filename
        
        fig = plot_parallel_coordinate(study, params=conf[1:])
        
        filename = save_path +'/' + "parallel" + suffix + ".pdf"
        _deal_with_image(fig, to_show[4], filename)
        
    ##### contour plots
    for conf in contour_sets:        
        # build suffix based on passed parameters
        suffix = "_".join(conf)
        
        fig = plot_contour(study, params=conf)
        
        filename = save_path +'/' + "contour_" + suffix + ".pdf"
        _deal_with_image(fig, to_show[5], filename)
        
        
    ##### value vs latent dimension
    latent_df = study_df.sort_values(by="params_conv_config_id")

    fig = px.scatter(latent_df, 
                     x="params_latent_space_dim", y="value",
                     labels = {"params_latent_space_dim":"Latent space dimension",
                               "value"                  :"Min Validation Loss",
                               "color"                  :"Conv. config ID",
                               "params_optimizer"       :"Optimizer",
                              },
                     color  = latent_df["params_conv_config_id"].astype(str),
                     color_discrete_sequence = px.colors.qualitative.Set1,
                     symbol = latent_df["params_optimizer"],
                     hover_name = "name", 
                     log_y  = True,
                    )
    fig.update_traces(marker={'size': 12})
    filename = save_path +'/' + "latent_vs_value.pdf"
    _deal_with_image(fig, to_show[6], filename)
    
    return



##### train and test tools #####-----------------------------------------------------------------------------

class LossesTracker(Callback):
    """
    Class that store the values for the train and validation losses.
    """
    def __init__(self, train_name="train_loss", val_name="val_loss"): 
        self.train_name = train_name
        self.val_name   = val_name
        self.train = []
        self.valid = []
    
    def on_train_epoch_end(self, trainer, module):
        self.train.append(trainer.logged_metrics[self.train_name])
        return
    
    def on_validation_epoch_end(self, trainer, module):       
        self.valid.append(trainer.logged_metrics[self.val_name])
        return



def run_training(model_class, datamodule, hypers, callbacks, max_epochs=100, ep_ratio=1., use_gpu=False):
    """
    Function to run the training of the autoencoder. 
        - model_class    : class onject of the autoencoder
        - datamodule     : pytorch-lightning datamodule
        - hypers         : hyperparameters of the model and optimizer
        - callbacks      : list of callbacks to use
        - max_epochs     : maximum number of training epochs
        - ep_ratio       : check validation "ep_ratio" times in every training epoch
        - use_gpu        : boolean flag to activate GPU usage
    """
    print( "Training started at:", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") )

    ### define model and hyper-parameters
    model = model_class(input_size    = datamodule.get_sample_size(),
                        params        = hypers["params"],
                        optimizer     = hypers["optimizer"],
                        learning_rate = hypers["learning_rate"],
                        L2_penalty    = hypers["L2_penalty"],
                        momentum      = hypers["momentum"],
                       )
    
    ### define trainer
    trainer = pl.Trainer(logger     = False,
                         max_epochs = max_epochs,
                         gpus       = 1 if use_gpu else None,
                         callbacks  = callbacks,
                         val_check_interval   = (1./ep_ratio),
                         enable_model_summary = False,
                         num_sanity_val_steps = 0,     # disable validation sanity check before training
                        )
    
    trainer.fit(model, datamodule=datamodule) # run the training
    
    print( "Training ended at:", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") )
    
    return model, trainer, callbacks

