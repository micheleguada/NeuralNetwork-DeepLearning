# PyTorch imports
import torch
import torch.optim as optim

# python imports
import os
import numpy as np
import time
import datetime
import json
import functools

# additional libraries
import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback, EarlyStopping
import plotly.express as px

import optuna.visualization as ov
#from optuna.visualization import plot_optimization_history, plot_contour, plot_intermediate_values
#from optuna.visualization import plot_parallel_coordinate, plot_param_importances

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
                                 optimizer     = optimizer,
                                 learning_rate = learning_rate,
                                 L2_penalty    = L2_penalty,
                                 momentum      = momentum,
                                )
        
        ### create trainer object
        losses_tracker = LossesTracker()
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
                             callbacks  = [pl_pruning, early_stop, losses_tracker],
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
        
        # getting minimum reached validation loss as during final training we will use checkpointing
        min_valid_loss   = torch.min(losses_tracker.valid)

        print(f"Trial [{trial.number}] ended at:", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")) 
        print(f"    Valid. loss: {final_valid_loss}; min valid. loss: {min_valid_loss}\n")
        
        return min_valid_loss
    
    
    
# decorator to add a function to a dictionary
def make_decorator(dictionary):
    def decorator_add_to_dict(key):
        def wrapper(func):
            dictionary.update({key:func})
            return func
        return wrapper
    return decorator_add_to_dict
    
class OptimizationInspector(object):
    """
    DANGER DANGER DANGER
    """
    # dict of plotting function
    plot_dict = {}
    _plot_dict_member = make_decorator(plot_dict)    
    
    def __init__(self, study, save_path="Results_test", figsize=(1024,600)):
        
        self.study     = study
        self.save_path = save_path + "/"
        
        # set figsize
        px.defaults.width  = figsize[0]
        px.defaults.height = figsize[1]
        
        # ensure folder existence
        os.makedirs(self.save_path, exist_ok=True)
        
        self.print_summary()
        
    def save_best_hypers_json(self, best_hypers_file):
        # save best hyperparameters to file (json)
        best_hypers = self.study.best_trial.user_attrs["hypers"]
        with open(best_hypers_file, 'w') as fp:
            json.dump(best_hypers, fp)
        print("Best hyper-parameters saved to: '"+best_hypers_file+"'.")
    
    def _handle_image(self, fig, show, name, save):
        # function to plot/save images
        if show == "1":
            fig.show()
        if (save) and (name is not None):
            full_path = self.save_path + name + ".pdf"
            fig.write_image(full_path)   
            print("New image saved: ", full_path)
        return
            
    
    def print_summary(self):
        print("Summary of the Optuna study: ", self.study.study_name)
        print("   Attempted trials: ", len(self.study.trials) )
        study_df  = self.study.trials_dataframe()
        completed = len(study_df[study_df["state"]=="COMPLETE"])
        pruned    = len(study_df[study_df["state"]=="PRUNED"  ])
        print("   Completed trials: ", completed)
        print("   Pruned trials   : ", pruned   )
        print("   Best Trial ID   : ", self.study.best_trial.number)
        print("   Best value      : ", self.study.best_value )
        
        best_hypers = self.study.best_trial.user_attrs["hypers"]        
        print("\nBest set of hyper-parameters:\n", best_hypers)
        
        # print dict of available plot functions
        print("\nAvailable plot methods:")
        print(list(self.plot_dict))
        
        return
    
    def plot_all(self, parallel_sets = [], contour_sets = [], slice_sets = [],
                 show = "111000000", save = True,
                ):
        """
        Produce all the defined plots in this class (actually 9). Showing is controlled by the variable 'show'.
        It can also save all the plotted pictures. Files names are fixed to some default value.
         - show : binary string of lenght 9 ('1' to show image, '0' to not show).
         - save : if to save the pictures on disk
        """
        data_dict = {"parallel": parallel_sets,
                     "contour" : contour_sets ,
                     "slice"   : slice_sets   ,
                    }
        
        for idx,key in enumerate(self.plot_dict):
            self.plot_dict[key](self, show=show[idx], save=save, data_dict=data_dict)
            
        return
    
    @_plot_dict_member("time_vs_value")
    def time_vs_value(self, show="0", name="time_vs_value", save=False, data_dict=None):  
        
        study_df = self.study.trials_dataframe()
        
        # compute time in minutes and the name for the hover functionality
        study_df["time"] = study_df.apply(lambda row: row['duration'].total_seconds()/60, axis=1)
        study_df["name"] = study_df.apply(lambda row: "Trial "+str(row['number']), axis=1)

        # plot picture
        fig = px.scatter(study_df, 
                         x="time", y="value",
                         labels     = {"time":"Training Time [min]", "value":"Min Validation Loss"},
                         color      = "state",
                         symbol     = "state",
                         hover_name = "name", 
                         hover_data = {"time":True,"value":True,"state":False},
                         log_y      = True,
                        )
        fig.update_traces(marker={'size': 8})

        # plot and save image
        self._handle_image(fig, show, name, save)
        return
    
    @_plot_dict_member("optimization_history")
    def optimization_history(self, show="0", name="optimization_history", save=False, data_dict=None):
        fig = ov.plot_optimization_history(self.study)
        self._handle_image(fig, show, name, save)
        return
    
    @_plot_dict_member("intermediate_values")
    def intermediate_values(self, show="0", name="intermediate_values", save=False, data_dict=None):
        fig = ov.plot_intermediate_values(self.study)
        self._handle_image(fig, show, name, save)
        return
    
    @_plot_dict_member("importances")
    def importances(self, show="0", name="importances", save=False, data_dict=None):
        fig = ov.plot_param_importances(self.study)
        self._handle_image(fig, show, name, save)
        return
        
    @_plot_dict_member("latent_dim_vs_value")
    def latent_dim_vs_value(self, show="0", name="latent_dim_vs_value", save=False, data_dict=None):

        study_df  = self.study.trials_dataframe()
        study_df["name"] = study_df.apply(lambda row: "Trial "+str(row['number']), axis=1)
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
        fig.update_traces(marker={'size': 8})
        self._handle_image(fig, show, name, save)
        
        return
    
    @_plot_dict_member("conv_vs_channels")
    def conv_vs_channels(self, show="0", name="conv_vs_channels", save=False, data_dict=None):
        
        study_df  = self.study.trials_dataframe()
        study_df["name"] = study_df.apply(lambda row: "Trial "+str(row['number']), axis=1)

        fig = px.scatter(study_df, 
                         x="params_channels_config_id", y="params_conv_config_id",
                         labels = {"params_channels_config_id":"Channels config ID",
                                   "params_conv_config_id"    :"Conv. config ID",
                                   "color"                    :"Min Validation Loss",
                                   "params_optimizer"         :"Optimizer",
                                  },
                         color  = study_df["value"],
                         symbol = study_df["params_optimizer"],
                         hover_name = "name", 
                         log_y  = True,
                        )
        fig.update_traces(marker={'size': 8})
        self._handle_image(fig, show, name, save)
        
        return
    
    @_plot_dict_member("parallel_plots")
    def parallel_plots(self, parallel_sets=[], show="0", name="parallel", save=False, data_dict=None):
        
        if data_dict is not None:
            parallel_sets = data_dict["parallel"]
        for conf in parallel_sets:        
            # build suffix 
            suffix = "_" + conf[0]   # first is the suffix for the filename
            
            fig = ov.plot_parallel_coordinate(self.study, params=conf[1:])
            name = name + suffix
            self._handle_image(fig, show, name, save)

        return
    
    @_plot_dict_member("contour_plots")
    def contour_plots(self, contour_sets=[], show="0", name="contour", save=False, data_dict=None):
        
        if data_dict is not None:
            contour_sets = data_dict["contour"]
        for conf in contour_sets:        
            # build suffix based on passed parameters
            suffix = "_".join(conf)
            
            fig = ov.plot_contour(self.study, params=conf)
            name = name + suffix
            self._handle_image(fig, show, name, save)

        return
    
    @_plot_dict_member("slice_plots")
    def slice_plots(self, slice_sets=[], show="0", name="slice", save=False, data_dict=None):
        
        if data_dict is not None:
            slice_sets = data_dict["slice"]
        for conf in slice_sets:        
            # build suffix based on passed parameters
            suffix = "_".join(conf)
            
            fig = ov.plot_slice(self.study, params=conf)
            name = name + suffix
            self._handle_image(fig, show, name, save)

        return
   
   

##### training tools #####-----------------------------------------------------------------------------

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

