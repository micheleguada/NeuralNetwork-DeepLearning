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



##### Optuna optimization tools #####------------------------------------------------------------------------
class MinValidationLoss(Callback):
    """
    Class that store the minimum reached validation loss.
    """
    def __init__(self, val_name="val_loss"): 
        self.val_name = val_name
        self.min_valid_loss = -1.
    
    def on_validation_epoch_end(self, trainer, module):  
        # get current validation loss
        current_val = trainer.logged_metrics[self.val_name].cpu().item()
        if ( (self.min_valid_loss >= current_val) or (self.min_valid_loss < 0.) ):
            self.min_valid_loss = current_val
            
        return


class Objective(object):
    """ Objective class for the optuna study optimization to be used with PyTorch-Lightning. """
    def __init__(self, model_class, datamodule, hp_space, max_epochs=50, min_delta=0.001,
                 early_stop_patience=5, use_gpu=False, auto_lr_find=False,
                ):  
        self.model_class = model_class
        self.datamodule  = datamodule
        self.hp_space    = hp_space     # object of class: 'HyperparameterSpace'
            
        self.max_epochs  = max_epochs
        self.min_delta   = min_delta
        self.patience    = early_stop_patience
        self.use_gpu     = use_gpu
        self.auto_lr_find = auto_lr_find
        
    def __call__(self, trial):
        
        print(f"Trial [{trial.number}] started at:", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        ### sample hyperparameters
        params, optimizer, learning_rate, L2_penalty, momentum = self.hp_space.sample_configuration(trial, 
                                                                                                    self.auto_lr_find,
                                                                                                    self.use_gpu,
                                                                                                    self.datamodule,
                                                                                                   )
        ### create model
        model = self.model_class(input_size = self.datamodule.get_sample_size(),
                                 params     = params,
                                 optimizer     = optimizer,
                                 learning_rate = learning_rate,
                                 L2_penalty    = L2_penalty,
                                 momentum      = momentum,
                                )
        
        ### create trainer object
        min_valid_callback = MinValidationLoss()
        early_stop = EarlyStopping(monitor   = "val_loss", 
                                   min_delta = self.min_delta, 
                                   patience  = self.patience,
                                   verbose   = False, 
                                   check_on_train_epoch_end=False, # check early_stop at end of validation
                                  )
        pl_pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        
        trainer = pl.Trainer(logger     = False,
                             max_epochs = self.max_epochs,
                             gpus       = 1 if self.use_gpu else None,  #trainer will take care of moving model and datamodule to GPU
                             callbacks  = [pl_pruning, early_stop, min_valid_callback],
                             enable_checkpointing = False,
                             enable_model_summary = False,
                             deterministic  = True,     #use deterministic algorithms for reproducibility
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
        min_valid_loss = min_valid_callback.min_valid_loss

        print(f"Trial [{trial.number}] ended at:", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        print(f"    Valid. loss: {final_valid_loss}; min valid. loss: {min_valid_loss}\n")
        
        return min_valid_loss     
    
    
# decorator to add a function to a dictionary
def make_decorator(dictionary):
    def decorator_add_to_dict():
        def wrapper(func):
            dictionary.update({func.__name__:func})
            return func
        return wrapper
    return decorator_add_to_dict
    
class OptimizationInspector(object):
    """
    This class provides some plotting functions to analyze the outcome of an optuna study.
    """
    # dictionary of plotting function
    plot_dict = {}
    _plot_dict_member = make_decorator(plot_dict)
    
    def __init__(self, study, save_path="Results_test", figsize=(1024,600), fmt=".pdf"):
        
        self.study     = study
        self.save_path = save_path + "/"
        self.data_dict = None
        self.fmt       = fmt
        
        # set figsize
        px.defaults.width  = figsize[0]
        px.defaults.height = figsize[1]
        
        # ensure folder existence
        os.makedirs(self.save_path, exist_ok=True)
        
    #def print_help(self):
        ## print dict of available plot functions
        #print("Available plot methods:")
        #for method in list(self.plot_dict):
            #print("   ", method)
            
        #print("Use the method 'plot_all' to run all the listed functions.")
        #return
        
    def save_best_hypers_json(self, best_hypers_file):
        # save best hyperparameters to file (json)
        best_hypers = self.study.best_trial.user_attrs["hypers"]
        with open(best_hypers_file, 'w') as fp:
            json.dump(best_hypers, fp)
        print("Best hyper-parameters saved to: '"+best_hypers_file+"'.")
    
    def _handle_image(self, fig, show, name, save):
        # remove eventual title
        fig.update_layout(title="",
                          margin=dict(l=40, r=20, t=20, b=40),
                         )
        
        # function to plot/save images
        if show == "1":
            fig.show()
        if (save) and (name is not None):
            full_path = self.save_path + name + self.fmt
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
        print("\nBest set of hyper-parameters:")
        width = max([len(tt) for tt in list(best_hypers)]) # string width when printing param name 
        for key,var in best_hypers.items():
            if key == "params":
                ww = max([len(tt) for tt in list(var)])
                print("    Model parameters:")
                for kk,vv in var.items():
                    print(f"        {kk: <{ww}}: {vv}")
            else:
                print(f"    {key: <{width}}: {var}")
        print("")
        
        return
    
    
    def plot_all(self, parallel_sets = [], contour_sets = [], slice_sets = [], importance_params = [],
                 show = "100011000", save = True,
                ):
        """
        Produce all the defined plots in this class (by now 9). Showing is controlled by the variable 'show'.
        It can also save all the plotted pictures. Files names are fixed to some default value.
         - show : binary string of lenght 9 ('1' to show image, '0' to not show).
         - save : if to save the pictures on disk (bool)
        """
        self.data_dict = {"parallel"         : parallel_sets    ,
                          "contour"          : contour_sets     ,
                          "slice"            : slice_sets       ,
                          "importance_params": importance_params,
                         }
        
        for idx,key in enumerate(self.plot_dict):
            if (show[idx] == "0") and not save: #skip plots that are not showed or saved
                print("   Skipping plot function:", key)
                continue
            self.plot_dict[key](self, show=show[idx], save=save)
            
        self.data_dict = None
            
        return
    
    @_plot_dict_member() #1
    def optimization_history(self, show="1", name="optimization_history", save=False):
        fig = ov.plot_optimization_history(self.study)
        fig.update_yaxes(type="log")
        self._handle_image(fig, show, name, save)
        return
    
    @_plot_dict_member() #2
    def intermediate_values(self, show="1", name="intermediate_values", save=False):
        fig = ov.plot_intermediate_values(self.study)
        self._handle_image(fig, show, name, save)
        return
    
    @_plot_dict_member() #3
    def importances(self, params=None, show="1", name="importances", save=False):
        if self.data_dict is not None:
            params = self.data_dict["importance_params"]
        fig = ov.plot_param_importances(self.study, params=params)
        self._handle_image(fig, show, name, save)
        return
    
    @_plot_dict_member() #4
    def time_vs_value(self, show="1", name="time_vs_value", save=False):  
        
        study_df = self.study.trials_dataframe()
        
        # compute time in minutes and the name for the hover functionality
        study_df["time"] = study_df.apply(lambda row: row['duration'].total_seconds()/60, axis=1)
        study_df["name"] = study_df.apply(lambda row: "Trial "+str(row['number']), axis=1)

        # plot picture
        fig = px.scatter(study_df, 
                         x="time", y="value",
                         labels     = {"time":"Training Time [min]", "value":"Objective Value"},
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
        
    @_plot_dict_member() #5
    def latent_dim_vs_value(self, show="1", name="latent_dim_vs_value", save=False):

        study_df  = self.study.trials_dataframe()
        study_df["name"] = study_df.apply(lambda row: "Trial "+str(row['number']), axis=1)
        latent_df = study_df.sort_values(by="params_conv_config_id")

        fig = px.scatter(latent_df, 
                        x="params_latent_space_dim", y="value",
                        labels = {"params_latent_space_dim":"Latent space dimension",
                                  "value"                  :"Min Validation Loss",
                                  "color"                  :"Optimizer",
                                 },
                        color  = latent_df["params_optimizer"].astype(str),
                        color_discrete_sequence = px.colors.qualitative.Set1,
                        hover_name = "name", 
                        log_y  = True,
                        )
        fig.update_traces(marker={'size': 8})
        self._handle_image(fig, show, name, save)
        
        return
    
    @_plot_dict_member() #6
    def conv_vs_channels(self, show="1", name="conv_vs_channels", save=False):
        
        study_df  = self.study.trials_dataframe()
        study_df["name"] = study_df.apply(lambda row: "Trial "+str(row['number']), axis=1)

        fig = px.scatter(study_df, 
                         x="params_channels_config_id", y="params_conv_config_id",
                         labels = {"params_channels_config_id":"Channels config ID",
                                   "params_conv_config_id"    :"Conv. config ID",
                                   "value"                    :"Value",
                                   "params_optimizer"         :"Optimizer",
                                  },
                         color      = study_df["value"],
                         color_continuous_scale = px.colors.sequential.Viridis,
                         symbol     = study_df["params_optimizer"],
                         hover_name = "name", 
                        )
        fig.update_traces(marker={'size': 8})
        fig.update_layout(legend=dict(title       = "Optimizer:",
                                      orientation = "h",
                                      yanchor = "bottom",
                                      y=1.02,
                                      xanchor = "right",
                                      x=1,
                         )           )
        self._handle_image(fig, show, name, save)
        
        return
    
    @_plot_dict_member() #7
    def parallel_plots(self, parallel_sets=[], show="1", name="parallel", save=False):
        
        if self.data_dict is not None:
            parallel_sets = self.data_dict["parallel"]
        for conf in parallel_sets:        
            # build suffix 
            suffix = "_" + conf[0]   # first is the suffix for the filename
            
            fig = ov.plot_parallel_coordinate(self.study, params=conf[1:])
            self._handle_image(fig, show, name + suffix, save)

        return
    
    @_plot_dict_member() #8
    def contour_plots(self, contour_sets=[], show="1", name="contour", save=False):
        
        if self.data_dict is not None:
            contour_sets = self.data_dict["contour"]
        for conf in contour_sets:        
            # build suffix based on passed parameters
            suffix = "_" + "_".join(conf)
            
            fig = ov.plot_contour(self.study, params=conf)
            self._handle_image(fig, show, name + suffix, save)

        return
    
    @_plot_dict_member() #9
    def slice_plots(self, slice_sets=[], show="1", name="slice", save=False):
        
        if self.data_dict is not None:
            slice_sets = self.data_dict["slice"]
        for conf in slice_sets:        
            # build suffix based on passed parameters
            suffix = "_" + "_".join(conf)
            
            fig = ov.plot_slice(self.study, params=conf)
            self._handle_image(fig, show, name + suffix, save)

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
        self.train.append(trainer.logged_metrics[self.train_name].cpu())
        return
    
    def on_validation_epoch_end(self, trainer, module):       
        self.valid.append(trainer.logged_metrics[self.val_name].cpu())
        return


