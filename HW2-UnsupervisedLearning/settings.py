import torch
import os


##### GENERAL GLOBAL SETTINGS ##### -------------------------------------------------------------------------

MAGIC_NUM = 23   # seed for random state

# device
if torch.cuda.is_available():
    print('GPU available')
    USE_GPU = True
else:
    print('GPU not available')
    print("Available CPU cores:", os.cpu_count())
    USE_GPU = False

DATASETS_DIR   = 'Datasets'    # folder to contain datasets
CHECKPOINT_DIR = 'Checkpoints' # folder for models checkpoints
RESULTS_DIR    = 'Results'     # root folder for plots, ...


##### AUTOENCODER ##### -------------------------------------------------------------------------------------

class autoencoder():
    ROOT_DIR             = RESULTS_DIR + "/Autoencoder"
    OPTUNA_DIR           = ROOT_DIR + "/OptunaStudy"
    RECONSTRUCTIONS_DIR  = ROOT_DIR + "/ReconstructionExamples"  # directory to store reconstruction of images 
    BEST_HYPERS_FILE     = ROOT_DIR + "/best_hypers.json"        # file where to store the best hyper-parameters obtained with optuna 
    BEST_MODEL_CKPT_FILE = ROOT_DIR + "/best_model.ckpt"         # file where to store the best checkpoint model
    
    OPTUNA_STUDY_NAME    = "HP_search_autoencoder" 
    

##### TRANSFER LEARNING ##### -------------------------------------------------------------------------------

class transfer_learning():
    ROOT_DIR             = RESULTS_DIR + "/TransferLearning"
    BEST_MODEL_CKPT_FILE = ROOT_DIR + "/best_model.ckpt"         # file where to store the best checkpoint model
    

##### DENOISING AUTOENCODER ##### ---------------------------------------------------------------------------

class denoisingAE():
    ROOT_DIR             = RESULTS_DIR + "/DenoisingAutoencoder"
    RECONSTRUCTIONS_DIR  = ROOT_DIR + "/ReconstructionExamples"  # directory to store reconstruction of images 
    BEST_MODEL_CKPT_FILE = ROOT_DIR + "/best_model.ckpt"         # file where to store the best checkpoint model
    

##### VARIATIONAL AUTOENCODER ##### -------------------------------------------------------------------------

class variationalAE():
    ROOT_DIR             = RESULTS_DIR + "/VariationalAutoencoder"
    RECONSTRUCTIONS_DIR  = ROOT_DIR + "/ReconstructionExamples"  # directory to store reconstruction of images  
    BEST_MODEL_CKPT_FILE = ROOT_DIR + "/best_model.ckpt"         # file where to store the best checkpoint model
    

