import matplotlib.pyplot as plt
import os
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
from pytorch_lightning.callbacks import Callback

from utilities import plot_tools


#DA SISTEMARE


class ImageReconstruction(Callback):
    """
    Callback to plot (and save) a comparison between original and reconstructed image during training
    """
    def __init__(self, original, figsize = (12,6), to_show=False, save_path=None):
        self.original = original   #original image 
        self.figsize  = figsize
        
        if (save_path is None) and (to_show == False):
            raise RuntimeError("Please provide a valid path where to save images or "+
                                "set 'to_show' to True."
                              )
        self.to_show = to_show
        self.save_path = save_path
        
        
    def on_validation_epoch_end(self, trainer, module): 
        
        # ensure that model is in eval mode
        assert not module.training
        with torch.no_grad():
            reconstructed = module(self.original)
            
        epoch = trainer.current_epoch
        
        images = [self.original, reconstructed]
        titles = ['Original image', 'Reconstructed image (EPOCH %d)'%(epoch + 1)]
        filename = "reconstruction_example_epoch_%d.pdf"%(epoch + 1)
        
        # create, plot, save the image
        plot_tools.plot_img_grid(grid_shape=(1,2), images=images, titles=titles, to_show=self.to_show,
                                 folder_path=self.save_path, filename=filename, figsize=self.figsize,
                                )
        
        return
        
        

class EncodedRepresentation(Callback):  ### DANGER rimuovila e usa trainer.predict (vedi google...) DANGER
    """ 
    Callback that stores a list of encoded representation of a dataset when trainer.test is called.
    """
    def __init__(self):
        self.encoded_samples = []
        self.labels          = []
        
    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx):
    
        orig, labels = batch
        encoded = module.encoder(orig)
        
        for sample,label in zip(encoded, labels):
            self.encoded_samples.append(sample.numpy())
            self.labels.append(label.item())
        
        return
    
    
    
class LatentSpaceAnalyzer(object):
    """BUG BUG """
    def __init__(self, encoded_samples, labels, label_names=None, 
                 save_path="Results",
                ):
        
        self.encoded_samples = encoded_samples
        self.labels          = labels
        if label_names is None:
            self.label_names = list(labels)
        else:
            self.label_names = label_names
        
        self.save_path = save_path
        
        
    def PCA_reduce(self, n_components=2, to_show=True, filename=None):
        
        pca = PCA(n_components=n_components)
        self.PCA_reduced_samples = pca.fit_transform(self.encoded_samples)
        
        fig = self._plot_subspace(self.PCA_reduced_samples)
        
        if to_show:
            fig.show()
            
        if filename is not None:
            path = self.save_path +'/' + filename
            fig.write_image(path) 
        
        return
        
        
    def TSNE_reduce(self, n_components=2, perplexity=30, to_show=True, filename=None):
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        self.TSNE_reduced_samples = tsne.fit_transform(self.encoded_samples)
        
        fig = self._plot_subspace(self.TSNE_reduced_samples)
        
        if to_show:
            fig.show()
            
        if filename is not None:
            path = self.save_path +'/' + filename
            fig.write_image(path) 
        
        return
    
        
    def _plot_subspace(self, subspace):
                
        # produce image
        fig = px.scatter(subspace, 
                         x=0, y=1, 
                         labels = {"0":"First Dim",
                                   "1":"Second Dim",
                                   "color":"Label",
                                  },
                         color=self.label_names, 
                         opacity=0.7,
                        )
        return fig
        
        


def generate_from_latent_code(encoded_sample, model):
    """Generate image from latent space representation."""
    latent_code = torch.tensor(encoded_sample).float().unsqueeze(dim=0)
    
    model.eval()
    with torch.no_grad():
        generated = model.decoder(latent_code)
    
    return generated


