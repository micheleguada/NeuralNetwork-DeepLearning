import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

import torch
from pytorch_lightning.callbacks import Callback

from utilities import plot_tools




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
        
        

class EncodedRepresentation(Callback):
    """ 
    Callback that stores a list of encoded representation of a dataset when trainer.test is called.
    """
    def __init__(self):
        self.encoded_samples = []
        self.labels          = []
        
    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx):
    
        orig, labels = batch
        
        # ensure that model is in eval mode
        assert not module.training
        with torch.no_grad():
            encoded = module.get_latent_representation(orig)
        
        for sample,label in zip(encoded, labels):
            self.encoded_samples.append(sample.cpu().numpy())
            self.labels.append(label.item())
        
        return
    
    
    
class LatentSpaceAnalyzer(object):
    """
    Class that implements some methods to visualize the latent representation of an autoencoder model. 
    Supported reduction method are: PCA, TSNE, Isomap.
    """
    def __init__(self, encoded_samples, labels, label_names=None, 
                 save_path="Results",
                ):
        
        self.encoded_samples = encoded_samples
        self.labels          = labels
        if label_names is None:
            self.label_names = sorted(list(set(labels)))
        else:
            self.label_names = label_names
        self.string_labels = [str(label_names[ii]) for ii in labels]
        
        self.save_path = save_path
        
        self.pca  = None
        self.tsne = None
        self.isomap = None
        
        
    def PCA_reduce(self, n_components=2, to_show=True, filename=None):
        
        self.pca = PCA(n_components=n_components)
        self.PCA_reduced_samples = self.pca.fit_transform(self.encoded_samples)
        
        fig = self._plot_subspace(self.PCA_reduced_samples, self.string_labels, to_show, filename)
        
        return
        
        
    def TSNE_reduce(self, n_components=2, perplexity=30, to_show=True, filename=None):
        
        self.tsne = TSNE(n_components=n_components, perplexity=perplexity, n_jobs=-1)
        self.TSNE_reduced_samples = self.tsne.fit_transform(self.encoded_samples)
        
        fig = self._plot_subspace(self.TSNE_reduced_samples, self.string_labels, to_show, filename)
        
        return
    
    def Isomap_reduce(self, n_components=2, to_show=True, filename=None):
        
        self.isomap = Isomap(n_components=n_components, n_jobs=-1)
        self.Isomap_reduced_samples = self.isomap.fit_transform(self.encoded_samples)
        
        fig = self._plot_subspace(self.Isomap_reduced_samples, self.string_labels, to_show, filename)
        
        return
    
        
    def _plot_subspace(self, subspace, labels, to_show, filename):
                
        # produce image
        fig = px.scatter(subspace, 
                         x=0, y=1, 
                         labels = {"0":"First Dim",
                                   "1":"Second Dim",
                                   "color":"Label",
                                  },
                         color=labels,
                         opacity=0.7,
                        )
        if to_show:
            fig.show()
            
        if filename is not None:
            path = self.save_path +'/' + filename
            fig.write_image(path) 
        
        return fig
    
    
    def PCA_overlap_points(self, points, to_show=True, filename=None):
        """
        Function to overlap some latent space samples on the visualization obtained with PCA.
        """
        if self.pca is None:
            raise RuntimeError("Run 'PCA_reduce' before calling 'overlap_points'.")
        transformed = self.pca.transform(points)
        
        # produce image
        fig = self._plot_subspace(self.PCA_reduced_samples, self.string_labels, to_show=False, filename=None)
        fig.add_trace(go.Scatter(x=transformed[:,0], 
                                 y=transformed[:,1], 
                                 mode = 'markers',
                                 marker_symbol = 'star',
                                 marker_size = 15,
                                 marker_color = "black",
                                 showlegend = False,
                                 name = "Generated",
                     )          )
        if to_show:
            fig.show()
            
        if filename is not None:
            path = self.save_path +'/' + filename
            fig.write_image(path) 
        
        return
        
        

