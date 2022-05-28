# python imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# pytorch imports
import torchmetrics



def plot_history(train, valid, figsize=(8,5), ylog=True, save_path=None):
    """
    Function that plot and save the training history with train and validation losses.
    """
    tr_epochs  = len(train)
    val_epochs = len(valid)
    
    ep_ratio = val_epochs // tr_epochs    # ep_ratio is supposed to be always >= 1
    left_rm  = int(ep_ratio - 1)
    ep_mod   = val_epochs % tr_epochs
    right_rm = val_epochs - ep_mod

    epochs     = np.arange(tr_epochs)
    val_epochs = (np.arange(val_epochs)-left_rm) / ep_ratio
    
    # plot of the losses
    fig = plt.figure(figsize=figsize)

    plt.title("Losses history")
    plt.xlabel("Train epoch")
    plt.ylabel("Loss value")
    if ylog:
        plt.yscale("log")
    
    plt.plot(epochs, train, "-o", label="train loss")
    plt.plot(val_epochs[left_rm:right_rm], valid[left_rm:right_rm], "-o", label="valid. loss")

    # plot horizontal line at minimum validation loss
    min_valid = min(valid)
    plt.plot([0, tr_epochs-1], [min_valid, min_valid], color="red", 
             ls="--", label="Minimum valid. loss: %.3f"%min_valid,
            )

    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return 
        
    
    
def plot_img_grid(grid_shape, images, titles=None, folder_path=None, filename="plot_img_grid_test.pdf", 
                  to_show=False, figsize=(12,6), suptitle=None, cmap="Greys", axis_off=True,
                 ):
    """
    Function that plots a grid of images with configurable shape.
    """
    rows = grid_shape[0]
    cols = grid_shape[1]
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)
    
    for idx in range(rows):
        # check if single row plot
        ax_row = axs[idx] if rows > 1 else axs
        for jdx in range(cols):
            # check if single column plot 
            ax = ax_row[jdx] if cols > 1 else ax_row
            
            img = images[jdx + cols*idx]            
            ax.imshow(img.cpu().squeeze().numpy(), cmap=cmap)
            if titles is not None:
                ax.set_title(titles[jdx + cols*idx])
            if axis_off:
                ax.axis("off")
    
    plt.tight_layout()
    
    # save figure
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)
        fig.savefig( folder_path + "/" + filename )
    
    # plot figure
    if to_show:
        plt.show()
    plt.close()
    
    return fig



def plot_confusion_matrix(guesses, true_labels, label_names, figsize=(8,5), save_path=None):
    """
    Function that produce and plot a confusion matrix from the true labels and model predictions.
    """
    num_classes = len(label_names)
    
    # bulding confusion matrix
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes)
    mat = confusion_matrix(guesses, true_labels).numpy()

    # plotting    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat)

    # Show all ticks and label them with the respective names
    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels(label_names)
    
    ax.set_yticks(np.arange(num_classes))
    ax.set_yticklabels(label_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    formatter = matplotlib.ticker.StrMethodFormatter("{x:d}")
    
    for i in range(num_classes):
        for j in range(num_classes):
            datum = mat[i, j].astype(int)
            color = "white" if (datum < np.amax(mat)*0.5) else "black"
            text = ax.text(j, i, formatter(datum), fontsize=12,
                           ha="center", va="center", color=color)

    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    fig.tight_layout()
    plt.show()
            
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return



