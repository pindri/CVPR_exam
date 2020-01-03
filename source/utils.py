# Define context manager to measure execution time
# and handy function to plot confusion matrices.

import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

class codeTimer:
    """
    Context manager, measures and prints the execution time of a function.

    Parameters
    ----------
    name : str
        Name the user assings to the procedure the context manager is timing.

    """
    
    def __init__(self, name = None):
        self.name = "Executed '"  + name + "'. " if name else ""

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        self.elapsed = (self.end - self.start)
        print('%s Elapsed time: %0.6fs' % (str(self.name), self.elapsed))
        
        

def plot_confusion_matrix(true_labels, predicted_labels,
                          title = "Confusion matrix", cmap = plt.cm.gray_r,
                          filename = None):
    """Utility function to plot a confusion matrix

    Parameters
    ----------
    true_labels : list
        List of strings containing the true labels.
    predicted_labels : list
        List of strings containing the predicted labels.
    title : str
        Title of the plot.
    cmap
        Colormap of the plot.
    filename
        Filename with which the plot will be stored on disk

    """
    
    labels = np.unique(true_labels)
    
    # Compute and normalise confusion matrix.
    conf_matrix = confusion_matrix(true_labels, predicted_labels,
                                   labels = labels)
    conf_matrix = conf_matrix / np.sum(conf_matrix, axis = 1)
    
    # Divide canvas using gridspec.
    fig, ax = plt.subplots()
    
    # Confusion matrix.
    im = ax.imshow(conf_matrix, cmap = cmap, vmin = 0, vmax = 1)
    
    # Colorbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad = 0.3)
    plt.colorbar(im, cax = cax)
    
    # Ranges.
    ax.set_xlim(-0.5, len(labels)-0.5)
    ax.set_ylim(-0.5, len(labels)-0.5)

    # Ticks.
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start+0.5, end, 1))
    ax.yaxis.set_ticks(np.arange(start+0.5, end, 1))

    # Labels.
    ax.set_xticklabels([''] + labels, rotation = 90)
    ax.set_yticklabels([''] + labels)
    
    # Axis and titles.
    ax.set_xlabel("Predicted", fontweight = 'bold')
    ax.set_ylabel("True", fontweight = 'bold')
    ax.set_title(title, fontweight = 'bold')
    
    print("Accuracy: {}".format(np.sum(np.diag(conf_matrix)) /
                                       np.sum(conf_matrix)))

    # Save and show.
    if filename:
        fig.savefig(filename, bbox_inches='tight')
    plt.show()
    
