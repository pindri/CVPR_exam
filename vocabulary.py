# Define functions to read images from a given path,
# compute the SURF descriptors and build a visual vocabulary,
# representing each image as an histogram over the visual words. 

import cv2
import os
import numpy as np
import numpy.linalg as lin
from sklearn.cluster import KMeans
import pandas as pd


def read_image(dir_path):
    """Generator function for reading image from disk.

    Parameters
    ----------
    dir_path : str
        Directory path.

    Yields
    ------
    str, str
        Image path and directory name.
    
    """
    
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            # Ignore useless files.
            if file != ".DS_Store":
                yield (os.path.join(subdir, file),
                       os.path.basename(subdir))
                

                
def compute_descriptors(dir_path):
    """Computes SURF descriptors for images and stores them in a Pandas
    Dataframe.

    The images should be stored in folders representing their labels. The
    parameter should be the path of the directory containing such folders.

    Parameters
    ----------
    dir_path : str
        Path where the images are stored.

    Returns
    -------
    Pandas Dataframe
        Contains the images' ID, their label and SURF descriptors.

    """
    
    # Lists to contain data.
    image_ids = []
    descriptors = []
    labels = []
    
    # Build SIFT object.
    orb = cv2.xfeatures2d.SIFT_create()
    
    image_id = 0
    
    for image, label in read_image(dir_path):
        # Detect and compute keypoints and descriptors.
        kp, des = orb.detectAndCompute(cv2.imread(image), None)
        # If no keypoints detected, continue.
        if kp == []:
            continue
        for d in range(len(des)):
            descriptors.append(des[d])
            image_ids.append(image_id)
            labels.append(label)
            
        image_id += 1
            
    des_df = pd.DataFrame(columns = ['image_id', 'descriptor', 'label'])
    des_df['image_id'] = image_ids
    des_df['descriptor'] = descriptors
    des_df['label'] = labels
    
    return des_df



def k_means_words(descriptors_df, n_clusters, num_descriptors):
    """Performs K-means clustering on a set of descriptors.

    Parameters
    ----------
    descriptors_df
        Pandas Dataframe containing descriptors.
    n_clusters : int
        Number of desired clusters.
    num_descriptors : int
        Number of descriptors.

    Returns
    -------
    KMeans object
        Contains the info on the cluster, along with its centroids.

    """
    
    # Sample `num_descriptors` descriptors.
    data = descriptors_df['descriptor'].sample(n = num_descriptors).tolist()
    print("Number of sampled descriptors: {}".format(len(data)))
    # Compute the kmeans algorithm on the sampled descriptors.
    kmeans = KMeans(n_clusters = n_clusters, n_jobs = -1).fit(data)
    print("Clustered the descriptors in {} clusters"
          .format(kmeans.n_clusters))
    
    return kmeans



def find_nearest_term(descriptor, dictionary):
    """Computes the nearest dictionary term for a given descriptor.

    Parameters
    ----------
    descriptor
        Multidimensional point representing the descriptor.
    dictionary
        Set of multidimensional words, representing a descriptor vocabulary.

    Returns
    -------
    int
        Index of the closest dictionary term.

    """
    distances = [lin.norm(descriptor - term) for term in dictionary]
    
    return np.argmin(distances)



def compute_histogram(descriptors_df, kmeans):
    """Computes the visual words histograms for the provided dataset.

    Extracts visual dictionary from a KMeans object and uses them as a 
    dictionary. For each image, a histogram over the dictionary is built:
    for each descriptor, the value of the closest dictionary term bin is
    increased by one.

    Parameters
    ----------
    descriptors_df
        Dataframe containing descriptors.
    kmeans
        KMeans object containing dictionary words.

    Returns
    -------
    list
        List of arrays, each array is a histogram.

    """

    # Estract visual words (centroids) from kmeans computation.
    dictionary = kmeans.cluster_centers_
    
    num_images = len(descriptors_df['image_id'])
    histograms = np.zeros((num_images, len(dictionary)))
    
    # For every image.
    for index, row in descriptors_df.iterrows():
        # For every descriptor.
        for descriptor in row['descriptor']:
            closest_centroid = find_nearest_term(descriptor, dictionary)
            histograms[index][closest_centroid] += 1
            
    # Compute norms and normalisation.
    norm = np.sum(histograms, axis = 1).reshape(num_images, 1)
    histograms = histograms / norm
    
    # More convenient format as list of arrays.
    histograms = list(histograms[row] for row in range(len(histograms)))
    
    return histograms
