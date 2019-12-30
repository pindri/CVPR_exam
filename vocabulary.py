# Define function to read images from a given path,
# compute the SURF descriptors and build a visual vocabulary,
# representing each image as an histogram over the visual words. 

import cv2
import os
import numpy as np
import numpy.linalg as lin
from sklearn.cluster import KMeans
import pandas as pd


# Read image and directory (image category)

def read_image(dir_path):
    """
    TBD iterator
    """
    
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            # Ignore useless files.
            if file != ".DS_Store":
                yield (os.path.join(subdir, file),
                       os.path.basename(subdir))
                

                
def compute_descriptors(dir_path):
    """
    TBD, note on ORB instead of SIFT/SURF
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
    """
    TBD
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
    """
    TBD
    """
    distances = [lin.norm(descriptor - term) for term in dictionary]
    
    return np.argmin(distances)



def compute_histogram(descriptors_df, kmeans):
    """
    TBD
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
