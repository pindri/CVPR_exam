# Implements different classifiers for image classification, along with
# their auxiliary functions to compute distances.

from scipy.stats import wasserstein_distance
import numpy as np
import numpy.linalg as lin
import pandas as pd
import cv2
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


# Distances.


def earth_mover_distance(hist_1, hist_2):
    """Implements the Earth Mover Distance

    Parameters
    ----------
    hist_1 : np.array
        First histogram as array of bins.
    hist_2 : np.array
        Second histogram as array of bins.

    Returns
    -------
    fload
        Wasserstein distance (EMD) between the histograms.

    """
    
    return wasserstein_distance(hist_1, hist_2)
    


def chi_2_distance(hist_1, hist_2):
    """Implements the Chi Squared distance.

    Note
    ----
    It does not follow the cv2 formula which is not symmetric.

    Parameters
    ----------
    hist_1 : np.array
        First histogram as array of bins.
    hist_2 : np.array
        Second histogram as array of bins.

    Returns
    -------
    fload
        Chi Squared distance between the histograms.

    """

    n_bins = len(hist_1)
    
    # Mask for non-zero entries in at least one of the histograms.
    non_zero = [index for index in range(n_bins)
                if (hist_1[index] != 0 and hist_2[index] != 0)]
    
    distance = 0.5 * np.sum((hist_1[non_zero] - hist_2[non_zero])**2 /
                            (hist_1[non_zero] + hist_2[non_zero]))
    
    return distance



def hamming_distance(arr_1, arr_2):
    """Computes the Hamming distance between two bit strings.

    Parameters
    ---------
    arr_1 : str
        First bit string.
    arr_2 : str
        Second bit string.

    Returns
    -------
    fload
        Hamming distance between the strings.

    """

    assert (len(arr_1) == len(arr_2)), "Arrays must have the same length."
    
    return sum(c1 != c2 for c1, c2 in zip(arr_1, arr_2))



# Classifiers.


def nn_classifier(train_df, test_df):
    """Nearest neighbour classifier.

    Parameters
    ----------
    train_df
        Dataframe containing the training data.
    test_df
        Dataframe containing the test data.

    Returns
    -------
    list, list
        List of true labels and list of predicted ones.

    """

    true_labels = test_df['label'].to_numpy()
    predicted_labels = []
    
    # For each test image.
    for test_index, test_row in test_df.iterrows():
        distances = []
        # Compute distance with each train image.
        for train_index, train_row in train_df.iterrows():
            distances.append(wasserstein_distance(train_row['histogram'],
                                                  test_row['histogram']))
        # The predicted label corresponds to the minimum distance.
        predicted_labels.append(train_df.iloc[np.argmin(distances)]['label'])
        
    return true_labels, np.array(predicted_labels)



def linear_SVM_classifier(train_df, test_df):
    """Multi-class Support Vector Machine classifier using a linear kernel.

    Implements a SVM for multi-class classification using the one-vs-rest
    approach, training a classifier for each class. For each image,
    a prediction is made with all the classifiers: the class corresponding
    to the largest hyperplane distance is the predicted one.

    Parameters
    ----------
    train_df
        Dataframe containing the training data.
    test_df
        Dataframe containing the test data.

    Returns
    -------
    list, list
        List of true labels and list of predicted ones.

    """
    
    true_labels = test_df['label'].to_numpy()
    predicted_labels = []
    
    # Training multiclass SVM, one-vs-rest approach.
    
    # Extract labels.
    train_labels = train_df['label'].to_numpy()
    classes = train_df['label'].unique()

    # Extract histograms, (n_samples, n_feature) numpy array.
    histo = np.asarray(train_df['histogram'].tolist())

    # Build a classifier for each class.
    clf = [SVC(kernel = 'linear') for _ in range(len(classes))]

    # Fit each of the classifiers.
    for idx, c in enumerate(clf):
        current_label = classes[idx]
        # 1 for current_label occurrences, -1 elsewhere.
        target = np.array(["1" if  label == current_label else
                           "-1" for label in train_labels])
        c = c.fit(histo, target)

    # Predicting label for each test image.
    for test_index, test_row in test_df.iterrows():
        distances = []
        # Compute real valued score with each classifier.
        for idx, c in enumerate(clf):
            # Compute distances from hyperplanes
            distances.append(c.decision_function(test_row['histogram']
                                                 .reshape(1, -1) / 
                                                 lin.norm(c.coef_)))
        # The predicted label: maximum distance from hyperplane.
        predicted_labels.append(classes[np.argmax(distances)])
    
    return true_labels, np.array(predicted_labels)



def gaussian_kernel(hist_1, hist_2, dist_function, sigma = 0.1):
    """Implements the computation of Gaussian kernel.

    Parameters
    ----------
    hist_1 : np.array
        First histogram as array of bins.
    hist_2 : np.array
        Second histogram as array of bins.
    dist_function
        Function to compute distances between histograms.
    sigma
        Sigma value for the Gaussian kernel.

    Returns
    -------
    float
        Value of the kernel for the two histograms.

    """

    distance = dist_function(hist_1, hist_2)

    return np.exp( - distance / (2 * sigma ** 2 ))



def gram_matrix(histograms_1, histograms_2,
                kernel_function = gaussian_kernel,
                dist_function = chi_2_distance):
    """Computes the Gram matrix for two sets of histograms.

    Parameters
    ----------
    histograms_1
        First list of histograms.
    histograms_2
        Second list of histograms.
    kernel_function
        Function that implements the computation of a kernel for a given
        distance.
    dist_function
        Function to compute distances between histograms.

    Returns
    -------
    np.array
        A 2D numpy array corresponding to the Gram matrix.

    """
    
    gram = np.zeros((histograms_1.shape[0], histograms_2.shape[0]))
    
    for i, h_1 in enumerate(histograms_1):
        for j, h_2 in enumerate(histograms_2):
            elem = kernel_function(h_1, h_2, dist_function)
            gram[i, j] = elem
    
    return gram



def gaussian_SVM_classifier(train_df, test_df,
                            kernel_function = gaussian_kernel,
                            dist = 'chi'):
    """Multi-class Support Vector Machine classifier using a Gaussian kernel.

    Implements a SVM for multi-class classification using the one-vs-rest
    approach, training a classifier for each class. For each image,
    a prediction is made with all the classifiers: the class corresponding
    to the largest hyperplane distance is the predicted one.

    Parameters
    ----------
    train_df
        Dataframe containing the training data.
    test_df
        Dataframe containing the test data.

    Returns
    -------
    list, list
        List of true labels and list of predicted ones.

    """
    
    if dist == 'chi':
        dist_function = chi_2_distance
    elif dist == 'emd':
        dist_function = earth_mover_distance
    else:
        raise ValueError("Unvalid distance. Use 'chi' or 'emd'")
    
    true_labels = test_df['label'].to_numpy()
    predicted_labels = []
    
    # Training multiclass SVM, one-vs-rest approach.
    
    # Extract labels and histograms.
    train_labels = train_df['label'].to_numpy()
    classes = train_df['label'].unique()
    train_histograms = train_df['histogram']
    
    # Build classifier.
    gram = gram_matrix(train_histograms, train_histograms,
                       kernel_function, dist_function)
    clf = OneVsRestClassifier(SVC(kernel="precomputed"), n_jobs=-1)
    clf.fit(gram, train_labels)
    
    # Predicting label for each test image.
    test_histograms = test_df['histogram']
    prediction_gram = gram_matrix(test_histograms, train_histograms,
                                  kernel_function, dist_function)
    predicted_labels = clf.predict(prediction_gram)
    
    return true_labels, np.array(predicted_labels)



def ecoc_classifier(train_df, test_df, n_classifiers = 25):
    """Multi-class Support Vector Machine classifier using the Error
    Correcting Output Codes approach.

    Learns several binary classifiers following the ECOC approach,
    representing each class as a bit string.
    For a test image, computes the value of all the classifiers and
    stores the result as a bit string: using the Hamming distance,
    the label of the closest class is used as a prediction.

    Parameters
    ----------
    train_df
        Dataframe containing the training data.
    test_df
        Dataframe containing the test data.
    n_classifiers : int
        Number of classifiers for the ECOC approach.

    Returns
    -------
    list, list
        List of true labels and list of predicted ones.

    """
    
    true_labels = test_df['label'].to_numpy()
    predicted_labels = []
    
    # Extract labels.
    train_labels = train_df['label'].to_numpy()
    classes = train_df['label'].unique()

    # Extract histograms, (n_samples, n_feature) numpy array.
    histo = np.asarray(train_df['histogram'].tolist())

    # Build randomly generated coding matrix and the classifiers.
    coding_matrix = np.random.choice([0, 1],
                                     size = (len(classes), n_classifiers))
    clf = [SVC() for _ in range(n_classifiers)]
    
    
    # Fit each of the classifiers.
    for idx, c in enumerate(clf):
        
        # class-code correspondences.
        codes = coding_matrix[:, idx]
        current_labels = classes[codes == 1]
        
        # 1 for current_labels occurrences, -1 elsewhere.
        target = np.array(["1" if  label in current_labels else
                           "0" for label in train_labels])
        c = c.fit(histo, target)
        
        
    # Make predictions.
    for test_index, test_row in test_df.iterrows():
        predicted_code = []
        for idx, c in enumerate(clf):
            # Predict code.
            predicted_code.append(int(c.predict(test_row['histogram']
                                                .reshape(1, -1))))
        
        # Compute distance with coding matrix rows.
        distances = []
        for row in coding_matrix:
            distances.append(hamming_distance(row, predicted_code))
                
        # The predicted label: minimum hamming distance.
        predicted_labels.append(classes[np.argmin(distances)])
            
            
    return true_labels, predicted_labels
