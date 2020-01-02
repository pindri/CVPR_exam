from scipy.stats import wasserstein_distance
import numpy as np
import numpy.linalg as lin
import pandas as pd
import cv2

def nn_classifier(train_df, test_df):
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



from sklearn.svm import SVC



def linear_SVM_classifier(train_df, test_df):
    
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


def chi_2_distance(hist_1, hist_2):
    # Note, not using the cv2 formula since it is not really a distance.
    n_bins = len(hist_1)
    
    non_zero = [index for index in range(n_bins)
                if (hist_1[index] != 0 and hist_2[index] != 0)]
    
    distance = 0.5 * np.sum((hist_1[non_zero] - hist_2[non_zero])**2 /
                            (hist_1[non_zero] + hist_2[non_zero]))
    
    return distance


def earth_mover_distance(hist_1, hist_2):
    
    return wasserstein_distance(hist_1, hist_2)
    


def gaussian_kernel(hist_1, hist_2, dist_function, sigma = 0.1):
    distance = dist_function(hist_1, hist_2)
    return np.exp( - distance / (2 * sigma ** 2 ))


def gram_matrix(histograms_1, histograms_2,
                kernel_function = gaussian_kernel,
                dist_function = chi_2_distance):
    
    gram = np.zeros((histograms_1.shape[0], histograms_2.shape[0]))
    
    for i, h_1 in enumerate(histograms_1):
        for j, h_2 in enumerate(histograms_2):
            elem = kernel_function(h_1, h_2, dist_function)
            gram[i, j] = elem
    
    return gram


from sklearn.multiclass import OneVsRestClassifier

def gaussian_SVM_classifier(train_df, test_df,
                            kernel_function = gaussian_kernel,
                            dist = 'chi'):
    
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



def hamming_distance(arr_1, arr_2):
    """Calculate the Hamming distance between two bit strings"""
    assert (len(arr_1) == len(arr_2)), "Arrays must have the same length."
    return sum(c1 != c2 for c1, c2 in zip(arr_1, arr_2))


def ecoc_classifier(train_df, test_df, n_classifiers = 25):
    """
    TBD
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