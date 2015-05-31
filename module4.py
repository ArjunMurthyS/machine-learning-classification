#!/usr/bin/env python

"""
Evaluate effectiveness of k-nearest neighbours classifier
"""

from __future__ import print_function
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import metrics
import wdbc

def test_nearest_neighbors(feature_data, classification_data_numerical):
    """
    Demonstrate simple use of a nearest neighbours classifier.
    """

    training_data = feature_data
    test_data = feature_data

    n_neighbors = 3
    nbrs = neighbors.NearestNeighbors(n_neighbors).fit(training_data)
    print(
        "k-nearest neighbours uses the Euclidean distance between each\n"
        "sample in the multi-dimensional space defined by the number of\n"
        "feature measurements to find the k most similar samples in a\n"
        "training data set to each sample in a test data set.\n"
    )
    neighbour_distances, neighbour_indices = nbrs.kneighbors(test_data)
    n_samples = 5
    print(
        "For the breast cancer data, the indices of the nearest "
        "%d neighbours\nto the first %d samples are:\n%s"
        % (n_neighbors, n_samples, neighbour_indices[:n_samples])
    )
    print(
        "The distances to these neighbours are:\n%s\n"
        % neighbour_distances[:n_samples]
    )

    print(
        "Let's try using a k-nearest neighbours classifier trained on the\n"
        "entire dataset and see how it performs against the same dataset.\n"
    )
    for n_neighbors in [3, 15]:
        knn = neighbors.KNeighborsClassifier(n_neighbors)
        knn_fitted = knn.fit(feature_data, classification_data_numerical)
        predicted_classification = knn_fitted.predict(feature_data)
        n_differences = \
            (predicted_classification != classification_data_numerical).sum()
        print(
            "For %d neighbours, the number of samples misclassified is: %d"
            % (n_neighbors, n_differences)
        )
        print(
            "This corresponds to an accuracy of %.1f%%\n" %
            (metrics.accuracy_score(
                classification_data_numerical, predicted_classification
            ) * 100)
        )

def optimise_nearest_neighbours(feature_data, classification_data_numerical):
    """
    Find the best combination of parameters for a nearest neighbour classifier
    for this data set.
    """

    max_accuracy = None
    best_n_neighbors = best_weighting = None
    for n_neighbors in range(3, 16):
        for weighting in ['uniform', 'distance']:
            knn = neighbors.KNeighborsClassifier(
                n_neighbors, weights=weighting
            )
            knn_fitted = knn.fit(feature_data, classification_data_numerical)
            predicted_classification = knn_fitted.predict(feature_data)
            accuracy = metrics.accuracy_score(
                classification_data_numerical, predicted_classification
            )
            print(
                "For %d nearest neighbours with %s weighting, accuracy is %.1f"
                % (n_neighbors, weighting, 100*accuracy)
            )
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_n_neighbors = n_neighbors
                best_weighting = weighting
    print(
        "Best parameters seem to be %d nearest neighbours with %s weighting"
        % (best_n_neighbors, best_weighting)
    )

def main():
    """
    Main function of the script.
    """
    feature_data, classification_data = wdbc.load_data_set()

    # scikit-learn functions require classification in terms of numerical
    # values (i.e. 0, 1, 2) instead of strings (e.g. 'benign', 'malignant')
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(classification_data)
    classification_data_numerical = label_encoder.transform(classification_data)

    test_nearest_neighbors(feature_data, classification_data_numerical)
    optimise_nearest_neighbours(feature_data, classification_data_numerical)

main()
