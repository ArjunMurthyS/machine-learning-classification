#!/usr/bin/env python

"""
Evaluate effectiveness of k-nearest neighbours classifier
"""

from __future__ import print_function
from sklearn import neighbors
import wdbc

def test_nearest_neighbors(feature_data):
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
        "The distances to these neighbours are:\n%s"
        % neighbour_distances[:n_samples]
    )

def main():
    """
    Main function of the script.
    """
    feature_data, classification_data = wdbc.load_data_set()
    test_nearest_neighbors(feature_data)

main()
