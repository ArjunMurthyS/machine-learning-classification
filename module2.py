#!/usr/bin/env python

"""
Plot feature data distribution from Fisher's iris data set.
"""

from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def load_iris_dict():
    """
    Load the dictionary containing the iris data set
    and print some information about it.
    """
    iris = load_iris()

    # always examine data before starting!
    # (data may have silently loaded in the wrong format, or truncated)
    print("Iris feature names are:\n%s\n" % iris.feature_names)

    # data.shape is a tuple, so must use (data.shape,) to form a tuple
    # containg that single tuple in order to print with string formatting
    print("Shape of measurement data is:\n%s" % (iris.data.shape,))
    print("This corresponds to %d samples (%d flowers)\n"
          "with %d features recorded for each sample\n"
          % (iris.data.shape[0], iris.data.shape[0], iris.data.shape[1]))

    print("Iris classification types are:\n%s\n" % iris.target_names)

    print("Shape of classifications is:\n%s\b" % (iris.target.shape,))
    print("i.e. one classification for each flower")

    return iris

def plot_features(iris_dict):
    """
    Plot one feature of the iris data set against another feature
    to show how different features form clusters for the different types of
    irises, enabling classification of one type from another.
    """
    # 'X' is the standard name for the feature data matrix
    x_matrix = iris_dict.data
    labels = iris_dict.target_names

    class_markers = ["+", "_", "x"]
    class_colours = ["blue", "magenta", "cyan"]
    # the index of each class in target_names
    class_indices = [0, 1, 2]
    # the indices (the column indices in iris.data)
    # of the two features to plot against each other
    feature_1_index = 0
    feature_2_index = 1

    graph_data = zip(class_markers, class_colours, class_indices, labels)
    for mark, col, i, label in graph_data:
        # this uses a couple of numpy ndarray-specific functions:
        # * ndarray == <scalar> does element-wise equality testing, returning
        #   a matrix the same size as the original filled with booleans
        # * ndarray[boolean array] returns a one-dimensional array filled with
        #   the elements of the original matrix corresponding to true values in
        #   the boolean array
        #   (http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#boolean-array-indexing)
        feature_1_data = x_matrix[iris_dict.target == i, feature_1_index]
        feature_2_data = x_matrix[iris_dict.target == i, feature_2_index]

        plt.scatter(
            x=feature_1_data, y=feature_2_data,
            marker=mark, c=col, label=label
        )

    plt.legend(loc='best')
    plt.xlabel(iris_dict.feature_names[feature_1_index])
    plt.ylabel(iris_dict.feature_names[feature_2_index])
    plt.show()

plot_features(load_iris_dict())
