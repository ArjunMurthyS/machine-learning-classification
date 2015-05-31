#!/usr/bin/env python

"""
Plot distribution and correlation of feature data from breast cancer data set.
"""

from __future__ import print_function
import csv
import numpy
import scipy
import scipy.stats
import matplotlib.pyplot as plt

def load_data_set():
    """
    Load the UCI breast cancer data set.
    (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
    """
    data_set_file_name = "wdbc.csv"
    with open(data_set_file_name, "r") as data_set_file:
        # csv.reader returns an interator which must be converted into a list first
        data_set_list = list(csv.reader(data_set_file))

    data_set = numpy.asarray(data_set_list)
    # always examine data before starting!
    # (data may have silently loaded in the wrong format, or truncated)
    # use (tuple,) to form single tuple from original tuple
    print("Shape of data set is:\n%s" % (data_set.shape,))
    print("First row of data set is:\n%s" % data_set[0, :])
    # second column is benign/malignant classification
    classification_data = data_set[:, 1]
    # ignore first two columns (sample ID and classification)
    # to get feature data
    feature_data = data_set[:, 2:].astype(float)
    print("Shape of feature data is:\n%s" %
        (feature_data.shape,))
    print("Shape of classification data is:\n%s\n" %
        (classification_data.shape,))

    return feature_data, classification_data

def plot_classification_distribution(classification_data):
    """
    Plot the distribution of the benign/malignant distribution
    using bar charts.
    """
    classification_frequency = scipy.stats.itemfreq(classification_data)

    plt.figure(2)

    for row_n, row in enumerate(classification_frequency):
        if row[0] == 'B':
            label = 'Benign'
            color = 'b'
        elif row[0] == 'M':
            label = 'Malignant'
            color = 'r'
        else:
            raise Exception("Unkown classification:", row[0])
        frequency = int(row[1])
        plt.bar(left=row_n, height=frequency, color=color, label=label)

    plt.gca().axes.xaxis.set_ticklabels([])
    plt.legend()
    plt.xlabel("Diagnosis")
    plt.ylabel("Frequency")
    plt.title("Distribution of Classifications")
    print(
        "In order to have our classifier be adept at spotting all classes,\n"
        "we must ensure our data has a reasonably equal distribution.\n"
    )
    plt.show()

def plot_feature_correlation(feature_data):
    """
    Plot correlation between features.
    """
    # rowvar=0: specify that the first dimension of the matrix (the rows)
    # represent the different cases, and look for correlation between features
    correlation_matrix = numpy.corrcoef(feature_data, rowvar=0)

    plt.figure(2)
    plt.title("Feature Correlation")
    colormap = plt.cm.Blues
    plt.gca().pcolor(correlation_matrix, cmap=colormap)
    plt.xlabel("Feature 1 index")
    plt.ylabel("Feature 2 index")

    print(
        "Ideally we want to pick features for input to a classifier that have\n"
        "a minimum amount of correlation. The idea is that the data should be\n"
        "as distinct as possible to enable maximum separation.\n"
        "In this plot, the darker the square, the stronger the correlation.\n"
    )

    plt.show()

def main():
    """
    Main function of the script.
    """
    feature_data, classification_data = load_data_set()
    plot_classification_distribution(classification_data)
    plot_feature_correlation(feature_data)

main()
