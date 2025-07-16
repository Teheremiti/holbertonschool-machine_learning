#!/usr/bin/env python3
"""
K-means clustering using sklearn.
"""

import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering on a dataset using sklearn.

    This function uses the scikit-learn KMeans implementation to cluster data
    points into k clusters. It returns the cluster centroids and the cluster
    assignments for each data point.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions for each data point
        k (int): The number of clusters

    Returns:
        tuple: A tuple containing (C, clss) where:
            - C (numpy.ndarray): Array of shape (k, d) containing the
                                centroid means for each cluster
            - clss (numpy.ndarray): Array of shape (n,) containing the
                                   index of the cluster in C that each
                                   data point belongs to

    Notes:
        - Uses sklearn.cluster.KMeans with default parameters
        - Cluster assignments are integers from 0 to k-1
        - Centroids are the final cluster centers after convergence
    """
    # Create KMeans object with k clusters
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)

    # Fit the model to the data and get cluster assignments
    clss = kmeans_model.fit_predict(X)

    # Get the cluster centroids
    C = kmeans_model.cluster_centers_

    return C, clss
