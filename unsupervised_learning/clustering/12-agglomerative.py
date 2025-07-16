#!/usr/bin/env python3
"""
Agglomerative clustering using scipy.
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset using Ward linkage.

    This function uses hierarchical clustering to group data points into
    clusters. It builds a linkage matrix using Ward's method, displays a
    dendrogram showing the clustering hierarchy, and returns cluster
    assignments based on the maximum cophenetic distance threshold.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions for each data point
        dist (float): Maximum cophenetic distance for all clusters. This
                     threshold determines where to cut the dendrogram to form
                     the final clusters.

    Returns:
        numpy.ndarray: Array of shape (n,) containing the cluster indices
                      for each data point

    Notes:
        - Uses Ward linkage method which minimizes within-cluster variance
        - Displays a dendrogram with different colors for each cluster
        - Cluster indices start from 1 (not 0) as per scipy convention
        - The dendrogram is displayed automatically when the function is called
    """
    # Perform hierarchical clustering using Ward linkage
    # Ward method minimizes the variance of clusters being merged
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method='ward')

    # Display the dendrogram with color threshold
    # Each cluster will be displayed in a different color
    plt.figure(figsize=(10, 6))
    scipy.cluster.hierarchy.dendrogram(
        linkage_matrix,
        color_threshold=dist,
        above_threshold_color='grey'
    )
    plt.title('Agglomerative Clustering Dendrogram')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    plt.show()

    # Extract flat clusters from the linkage matrix
    # Using 'distance' criterion with the specified threshold
    clss = scipy.cluster.hierarchy.fcluster(
        linkage_matrix,
        dist,
        criterion='distance'
    )

    return clss
