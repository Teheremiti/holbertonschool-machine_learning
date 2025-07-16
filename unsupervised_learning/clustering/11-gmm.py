#!/usr/bin/env python3
"""
Gaussian Mixture Model using sklearn.
"""

import sklearn.mixture


def gmm(X, k):
    """
    Calculates a Gaussian Mixture Model from a dataset using sklearn.

    This function uses the scikit-learn GaussianMixture implementation to fit a
    GMM with k components to the data. It returns the model parameters
    including priors, means, covariances, cluster assignments, and BIC score.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions for each data point
        k (int): The number of clusters (mixture components)

    Returns:
        tuple: A tuple containing (pi, m, S, clss, bic) where:
            - pi (numpy.ndarray): Array of shape (k,) containing the cluster
                                 priors
            - m (numpy.ndarray): Array of shape (k, d) containing the centroid
                                means
            - S (numpy.ndarray): Array of shape (k, d, d) containing the
                                covariance matrices
            - clss (numpy.ndarray): Array of shape (n,) containing the cluster
                                   indices for each data point
            - bic (float): The BIC (Bayesian Information Criterion) value for
                          the model

    Notes:
        - Uses sklearn.mixture.GaussianMixture with default parameters
        - Cluster assignments are determined by the component with highest
          posterior probability
        - BIC is computed automatically by sklearn for model comparison
    """
    # Create GaussianMixture object with k components
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)

    # Fit the model to the data
    gmm_model.fit(X)

    # Get cluster priors (mixing coefficients)
    pi = gmm_model.weights_

    # Get cluster means
    m = gmm_model.means_

    # Get covariance matrices
    S = gmm_model.covariances_

    # Get cluster assignments (predict the most likely component pour each
    # sample)
    clss = gmm_model.predict(X)

    # Get BIC score
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
