#!/usr/bin/env python3
"""
GaussianProcess class for 1D Gaussian Process Regression.
"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian Process.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes the Gaussian Process.

        Args:
            X_init (np.ndarray): shape (t, 1), the inputs sampled
            Y_init (np.ndarray): shape (t, 1), the outputs of
                the black-box function
            l (float): length-scale parameter
            sigma_f (float): signal variance
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two inputs using
        the RBF kernel.

        Args:
            X1 (np.ndarray): shape (m, 1)
            X2 (np.ndarray): shape (n, 1)

        Returns:
            Covariance matrix as np.ndarray of shape (m, n)
        """
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
            np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)
