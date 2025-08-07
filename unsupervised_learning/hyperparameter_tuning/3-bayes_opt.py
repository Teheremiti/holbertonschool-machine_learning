#!/usr/bin/env python3
"""
BayesianOptimization class for 1D noiseless optimization.
"""

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes the Bayesian optimizer.

        Args:
            f (callable): the black-box function to be optimized
            X_init (np.ndarray): shape (t, 1), sampled inputs
            Y_init (np.ndarray): shape (t, 1), sampled outputs
            bounds (tuple): (min, max), the search space bounds
            ac_samples (int): number of acquisition sample points
            l (float): length-scale for the kernel
            sigma_f: standard deviation for the black-box output
            xsi (float): exploration-exploitation factor
            minimize (bool): whether to minimize (True) or maximize (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.xsi = xsi
        self.minimize = minimize

        X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.X_s = X_s
