#!/usr/bin/env python3
"""
BayesianOptimization class for 1D noiseless optimization.
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes the Bayesian optimizer.

        Parameters:
            f (function): the black-box function to be optimized
            X_init (np.ndarray): shape (t, 1), sampled inputs
            Y_init (np.ndarray): shape (t, 1), sampled outputs
            bounds (tuple): (min, max), the search space bounds
            ac_samples (int): number of acquisition sample points
            l (float): length-scale for the kernel
            sigma_f (float): standard deviation for the black-box output
            xsi (float): exploration-exploitation factor
            minimize (bool): whether to minimize (True) or maximize (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.xsi = xsi
        self.minimize = minimize

        X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.X_s = X_s

    def acquisition(self):
        """
        Computes the Expected Improvement (EI) acquisition function.

        Returns:
            X_next (np.ndarray): shape (1,), the next best sample point
            EI (np.ndarray): shape (ac_samples,), expected improvement values
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        Z = np.zeros_like(mu)
        mask = sigma > 0

        Z[mask] = imp[mask] / sigma[mask]
        EI = np.zeros_like(mu)
        EI[mask] = imp[mask] * \
            norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(Z[mask])

        X_next = self.X_s[np.argmax(EI)].reshape(1)

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function using Bayesian optimization.

        Parameters:
            iterations (int): maximum number of iterations to run

        Returns:
            X_opt (np.ndarray): shape (1,), optimal input found
            Y_opt (np.ndarray): shape (1,), optimal output found
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Stop early if X_next was already sampled
            if np.any(np.isclose(self.gp.X, X_next).all(axis=1)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx].copy()
        Y_opt = self.gp.Y[idx].copy()

        return X_opt, Y_opt
