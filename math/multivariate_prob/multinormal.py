#!/usr/bin/env python3
"""MultiNormal class"""
import numpy as np


class MultiNormal:
    """Multivariate Normal distribution"""

    def __init__(self, data):
        """
        Class constructor.

        Args:
        data (np.ndarray): Data set of shape (d, n), with `n` and `d` the
        numbers of data points and dimensions in each data point, respectively.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        mean_cov = __import__('0-mean_cov').mean_cov

        # Transpose data to match the dimensions in the `mean_cov` method
        mean, cov = mean_cov(data.T)
        self.mean = mean.T
        self.cov = cov

    def pdf(self, x):
        """
        Calculates the PDF at a data point using
        ```
        f(x) = (1 / (sqrt((2 * π)^k * det(Σ))))
               * exp(-0.5 * (x - μ)^T * Σ^(-1) * (x - μ))
        ```

        Args:
        x (np.ndarray): Data point of shape (d, 1) whose PDF should be
        calculated.

        Returns:
        The value of the PDF.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.ndim != 2 or x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        diff = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)

        if cov_det <= 0:
            raise ValueError("Covariance matrix must be positive definite")

        exponent = -0.5 * (diff.T @ cov_inv @ diff)
        coeff = 1.0 / np.sqrt(((2 * np.pi) ** d) * cov_det)
        pdf = coeff * np.exp(exponent)

        return float(pdf)
