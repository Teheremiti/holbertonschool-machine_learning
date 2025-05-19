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
