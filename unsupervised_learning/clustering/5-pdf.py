#!/usr/bin/env python3
"""
Probability Density Function (PDF)
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    This function computes the multivariate Gaussian PDF for given data points
    using the mean vector and covariance matrix. The PDF formula is:
    P(x) = (1 / ((2π)^(d/2) * |Σ|^(1/2))) * exp(-1/2 * (x-μ)ᵀ * Σ⁻¹ * (x-μ))

    Args:
        X (numpy.ndarray): Data points of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions
        m (numpy.ndarray): Mean vector of shape (d,) containing the mean
                          of the distribution
        S (numpy.ndarray): Covariance matrix of shape (d, d) containing
                          the covariance of the distribution

    Returns:
        numpy.ndarray: Array of shape (n,) containing the PDF values for
                      each data point, or None on failure

    Notes:
        - All PDF values have a minimum value of 1e-300 to avoid numerical
          issues
        - Uses vectorized operations without loops for efficiency
        - Handles singular covariance matrices gracefully
    """
    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None

    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    # Get dimensions
    n, d = X.shape

    # Check dimension compatibility
    if m.shape[0] != d or S.shape != (d, d):
        return None

    # Calculate determinant of covariance matrix
    try:
        det_S = np.linalg.det(S)
        if det_S <= 0:
            return None
    except np.linalg.LinAlgError:
        return None

    # Calculate inverse of covariance matrix
    try:
        inv_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    # Calculate normalization constant
    # (2π)^(d/2) * |Σ|^(1/2)
    norm_const = ((2 * np.pi) ** (d / 2)) * (det_S ** 0.5)

    # Center the data points (X - μ)
    centered_X = X - m

    # Calculate quadratic form: (x - μ)ᵀ * Σ⁻¹ * (x - μ)
    # Using einsum for efficient matrix multiplication
    # 'ij,jk,ik->i' means: for each row i, compute sum over j,k of
    # centered_X[i,j] * inv_S[j,k] * centered_X[i,k]
    quadratic_form = np.einsum('ij,jk,ik->i', centered_X, inv_S, centered_X)

    # Calculate PDF values
    P = (1.0 / norm_const) * np.exp(-0.5 * quadratic_form)

    # Apply minimum value constraint
    P = np.maximum(P, 1e-300)

    return P
