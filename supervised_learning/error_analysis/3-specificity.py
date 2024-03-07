#!/usr/bin/env python3
""" Function specificity """
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion (ndarray): Matrix of shape (classes, classes) where row
            indices represent the correct labels and column indices represent
            the predicted labels.

    Returns:
        ndarray: Matrix of shape (classes,) containing the specificity of each
            class.
    """
    classes = confusion.shape[0]
    specificity = np.zeros((classes,))

    for i in range(classes):
        true_pos = confusion[i, i]
        false_pos = np.sum(confusion[:, i]) - true_pos
        false_neg = np.sum(confusion[i, :]) - true_pos
        true_neg = np.sum(confusion) - (true_pos + false_pos + false_neg)
        specificity[i] = true_neg / (true_neg + false_pos)

    return specificity
