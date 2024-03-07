#!/usr/bin/env python3
""" Function precision """
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (ndarray): Matrix of shape (classes, classes) where row
            indices represent the correct labels and column indices represent
            the predicted labels.

    Returns:
        ndarray: Matrix of shape (classes,) containing the precision of each
            class.
    """
    classes = confusion.shape[0]
    precision = np.zeros((classes,))

    for i in range(classes):
        true_positif = confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - true_positif
        precision[i] = true_positif / (true_positif + false_positives)

    return precision
