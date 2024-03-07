#!/usr/bin/env python3
""" Function sensitivity """
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion (ndarray): Confusion matrix of shape (classes, classes) where
            row indices represent the correct labels and column indices
            represent the predicted labels. Classes is the number of classes.

    Returns:
        ndarray: Matrix of shape (classes,) containing the sensitivity of each
            class.
    """
    classes = confusion.shape[0]
    sensitivity = np.zeros((classes,))

    for i in range(classes):
        positif = confusion[i, i]
        total = np.sum(confusion[i, :])
        sensitivity[i] = positif / total

    return sensitivity
