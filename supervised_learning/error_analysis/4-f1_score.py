#!/usr/bin/env python3
""" Function f1_score """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix.

    Args:
        confusion (ndarray): Matrix of shape (classes, classes) where row
            indices represent the correct labels and column indices represent
            the predicted labels.

    Returns:
        ndarray: Matrix  of shape (classes,) containing the F1 score of each
            class.
    """
    classes = confusion.shape[0]
    score = np.zeros((classes,))

    preci = precision(confusion)
    sensi = sensitivity(confusion)

    for i in range(classes):
        score[i] = (2 * (preci[i] * sensi[i]) / (preci[i] + sensi[i]))

    return score
