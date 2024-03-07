#!/usr/bin/env python3
""" Function create_confusion_matrix """
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels (ndarray): Matrix of shape (m, classes), contains the correct
            labels for each data point.
        logits (ndarray): Matrix of shape (m, classes), predicted labels.

    Returns:
        ndarray: The confusion matrix of shape (m, classes) with row indices
            representing the correct labels and column indices representing the
            predicted labels.
    """
    m = labels.shape[0]
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    for i in range(m):
        correct = np.argmax(labels[i])
        predicted = np.argmax(logits[i])
        confusion[correct, predicted] += 1

    return confusion
