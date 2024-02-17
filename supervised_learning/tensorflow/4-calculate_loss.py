#!/usr/bin/env python3
""" calculate_loss function """
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y (placeholder): Placeholder for the labels of the input data.
        y_pred (tensor): Tensor containing the network’s predictions.

    Returns:
        A tensor containing the loss of the prediction.
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
