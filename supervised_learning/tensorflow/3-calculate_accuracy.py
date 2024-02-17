#!/usr/bin/env python3
""" calculate_accuracy function """
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y (placeholder): Placeholder for the labels of the input data.
        y_pred (tensor): Tensor containing the network’s predictions.

    Returns:
        A tensor containing the decimal accuracy of the prediction.
    """
    prediction = tf.cast(
        tf.equal(
            tf.argmax(y, axis=1),
            tf.argmax(y_pred, axis=1)
        ),
        dtype="float32"
    )
    accuracy = tf.reduce_mean(prediction)
    return accuracy
