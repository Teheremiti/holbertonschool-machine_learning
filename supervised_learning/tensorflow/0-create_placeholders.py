#!/usr/bin/env python3
""" create_placeholder function """
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Creates two placeholders for the intput data.

    Args:
        nx (int): The number of feature columns in our data.
        classes (int): The number of classes in our classifier.

    Returns:
        Placeholders x and y for the input data to the neural network and
        for the one-hot labels for the input data, respectively.
    """
    x = tf.placeholder(dtype="float32", shape=[None, nx], name="x")
    y = tf.placeholder(dtype="float32", shape=[None, classes], name="y")
    return x, y
