#!/usr/bin/env python3
""" create_train_op function """
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Args:
        loss (tensor): The loss of the network’s prediction.
        alpha (float): The leearning rate.

    Returns:
        An operation that trains the network using gradient descent.
    """
    grad_descent = tf.train.GradientDescentOptimizer(learning_rate=alpha,
                                                     name='GradientDescent')
    return grad_descent.minimize(loss)
