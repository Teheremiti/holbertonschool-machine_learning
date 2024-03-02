#!/usr/bin/env python3
""" Learning Rate decay upgraded """
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates learning rate decay operation in tensorflow using inverse
    time decay.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight used to determine the rate at which
            alpha will decay.
        global_step (int): The number of passes of gradient descent that have
            elapsed.
        decay_step (int): The number of passes of gradient descent that should
            occur before alpha is decayed further.

    Returns:
        The learning rate decay operation.
    """
    return tf.compat.v1.train.inverse_time_decay(
        learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        global_step=global_step,
        staircase=True)
