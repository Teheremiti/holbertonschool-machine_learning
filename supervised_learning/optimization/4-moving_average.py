#!/usr/bin/env python3
""" moving_average function """
import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Args:
        data (list): The list of data to calculate the moving average of.
        beta (float): The weight used for the moving average.

    Returns:
        The list containing the moving averages of data.
    """
    m_av = []
    w = 0
    for i, d in enumerate(data):
        w = beta * w + (1 - beta) * d
        w_new = w / (1 - beta ** (i + 1))
        m_av.append(w_new)
    return m_av
