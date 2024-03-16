#!/usr/bin/env python3
""" Early stopping """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.

    Args:
        cost (float): The current validation cost of the neural network.
        opt_cost (float): The lowest recorded validation cost of the neural
            network.
        threshold (float): the threshold used for early stopping.
        patience (int): the patience count used for early stopping.
        count (int): the count of how long the threshold has not been met.

    Returns:
        bool, int: Whether the network should be stopped early, followed by the
            updated count.
    """
    count = 0 if opt_cost - cost > threshold else count + 1
    return count == patience, count
