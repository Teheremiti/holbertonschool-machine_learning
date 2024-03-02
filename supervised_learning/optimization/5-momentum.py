#!/usr/bin/env python3
""" Momentum """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization
    algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (ndarray): Matrix containing the variable to be updated.
        grad (ndarray): Matrix containing the gradient of var.
        v (ndarray): The previous first moment of var.

    Returns:
        The updated variable and the new moment.
    """
    dW = beta1 * v + (1 - beta1) * grad
    var_new = var - dW * alpha
    return var_new, dW
