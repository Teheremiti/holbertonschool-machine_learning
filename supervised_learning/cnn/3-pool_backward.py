#!/usr/bin/env python3
""" Pooling back propagation """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Args:
        dA (np.ndarray): Matrix of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the output of the pooling
            layer.
            - m is the number of examples
            - h_new is the height of the output
            - w_new is the width of the output
            - c is the number of channels
        A_prev (np.ndarray): Matrix of shape (m, h_prev, w_prev, c) containing
            the output of the previous layer.
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
        kernel_shape (tuple): tuple of (kh, kw) containing the size of the
            kernel for the pooling.
            - kh is the kernel height
            - kw is the kernel width
        stride (tuple, optional): Tuple of (sh, sw) containing the strides for
            the pooling. Defaults to (1, 1).
            - sh is the stride for the height
            - sw is the stride for the width.
        mode (str, optional): Either max or avg, indicating whether to perform
            maximum or average pooling, respectively. Defaults to 'max'.

    Returns:
        np.ndarray: The partial derivatives with respect to the previous layer
            (dA_prev).
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    h_s = h * sh
                    w_s = w * sw

                    if mode == 'avg':
                        avg_dA = dA[i, h, w, f] / kh / kw
                        dA_prev[i, h_s:h_s+kh, w_s:w_s+kw, f] += (
                            np.ones((kh, kw)) * avg_dA)
                    elif mode == 'max':
                        zone = A_prev[i, h_s:h_s+kh, w_s:w_s+kw, f]
                        mask = (zone == np.max(zone))
                        dA_prev[i, h_s:h_s+kh, w_s:w_s+kw, f] +=\
                            mask * dA[i, h, w, f]

    return dA_prev
