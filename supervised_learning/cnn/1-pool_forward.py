#!/usr/bin/env python3
""" Pooling forward propagation """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Args:
        A_prev (np.ndarray): Matrixof shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
            - m is the number of examples
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
            - c_prev is the number of channels in the previous layer
        kernel_shape (tuple): Tuple of (kh, kw) containing the size of the
            kernel for the pooling.
            - kh is the kernel height
            - kw is the kernel width
        stride (tuple, optional): tuple of (sh, sw) containing the strides for
            the pooling. Defaults to (1, 1).
            - sh is the stride for the height
            - sw is the stride for the width.
        mode (str, optional): Either max or avg, indicating whether to perform
            maximum or average pooling, respectively. Defaults to 'max'.

    Returns:
        np.ndarray: The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_height = (h_prev - kh) // sh + 1
    output_width = (w_prev - kw) // sw + 1

    pooled_images = np.zeros((m, output_height, output_width, c_prev))

    for i in range(output_height):
        for j in range(output_width):
            i_s = i * sh
            j_s = j * sw

            image_zone = A_prev[:, i_s:i_s+kh, j_s:j_s+kw, :]

            methods = {'max': np.max, 'avg': np.average}
            pooled_images[:, i, j, :] = methods[mode](image_zone, axis=(1, 2))

    return pooled_images
