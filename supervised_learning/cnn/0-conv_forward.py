#!/usr/bin/env python3
""" Convolutional forward propagation """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs a forward propagation over a convolutional layer of a neural
    network.

    Args:
        A_prev (np.ndarray): Matrix of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
            - m is the number of examples
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
            - c_prev is the number of channels in the previous layer
        W (np.ndarray): Matrix of shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution.
            - kh is the filter height
            - kw is the filter width
            - c_prev is the number of channels in the previous layer
            - c_new is the number of channels in the output
        b (np.ndarray): Matrix of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution.
        activation (str): The activation function applied to the convolution.
        padding (str, optional): Either 'same' or 'valid', indicating the type
            of padding used. Defaults to "same".
        stride (tuple, optional): Tuple of (sh, sw) containing the strides for
            the convolution. Defaults to (1, 1).
            - sh is the stride for the height
            - sw is the stride for the width.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    pad = {'valid': (0, 0),
           'same': ((kh - h_prev + (h_prev - 1)*sh) // 2,
                    (kw - w_prev + (w_prev - 1)*sw) // 2)}
    ph, pw = pad[padding]

    output_height = (h_prev + 2*ph - kh) // sh + 1
    output_width = (w_prev + 2*pw - kw) // sw + 1

    padded_images = np.pad(A_prev,
                           ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    convolutions = np.zeros((m, output_height, output_width, c_new))

    for k in range(c_new):
        for i in range(output_height):
            for j in range(output_width):
                i_s = i * sh
                j_s = j * sw
                kernel = W[:, :, :, k]

                zone = padded_images[:, i_s:i_s+kh, j_s:j_s+kw, :]
                convolutions[:, i, j, k] = np.sum(zone*kernel, axis=(1, 2, 3))

    Z = convolutions + b
    A = activation(Z)

    return A
