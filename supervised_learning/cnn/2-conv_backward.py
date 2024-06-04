#!/usr/bin/env python3
""" Convolutional backward propagation """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Args:
        dZ (np.ndarray): Matrix of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated output of
            the convolutional layer.
            - m is the number of examples
            - h_new is the height of the output
            - w_new is the width of the output
            - c_new is the number of channels in the output
        A_prev (np.ndarray): Matrix of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
            - c_prev is the number of channels in the previous layer
        W (np.ndarray): Matrix of shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution.
            - kh is the filter height
            - kw is the filter width
        b (np.ndarray): Matrix of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution.
        padding (str, optional): Either same or valid, indicating the type of
            padding used. Defaults to "same".
        stride (tuple, optional): tuple of (sh, sw) containing the strides for
            the convolution. Defaults to (1, 1).
            - sh is the stride for the height
            - sw is the stride for the width.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: The partial derivatives with
            respect to the previous layer (dA_prev), the kernels (dW), and the
            biases (db), respectively.
    """
    m_new, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, kc_prev, kc_new = W.shape
    sh, sw = stride

    pad = {'valid': (0, 0),
           'same': ((kh - h_prev + (h_prev - 1)*sh) // 2 + 1,
                    (kw - w_prev + (w_prev - 1)*sw) // 2 + 1)}
    ph, pw = pad[padding]

    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')

    dA_prev = np.zeros(A_prev_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    h_s = h * sh
                    w_s = w * sw

                    dA_prev[i, h_s:h_s+kh, w_s:w_s+kw, :] += W[:, :, :, f]\
                        * dZ[i, h, w, f]

                    dW[:, :, :, f] += A_prev_pad[i, h_s:h_s+kh, w_s:w_s+kw, :]\
                        * dZ[i, h, w, f]

    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
