#!/usr/bin/env python3
""" Multiple Kernels """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images (np.ndaray): Matrix with shape (m, h, w, c) containing images.
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
            - c is the number of channels in the image
        kernels (np.ndarray): Matrix with shape (kh, kw, c, nc) containing the
            kernels for the convolution.
            - kh is the height of a kernel
            - kw is the width of a kernel
            - nc is the number of kernels
        padding (str, optional): Either tuple of (ph, pw), 'same', or 'valid'.
            Defaults to 'same'.
            - if 'same', performs a same convolution
            - if 'valid', performs a valid convolution
            - if a tuple:
                - ph is the padding for the height of the image
                - pw is the padding for the width of the image.
        stride (tuple, optional): Tuple of (sh, sw). Defaults to (1, 1).
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image.

    Returns:
        np.ndarray: The matrix containing the convolved images.
    """
    m, h, w, c_i = images.shape
    kh, kw, c_k, nc = kernels.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    else:
        # Same -> p = (k - h + (h - 1)*s) // 2 + 1
        pad = {'valid': (0, 0),
               'same': ((kh - h + (h - 1)*sh) // 2 + 1,
                        (kw - w + (w - 1)*sw) // 2 + 1)}
        ph, pw = pad[padding]

    output_height = (h + 2*ph - kh) // sh + 1
    output_width = (w + 2*pw - kw) // sw + 1

    convolutions = np.zeros((m, output_height, output_width, nc))

    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    for k in range(nc):
        for i in range(output_height):
            for j in range(output_width):
                i_s = i * sh
                j_s = j * sw
                kernel = kernels[:, :, :, k]

                zone = padded_images[:, i_s:i_s+kh, j_s:j_s+kw, :]
                convolutions[:, i, j, k] = np.sum(zone*kernel, axis=(1, 2, 3))

    return convolutions
