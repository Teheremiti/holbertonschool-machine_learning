#!/usr/bin/rnv python3
""" Convolution with Channels """
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images (np.ndarray): Matrix with shape (m, h, w, c) containing images.
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
            - c is the number of channels in the image
        kernel (_type_): Matrix with shape (kh, kw, c) containing the kernel
            for the convolution.
            - kh is the height of the kernel
            - kw is the width of the kernel
        padding (str, optional): Either a tuple of (ph, pw), 'same', or
            'valid'. Defaults to 'same'.
            - ph is the padding for the height of the image
            - pw is the padding for the width of the image
        stride (tuple, optional): Tuple of (sh, sw). Defaults to (1, 1).
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image

    Returns:
        np.ndaray: The matrix containing the convolved images.
    """
    m, h, w, c_i = images.shape
    kh, kw, c_k = kernel.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    else:
        # Same -> p = (k - h + (h - 1)*s) // 2 + 1
        pad = {'valid': (0, 0),
               'same': ((kh - h + (h - 1)*sh) // 2 + 1,
                        (kw - w + (w - 1)*sw) // 2 + 1)}
        ph, pw = pad[padding]

    output_height = (h+ 2*ph - kh) // sh + 1
    output_width = (w + 2*pw - kw) // sw + 1

    convolutions = np.zeros((m, output_height, output_width))

    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    for i in range(output_height):
        for j in range(output_width):
            i_s = i * sh
            j_s = j * sw
            image_zone = padded_images[:, i_s:i_s+kh, j_s:j_s+kw, :]
            convolutions[:, i, j] = np.sum(image_zone * kernel, axis=(1, 2, 3))

    return convolutions
