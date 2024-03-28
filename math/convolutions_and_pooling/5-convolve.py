#!/usr/bin/env python3
""" Multiple Kernels """
import numpy as np


def convolve(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernel.

    Args:
        images (ndarray): Matrxi of shape (m, h, w, c), multiple images.
        kernel (ndarray): Matrxi of shape (kh,kw,c,nc), kernel for convolution.
        padding (tuple): (ph,pw) or "same" or "valid".
        stride (tuple): (sh, sw).

    Returns:
        ndarray: The matrix containing the convolved images.
    """
    # size images, kernel, padding, stride
    m, h, w, c = images.shape
    kh, kw, _, nc = kernel.shape
    sh, sw = stride

    # output size and padding
    if padding == 'valid':
        # no padding
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)

    elif isinstance(padding, tuple):
        ph, pw = padding

    # generalize output calcul
    output_height = int((h - kh + 2 * ph) / sh + 1)
    output_width = int((w - kw + 2 * pw) / sw + 1)

    # initialize output
    convolved_images = np.zeros((m, output_height, output_width, nc))

    # pad image
    image_pad = np.pad(images,
                       ((0, 0), (ph, ph),
                        (pw, pw), (0, 0)), mode='constant')

    # convolution
    for k in range(nc):
        for i in range(output_height):
            for j in range(output_width):
                # extract region from each image
                image_zone = image_pad[:, i * sh:i * sh + kh,
                                       j * sw:j * sw + kw, :]

                # element wize multiplication
                convolved_images[:, i, j, k] = np.sum(image_zone
                                                      * kernel[:, :, :, k],
                                                      axis=(1, 2, 3))

    return convolved_images
