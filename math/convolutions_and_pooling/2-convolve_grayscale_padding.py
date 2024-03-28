#!/usr/bin/env python3
""" Convolution with Padding """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images (ndarray): Matrix of shape (m, h, w), multiple grayscale images.
        kernel (ndarray): Matrix of shape (kh,kw), kernel for convolution.
        padding (tupple): (ph,pw).

    Returns:
        ndarray: The matrix containing the convolved images.C
    """
    # size images and kernel
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # output size
    output_height = h - kh + 1 + 2 * ph
    output_width = w - kw + 1 + 2 * pw

    # initialize output
    convolved_images = np.zeros((m, output_height, output_width))

    # add zero padding to the input images
    image_pad = np.pad(images,
                       ((0, 0), (ph, ph),
                        (pw, pw)), mode='constant')

    # convolution
    for i in range(output_height):
        for j in range(output_width):
            # extract region from each image
            image_zone = image_pad[:, i:i+kh, j:j+kw]

            # element wize multiplication
            convolved_images[:, i, j] = np.sum(image_zone * kernel,
                                               axis=(1, 2))

    return convolved_images
