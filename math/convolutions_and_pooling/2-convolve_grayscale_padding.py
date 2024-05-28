#!/usr/bin/env python3
""" Convolution with padding """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images (np.ndarray): Matrix with shape (m, h, w) containing multiple
            grayscale images.
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
        kernel (np.ndarray): Matrix with shape (kh, kw) containing the kernel
            for the convolution.
            - kh is the height of the kernel
            - kw is the width of the kernel
        padding (tuple): Tuple of shape (ph, pw) defining the height and width
            of the padding.
            - ph is the padding for the height of the image
            - pw is the padding for the width of the image

    Returns:
        np.ndarray: The matrix containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded_images = np.pad(images,
                           ((0, 0),
                            (ph, ph),
                            (pw, pw)),
                           mode='constant')

    output_height = h + 2*ph - kh + 1
    output_width = w + 2*pw - kw + 1

    convolutions = np.zeros((m, output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            image_zone = padded_images[:, i:i+kh, j:j+kw]
            convolutions[:, i, j] = np.sum(image_zone * kernel, axis=(1, 2))

    return convolutions
