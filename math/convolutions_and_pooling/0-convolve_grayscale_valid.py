#!/usr/bin/env python3
""" Valid Convolution """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

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

    Returns:
        np.ndarray: The matrix containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_height = h - kh + 1
    output_width = w - kw + 1

    convolutions = np.zeros((m, output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            # extract region from each image
            image_zone = images[:, i:i+kh, j:j+kw]

            # element wize multiplication
            convolutions[:, i, j] = np.sum(image_zone * kernel, axis=(1, 2))

    return convolutions
