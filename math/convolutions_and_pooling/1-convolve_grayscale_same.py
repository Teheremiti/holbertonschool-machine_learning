#!/usr/bin/env python3
""" Same Convolution """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

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

    padding_height = int((kh - 1) / 2)
    padding_width = int((kw - 1) / 2)

    convolutions = np.zeros((m, h, w))

    padded_images = np.pad(images,
                           ((0, 0), (padding_height, padding_height),
                            (padding_width, padding_width)),
                           mode='constant')

    for i in range(h):
        for j in range(w):
            image_zone = padded_images[:, i:i+kh, j:j+kw]
            convolutions[:, i, j] = np.sum(image_zone * kernel, axis=(1, 2))

    return convolutions
