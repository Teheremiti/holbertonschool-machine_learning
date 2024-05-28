#!/usr/bin/env python3
""" Strided Convolution """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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
        padding ({tuple, str}): Either a tuple of (ph, pw) defining the height
            and width of the padding, or 'same' or 'valid'. Defaults to 'same'.
            - ph is the padding for the height of the image
            - pw is the padding for the width of the image
        stride (tuple, optional): Tuple of (sh, sw) defining the height and
            width of the stride. Defaults to (1 , 1).
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image

    Returns:
        np.ndarray: The matrix containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    else:
        pad = {'valid': (0, 0), 'same': (kh // 2, kw // 2)}
        ph, pw = pad[padding]

    output_height = (h + 2*ph - kh) // sh + 1
    output_width = (w + 2*pw - kw) // sw + 1

    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    convolutions = np.zeros((m, output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            i_s = i * sh
            j_s = j * sw
            image_zone = padded_images[:, i_s:i_s+kh, j_s:j_s+kw]
            convolutions[:, i, j] = np.sum(image_zone * kernel, axis=(1, 2))

    return convolutions
