#!/usr/bin/env python3
""" Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images (np.ndarray): Matrix with shape (m, h, w, c) containing images.
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
            - c is the number of channels in the image
        kernel_shape (np.ndarray): Tuple of (kh, kw) containing the kernel
            shape for the pooling.
            - kh is the height of the kernel
            - kw is the width of the kernel
        stride (_type_):  Tuple of (sh, sw).
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image
        mode (str, optional): Indicates the type of pooling.
            - max indicates max pooling
            - avg indicates average pooling. Defaults to 'max'.

    Returns:
        np.ndarray: The matrix containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_height = (h - kh) // sh + 1
    output_width = (w - kw) // sw + 1

    pooled_images = np.zeros((m, output_height, output_width, c))

    for i in range(output_height):
        for j in range(output_width):
            i_s = i * sh
            j_s = j * sw

            image_zone = images[:, i_s:i_s+kh, j_s:j_s+kw, :]

            methods = {'max': np.max, 'avg': np.average}
            pooled_images[:, i, j, :] = methods[mode](image_zone, axis=(1, 2))

    return pooled_images
