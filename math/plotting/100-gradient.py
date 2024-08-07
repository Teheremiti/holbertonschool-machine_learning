#!/usr/bin/env python3
""" Gradient """

import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """ Scatter plots sampled elevations on a mountain. """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    plt.title('Mountain Elevation')
    plt.scatter(x, y, c=z)
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.colorbar().set_label('elevation (m)')

    plt.show()
