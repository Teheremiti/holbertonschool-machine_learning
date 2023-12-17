#!/usr/bin/env python3
""" Exponential class """


class Exponential:
    """ Class that represents the Exponential distribution """
    def __init__(self, data=None, lambtha=1.):
        """
        Constructor method for the Exponential class

        Args:
            data (list, optional): List of the data to be used to estimate
                the distribution. Defaults to None.
            lambtha (float, optional): Expected number of occurences in a
                given time frame. Defaults to 1..
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = 1 / (sum(data) / len(data))
