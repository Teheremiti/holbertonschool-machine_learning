#!/usr/bin/env python3
""" Poisson class """


class Poisson:
    """ Class that represents a poisson distribution """
    def __init__(self, data=None, lambtha=1.):
        """
        Constructor function

        Args:
            data (list, optional): List of the data to be used to estimate
                the distribution. Defaults to None.
            lambtha (_type_, optional): The expected number of occurences
                in a given time frame (the mean number of successes).
                Defaults to 1..

        Raises:
            ValueError: If lambtha is not a positive value
            TypeError: If data is not a list
            ValueError: If data doesn't contain multiple values
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
                self.lambtha = sum(data) / len(data)
