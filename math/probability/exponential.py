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

        Raises:
            ValueError: If lambtha is less or equal to zero, or if data doesn't
                contain at least two values.
            TypeError: If data is not a list.
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

    def pdf(self, x):
        """
        Calculates the exponential PDF for a given time period.

        Args:
            x (int/float): The time period

        Returns:
            float: The exponential PDF value for x
        """
        if x < 0:
            return 0
        e = 2.7182818285
        µ = self.lambtha
        return µ * (e ** -(µ * x))

    def cdf(self, x):
        """
        Calculates the exponential CDF for a given time period.

        Args:
            x (int/float): The time period

        Returns:
            float: The exponential CDF value for x
        """
        if x < 0:
            return 0
        e = 2.7182818285
        µ = self.lambtha
        return 1 - e**(-µ * x)
