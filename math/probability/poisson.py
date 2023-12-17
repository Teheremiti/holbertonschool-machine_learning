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

    def pmf(self, k):
        """
        Calculates the PMF for a guven number of successes

        Args:
            k (int): Number of successes.

        Returns:
            float: The PMF value for k.
        """
        if k <= 0:
            return 0

        elif not isinstance(k, int):
            k = int(k)

        e = 2.7182818285
        µ = self.lambtha
        kfacto = 1
        for i in range(1, k + 1):
            kfacto *= i

        return (e ** -µ) * (µ ** k) / kfacto

    def cdf(self, k):
        """
        Calculates the CDF for a given number of succeses

        Args:
            k (int): The number of successes

        Returns:
            float: The CDF value for k successes
        """
        def ifacto(i):
            """ Computes the factorial for i """
            ifacto = 1
            for j in range(1, i + 1):
                ifacto *= j
            return ifacto

        if k <= 0:
            return 0

        elif not isinstance(k, int):
            k = int(k)

        µ = self.lambtha
        sum = 0
        for i in range(k + 1):
            sum += (µ ** i) / ifacto(i)

        e = 2.7182818285
        return (e ** -µ) * sum
