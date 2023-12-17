#!/usr/bin/env python3
""" Binonial class """


class Binomial:
    """ Class that represents the Binomial distribution """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Constructor method for the Binomial class.

        Args:
            data (list, optional): The list of the data to be used to estimate
                the binomial distribution. Defaults to None.
            n (int, optional): The number of Bernoulli trials. Defaults to 1.
            p (float, optional): The probablity of a success. Defaults to 0.5.

        Raises:
        ValueError: If n is less or equal to zero, or p is not between 0 and 1,
            or if data doesn't contain at least two values.
        TypeError: If data is not a list.
        """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            elif not 0 < p < 1:
                raise ValueError('p must be greater than 0 and less than 1')
            else:
                self.n = n
                self.p = p

        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.p = (sum(data) / len(data)) / 50
                self.n = round(len(data) / 2)
