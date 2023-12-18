#!/usr/bin/env python3
""" Binonial class """


def facto(n):
    """ Factorial function """
    nfacto = 1
    for i in range(1, n + 1):
        nfacto *= i
    return nfacto


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
                µ = sum(data) / len(data)
                variance = sum((x - µ)**2 for x in data) / len(data)
                self.p = 1 - (variance / µ)
                self.n = round(µ / self.p)
                self.p = µ / self.n

    def pmf(self, k):
        """
        Calculates the binomial PMF for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The binomial PMF for k successes.
        """

        if k < 0:
            return 0
        elif not isinstance(k, int):
            k = int(k)

        n = self.n
        p = self.p
        bi_coef = facto(n) / (facto(k) * facto(n - k))
        return bi_coef * p**k * (1 - p)**(n - k)

    def cdf(self, k):
        """
        Calculates the binomial CDF for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The binomial CDF for k successes.
        """
        if k < 0:
            return 0
        elif not isinstance(k, int):
            k = int(k)
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
