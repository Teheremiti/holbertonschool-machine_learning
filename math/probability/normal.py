#!/usr/bin/env python3
""" Normal class """
π = 3.1415926536
e = 2.7182818285


class Normal:
    """ Class that represents a normal distribution """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Constructor method for the Normal class.

        Args:
            data (list, optional): List of the data to be used to estimate
                the distribution. Defaults to None.
            mean (float, optional): The mean of the distribution.
                Defaults to 0..
            stddev (float, optional): The standard deviation of the
                distribution. Defaults to 1..

        Raises:
            ValueError: If stddev if less or equal to zero, or if data doesn't
                contain at least two values.
            TypeError: If data is not a list
        """
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.mean = sum(data) / len(data)
                var = sum([(ele - self.mean)**2 for ele in data]) / len(data)
                self.stddev = var**0.5

    def z_score(self, x):
        """
        Calculates the z_score for a given x-value.

        Args:
            x (int/float): The x-value

        Returns:
            float: The z-score for x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x_score for a given z-score.

        Args:
            z (int/float): The z-score

        Returns:
            float: The x-value for z
        """
        return self.stddev * z + self.mean

    def pdf(self, x):
        """
        Calculates the Normal PDF for a given x-value

        Args:
            x (int/float): The x-value

        Returns:
            float: The Normal PDF value for x
        """
        µ = self.mean
        sdev = self.stddev
        return e**(-0.5 * ((x - µ) / sdev)**2) / (sdev * (2 * π)**0.5)

    def cdf(self, x):
        """
        Calculates the Normal CDF for a given x-value

        Args:
            x (int/float): The x-value

        Returns:
            float: The Normal CDF for x
        """
        µ = self.mean
        sdev = self.stddev

        def erf(x):
            """ Computes the error function on x """
            series = x - ((x**3) / 3) + ((x**5) / 10)\
                - ((x**7) / 42) + ((x**9) / 216)
            return (2 / π**0.5) * series

        return (1 + erf((x - µ) / (sdev * 2**0.5))) / 2
