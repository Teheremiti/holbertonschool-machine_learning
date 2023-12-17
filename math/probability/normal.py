#!/usr/bin/env python3
""" Normal class """


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
                raise TypeError('data must a list')
            elif len(data) < 2: 
                raise ValueError('data must contain multiple values')
            else:
                self.mean = sum(data) / len(data)
                var = sum([(ele - self.mean)**2 for ele in data]) / len(data)
                self.stddev = var**0.5
