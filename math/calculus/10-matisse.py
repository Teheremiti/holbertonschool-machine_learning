#!/usr/bin/env python3
""" poly_derivative function """


def poly_derivative(poly):
    """
    Returns the derivative of a polynomial

    Args:
        poly (list): The polynomial as a list of coefficients. The index in the
        list represents the power of x.

    Returns:
        list: The list of coefficients corresponding to the derivative of poly
    """
    if not poly\
       or not isinstance(poly, list)\
       or any([not isinstance(ele, int) for ele in poly]):
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power in range(1, len(poly)):
        coef = poly[power]
        derivative.append(power * coef)
    return derivative
