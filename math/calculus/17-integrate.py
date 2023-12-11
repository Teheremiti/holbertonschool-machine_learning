#!/usr/bin/env python3
""" poly_integral function """


def poly_integral(poly, C=0):
    """
    Returns the integral of a polynomial

    Args:
        poly (list): The polynomial as a list of coefficients
        C (int, optional): The integration constant. Defaults to 0.

    Returns:
        list: The integral polynomial as a list
    """
    if not poly\
       or not isinstance(poly, list)\
       or any([not isinstance(ele, int) for ele in poly])\
       or not isinstance(C, int):
        return None

    if poly == [0]:
        return [C]

    integral = [C]
    for power in range(len(poly)):
        coef = poly[power]
        if coef % (power + 1) == 0:
            new_coef = coef // (power + 1)
        else:
            new_coef = coef / (power + 1)
        integral.append(new_coef)
    return integral
