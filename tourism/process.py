from __future__ import division
import numpy


def CalculateMASE(test_in, cor_in):

    try:
        abs_error = abs(test_in - cor_in)
    except TypeError:
        abs_error = abs(numpy.fromiter(test_in, numpy.float) - numpy.fromiter(cor_in, numpy.float))

    mae = abs_error.mean()

    scaled_errors = abs_error/mae
    return scaled_errors.mean()
