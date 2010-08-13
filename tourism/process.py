from __future__ import division
import csv
import numpy
from operator import itemgetter
from itertools import imap, chain


def CalculateMASE(test_in, cor_in):

    try:
        abs_error = abs(test_in - cor_in)
    except TypeError:
        abs_error = abs(numpy.fromiter(test_in, numpy.float) - numpy.fromiter(cor_in, numpy.float))

    mae = abs_error.mean()

    scaled_errors = abs_error/mae
    return scaled_errors.mean()


def FloatConv(string):
    try:
        v = float(string)
    except ValueError:
        v = numpy.NaN
    return v


if __name__ == '__main__':


    with open('initial-data/tourism_data.csv') as handle:
        keys = map(lambda x:"Y%i"%x, range(1,518))
        string_data = imap(itemgetter(*keys), csv.DictReader(handle))
        data = imap(FloatConv, chain.from_iterable(string_data))
        test_data = numpy.fromiter(data, numpy.float)

    test_data = numpy.reshape(test_data, (-1, len(keys)))

