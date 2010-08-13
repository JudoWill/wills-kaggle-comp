from __future__ import division
import csv
import numpy
from operator import itemgetter
from itertools import imap, chain


def CalculateMASE(test_in, cor_in):
    """Calculates the Mean Absolute Scaled Error"""

    try:
        abs_error = abs(test_in - cor_in)
    except TypeError:
        abs_error = abs(numpy.fromiter(test_in, numpy.float) - numpy.fromiter(cor_in, numpy.float))

    mask = ~numpy.isnan(abs_error)
    mae = abs_error[mask].mean()

    #scaled_errors = abs_error[mask]/mae
    #mase = scaled_errors.mean()
    return mae


def FloatConv(string):
    """Converts to string2float and returns Nan if not possible"""
    try:
        v = float(string)
    except ValueError:
        v = numpy.NaN
    return v


def PolyFit(t, x, deg = 4):
    """Fits polynomial with NaNs"""

    v = ~numpy.isnan(x)
    if sum(v) == 0:
        return numpy.array([0])
    use_deg = min(deg, numpy.sum(v)-1)
    return numpy.polyfit(t[v], x[v], use_deg)


if __name__ == '__main__':


    with open('initial-data/tourism_data.csv') as handle:
        keys = map(lambda x:"Y%i"%x, range(1,518))
        string_data = imap(itemgetter(*keys), csv.DictReader(handle))
        data = imap(FloatConv, chain.from_iterable(string_data))
        all_data = numpy.fromiter(data, numpy.float)

    all_data = numpy.reshape(all_data, (-1, len(keys)))
    times = numpy.arange(all_data.shape[0])

    train_time = times[0:7]
    train_data = all_data[0:7,:]

    test_time = times[8:]
    test_data = all_data[8:,:]

    poly_coef = []
    for col in range(test_data.shape[1]):
        coef = PolyFit(train_time, train_data[:, col])
        poly_coef.append(coef)

    guess_val = []
    real_val = []

    for col, coef in zip(xrange(test_data.shape[1]), poly_coef):
        guess_val.extend(numpy.polyval(coef, test_time))
        real_val.extend(test_data[:,col])


    mase = CalculateMASE(guess_val, real_val)
    print mase






