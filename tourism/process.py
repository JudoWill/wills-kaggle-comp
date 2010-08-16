from __future__ import division
import csv
import numpy
from operator import itemgetter
from itertools import imap, chain


def nanmean(v1, axis = None):
    """Averages the non-nan elements."""

    temp = v1
    mask = numpy.isnan(temp)

    temp[mask] = 0
    summed = numpy.sum(temp, axis = axis)

    return summed / numpy.sum(~mask, axis = axis)





def CalculateMASE(train_guess, train_correct, test_guess, test_correct):
    """Calculates the Mean Absolute Scaled Error"""

    def CacluateNaive(train_correct):

        error = 0
        for t1, t2 in zip(train_correct[1:], train_correct):
            error += abs(t1-t2)
        return error/(len(train_correct)-1)


    try:
        abs_error_train = abs(train_guess - train_correct)
        abs_error_test  = abs(test_guess - test_correct)
    except TypeError:
        #if they're the wrong type then convert them accordingly
        train_guess = numpy.fromiter(train_guess, numpy.float)
        train_correct = numpy.fromiter(train_correct, numpy.float)
        test_guess = numpy.fromiter(test_guess, numpy.float)
        test_correct = numpy.fromiter(test_correct, numpy.float)
        abs_error_train = abs(train_guess - train_correct)
        abs_error_test  = abs(test_guess - test_correct)

    naive_scale = CacluateNaive(train_correct)

    train_scaled_errors = abs_error_train/naive_scale
    test_scaled_errors = abs_error_test/naive_scale

    train_mase = nanmean(train_scaled_errors)
    test_mase = nanmean(test_scaled_errors)
    return train_mase, test_mase


def FloatConv(string):
    """Converts to string2float and returns Nan if not possible"""
    try:
        v = float(string)
    except ValueError:
        v = numpy.NaN
    return v


def PolyFit(t, x, deg = 1):
    """Fits polynomial with NaNs"""

    v = ~numpy.isnan(x)
    use_deg = min(deg, numpy.sum(v)-1)
    if sum(v) == 0 or deg < 1:
        return numpy.array([0])

    return numpy.polyfit(t[v], x[v], use_deg)


if __name__ == '__main__':


    with open('initial-data/tourism_data.csv') as handle:
        keys = map(lambda x:"Y%i"%x, range(1,518))
        string_data = imap(itemgetter(*keys), csv.DictReader(handle))
        data = imap(FloatConv, chain.from_iterable(string_data))
        all_data = numpy.fromiter(data, numpy.float)

    all_data = numpy.reshape(all_data, (-1, len(keys)))
    times = numpy.arange(all_data.shape[0])
    tnum = 39

    train_time = times[0:tnum]
    train_data = all_data[0:tnum,:]

    test_time = times[tnum+1:]
    test_data = all_data[tnum+1:,:]

    poly_coef = []
    for col in range(test_data.shape[1]):
        coef = PolyFit(train_time, train_data[:, col])
        poly_coef.append(coef)

    guess_val = []
    real_val = []

    for col, coef in zip(xrange(test_data.shape[1]), poly_coef):
        if len(coef) >= 2:
            g = numpy.polyval(coef, test_time)
            r = test_data[:,col]
            #print r, g
            guess_val.extend(g)
            real_val.extend(r)



    






