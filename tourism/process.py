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
        c = 0
        for t1, t2 in zip(train_correct[1:], train_correct):
            res = abs(t1-t2)
            if not numpy.isnan(res):
                error += res
                c += 1
        return error/c


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


def SeriesList2Mat(series_list):

    def data_yield(series_list):
        for series in series_list:
            for val in series.data:
                yield val

    mat = numpy.fromiter(data_yield(series_list), numpy.float)
    mat = numpy.reshape(mat, (-1, len(series_list)))

    return mat



class TourismSeries():

    def __init__(self, times, data, ID):

        self.times = times
        self.data = data
        self.ID = ID
        self.predicted_data = numpy.ones_like(data)*numpy.nan
        self.coef = None

    def PredictData(self, series_list, linkage, train_rows = 39):

        other_vals = SeriesList2Mat(series_list)
        contib = numpy.nansum(other_vals*self.data, axis = 1)
        unaccounted_data = self.data-contrib

        self.coef = PolyFit(self.times[:train_rows], unaccounted_data[:train_rows,:])

        res = numpy.polyval(self.coef,  self.times)

        self.predicted_data = res+contrib



class TourismModel():

    def __init__(self):

        self.series_list = []
        self.linkage_matrix = None






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
    for col in range(train_data.shape[1]):
        poly_coef.append(PolyFit(train_time, train_data[:,col]))

    guess_val = []
    real_val = []
    train_vals = []
    test_vals = []

    for col, coef in zip(xrange(test_data.shape[1]), poly_coef):

        train_guess = numpy.polyval(coef, train_time)
        test_guess = numpy.polyval(coef, test_time)

        train_nval, test_nval = CalculateMASE(train_guess, train_data[:,col],
                                              test_guess, test_data[:,col])

        print train_nval, test_nval
        train_vals.append(train_nval)
        test_vals.append(test_nval)
    train_vals = numpy.fromiter(train_vals, numpy.float)
    test_vals = numpy.fromiter(test_vals, numpy.float)

    print test_vals.mean()




    






