from __future__ import division
import csv
import numpy
from operator import itemgetter
from itertools import imap, chain
from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators




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
            for val in series.real_data:
                yield val

    mat = numpy.fromiter(data_yield(series_list), numpy.float)
    mat = numpy.reshape(mat, (-1, len(series_list)))

    return mat



class TourismSeries():

    def __init__(self, times, data, ID):

        self.times = times
        self.real_data = data
        self.ID = ID
        self.predicted_data = numpy.ones_like(data)*numpy.nan
        self.coef = None
        self.train_rows = None

    def PredictData(self, other_vals, linkage, train_rows = 39):

        
        self.train_rows = train_rows

        contrib = numpy.nansum(other_vals * linkage, axis = 1)

        ex = numpy.hstack((0, -contrib[1:]))
        ex *= ~numpy.isnan(ex)

        unaccounted_data = self.real_data - ex
        print unaccounted_data

        self.coef = PolyFit(self.times[:train_rows], unaccounted_data[:train_rows])

        res = numpy.polyval(self.coef,  self.times)

        self.predicted_data = res+contrib


    def EvaluateSeries(self):

        mase = CalculateMASE(self.predicted_data[:self.train_rows],
                             self.real_data[:self.train_rows],
                             self.predicted_data[self.train_rows+1:],
                             self.real_data[self.train_rows+1:],)

        return mase


class TourismModel():

    def __init__(self, times, train_nums):

        self.series_list = []
        self.series_mat = None
        self.contrib = None
        self.linkage_matrix = None
        self.times = times
        self.nums = train_nums


    def EvalFromParam(self, linkage_array):
        link = numpy.fromiter(linkage_array,
                              numpy.float).reshape((len(self.series_list),
                                                    len(self.series_list)))
        score = 0
        for col, series in zip(range(len(self.series_list)), self.series_list):
            series.PredictData(self.series_mat, link[:, col],
                                train_rows = self.nums)
            score += series.EvaluateSeries()[1]
        print score / len(self.series_list)
        return score / len(self.series_list)



    def DoEvolution(self):

        self.series_mat = SeriesList2Mat(self.series_list)


        genome = G1DList.G1DList(len(self.series_list)**2)
        genome.initializator.set(Initializators.G1DListInitializatorReal)
        genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
        genome.setParams(rangemin = -1, rangemax = 1,
                         gauss_mu = 0, gauss_sigma = 0.00001 )
        genome.evaluator.set(self.EvalFromParam)
        
        ga = GSimpleGA.GSimpleGA(genome)

        ga.evolve(freq_stats = 1)

        self.linkage_matrix = ga.bestIndividual()







if __name__ == '__main__':


    with open('initial-data/tourism_data.csv') as handle:
        keys = map(lambda x:"Y%i"%x, range(1,518))
        string_data = imap(itemgetter(*keys), csv.DictReader(handle))
        data = imap(FloatConv, chain.from_iterable(string_data))
        all_data = numpy.fromiter(data, numpy.float)

    all_data = numpy.reshape(all_data, (-1, len(keys)))
    times = numpy.arange(all_data.shape[0])
    nums = 39

    model = TourismModel(times, nums)
    for col in xrange(len(keys)):
        model.series_list.append(TourismSeries(times, all_data[:,col],
                                               keys[col]))

    model.DoEvolution()





    






