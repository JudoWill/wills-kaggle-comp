from __future__ import division
import csv
from math import sqrt, log, exp
from collections import defaultdict, deque
from operator import attrgetter
from types import ListType
import random, optparse
from itertools import combinations, izip, product, repeat, imap, chain
import numpy
import scipy.optimize

from PlayerClass import *
from utils import *
















def WritePrediction(model_dict, csv_gen, out_handle):

    out_handle.write('"Month #","White Player #","Black Player #","Score"\n')
    for row in csv_gen:
        p1 = int(row["White Player #"])
        p2 = int(row["Black Player #"])
        m = row['Month #']
        score = BayesComb(0.5, model_dict, p1, p2, m)
        out_handle.write('%s,%i,%i,%f\n' % (m, p1, p2, score))



def ObjFun(xtest, fields, train_rows, test_rows, check_train, check_indiv):
    if numpy.any(xtest<0) or numpy.any(xtest>1):
        print 'here'
        return numpy.Inf

    pdict = dict(zip(fields, list(iter(xtest)) + [check_train, check_indiv]))

    rmodel = TrainModel(train_rows, **pdict)
    val = EvaluateModel(rmodel, test_rows)
    print xtest, val
    return val

    
if __name__ == '__main__':



    parser = optparse.OptionParser()
    parser.add_option('-r', '--run', dest = 'run',
                      action = 'store_true', default = False)
    parser.add_option('-o', '--optimize', dest = 'optimize',
                      action = 'store_true', default = False)
    (options, args) = parser.parse_args()


    INIT_DATA_FILE = 'initial-data/training_data.csv'
    TEST_DATA_FILE = 'initial-data/test_data.csv'
    OUTFILE = 'results.csv'


    with open(INIT_DATA_FILE) as handle:
        train_rows = list(csv.DictReader(handle))
    ntrain = int(0.8*len(train_rows))

    if not options.run and not options.optimize:
        #runs off default values
        model = TrainModel(train_rows[:ntrain], default_rank = 0.5)
        val =  EvaluateModel(model, train_rows[ntrain+1:])
        print 'real val ', val

    best_dict = {}
    if options.optimize or options.run:
        fields = ('default_rank', 'seed_frac', 'rep_frac',
                  'prior', 'prior_score',
                  'check_train', 'check_indiv')

        check_trains = (True, False)
        check_indivs = (True, False)
        bval = 100


        for check_train, check_indiv in product(check_trains, check_indivs):

            out = scipy.optimize.anneal(ObjFun,[0.5, 0.5, 0.5, 0.5, 0.5],
                                      upper = numpy.array([1, 1, 1, 1, 1]),
                                      lower = numpy.array([0, 0, 0, 0, 0]),
                                      T0 = 1.2,
                                      args = (fields, train_rows[:ntrain],
                                              train_rows[ntrain+1:],
                                              check_train, check_indiv),
                                      full_output = True, maxeval = 500)
            print 'finished annealing:', out

            x, val, retval, T, feval, iters, accept = out

            pdict = dict(zip(fields, list(iter(x)) + [check_train, check_indiv]))

            if val < bval:
                best_dict = pdict
                bval = val
                print 'checked', pdict
                print 'real val ', val


    if options.run:
        rmodel = TrainModel(train_rows, **best_dict)

        with open(TEST_DATA_FILE) as thandle:
            csv_gen = csv.DictReader(thandle)
            with open(OUTFILE, 'w') as ohandle:
                WritePrediction(rmodel, csv_gen, ohandle)






