from __future__ import division
import csv
from operator import itemgetter
import  optparse
from itertools import combinations, izip, product, repeat, imap, chain
import numpy, yaml
import scipy.optimize
import os.path

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
    parser.add_option('-s', '--store', dest = 'store',
                      type = 'string', default = 'optimized.yaml')
    parser.add_option('', '--storeiter', dest = 'storeiter',
                      action = 'store_true', default = True)
    parser.add_option('-i', '--iterations', default = 10, type = 'int',
                      dest = 'iterations')
    (options, args) = parser.parse_args()


    INIT_DATA_FILE = 'initial-data/training_data.csv'
    TEST_DATA_FILE = 'initial-data/test_data.csv'
    OUTFILE = 'results.csv'


    with open(INIT_DATA_FILE) as handle:
        train_rows = list(csv.DictReader(handle))
    train, test = TrainTestInds(train_rows, frac = 0.6)

    if not options.run and not options.optimize:
        #runs off default values
        model = TrainModel(train, default_rank = 0.5)
        val =  EvaluateModel(model, test)
        print 'real val ', val

    cdict = {}
    bdict = None
    if options.optimize or options.run:
        fields = ('default_rank', 'seed_frac', 'rep_frac',
                  'prior', 'prior_score',
                  'check_train', 'check_indiv')

        check_trains = (True, False)
        check_indivs = (True, False)
        bval = 100

        gdict = {}
        start = 0
        if options.storeiter and os.path.exists('restart.yaml'):
            with open('restart.yaml') as handle:
                resdict = yaml.load(handle)
                start = resdict['start']
                gdict = resdict['results']

        for i in range(start, options.iterations):
            train, test = TrainTestInds(train_rows, frac = 0.6)
            for check_train, check_indiv in product(check_trains, check_indivs):
                cdict = gdict.get((check_train, check_indiv), {'val':100,
                                                               'init':[0.5, 0.5, 0.5, 0.5, 0.5]})
                out = scipy.optimize.anneal(ObjFun, cdict['init'],
                                          T0 = 1.2, lower = [0,0,0,0,0],
                                          upper = [1,1,1,1,1], dwell = 10,
                                          args = (fields, train, test,
                                                  check_train, check_indiv),
                                          full_output = True, maxeval = 100)
                print 'finished annealing:', out

                x, val, retval, T, feval, iters, accept = out

                pdict = dict(zip(fields, list(iter(x)) + [check_train, check_indiv]))
                pdict['val'] = val
                pdict['init'] = x

                if pdict['val'] < cdict['val']:
                    gdict[(check_train, check_indiv)] = pdict
                    print 'got better: ', pdict
            if options.storeiter:
                print 'save restart point'
                resdict = {'start':i+1, 'results':gdict}
            
        bdict = min(gdict.values(), key = itemgetter('val'))
        if options.store:
            with open(options.store, 'w') as handle:
                yaml.dump(bdict, handle)



    if options.run:
        if bdict is None:
            with open(options.store) as handle:
                bdict = yaml.load(handle)

        rmodel = TrainModel(train_rows, **bdict)

        with open(TEST_DATA_FILE) as thandle:
            csv_gen = csv.DictReader(thandle)
            with open(OUTFILE, 'w') as ohandle:
                WritePrediction(rmodel, csv_gen, ohandle)






