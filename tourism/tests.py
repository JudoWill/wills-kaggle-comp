import nose.tools
from process import *
import csv




def testMASE():
    """Test Mean Absolute Scaled Error function"""

    #get the start-colums from the example file
    start_cols = (2, 9, 15, 21, 28)

    with open('initial-data/mase_example.csv') as handle:
        rows = list(csv.reader(handle))
    print 'hre;', len(rows)
    for start_col in start_cols:            
        print 'start', start_col
        
        true_train = [float(x[1]) for x in rows[3:26]]
        guessed_train = [float(x[start_col]) for x in rows[3:26]]
        true_test = [float(x[1]) for x in rows[26:38]]
        guess_test = [float(x[start_col]) for x in rows[26:38]]
        try:
            test_mase = float(rows[38][start_col+6])
            train_mase = float(rows[25][start_col+6])
        except ValueError:
            test_mase = float(rows[38][start_col+4])
            train_mase = float(rows[25][start_col+4])


        yield CheckMASE, guessed_train, true_train, guess_test, \
                true_test, train_mase, test_mase

def CheckMASE(*args):
    train_mase, test_mase = CalculateMASE(*args[:4])
    nose.tools.assert_almost_equals(train_mase, args[4], 1)
    nose.tools.assert_almost_equals(test_mase, args[5], 1)



def testNanmean():
    """Test the nanmean function"""

    tests = [(numpy.array([1.0, 2.0, numpy.nan]).transpose(), 1.5),
            (numpy.array([1.0, 2.0, 3.0]), 2.0),
            (numpy.array([0.0, 0.0, numpy.nan]), 0)]
    
    for test, val in tests:
        yield CheckNanmean, test, val


def CheckNanmean(test, val):    
    res = nanmean(test)
    nose.tools.assert_almost_equals(res, val, 3)
    