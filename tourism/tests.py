import nose.tools
from process import *
from csv import DictReader


def testMASE():
    """Test Mean Absolute Scaled Error function"""

    tes = []
    act = []
    with open('initial-data/mase_example.csv') as handle:
        for row in DictReader(handle):
            tes.append(row['Guess'])
            act.append(row['Actual'])
    mase = CalculateMASE(tes, act)
    nose.tools.assert_almost_equals(mase, 1.00, 2)