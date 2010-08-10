import nose.tools
import csv
from process import *

def testErrorFunc():

    with open('initial-data/prediction_check.csv') as handle:
        csv_list = list(csv.DictReader(handle))
    print csv_list
    class Mock():

        def __init__(self, csv_list):

            self.mock_out = {}
            for row in csv_list:
                p1 = int(row["White Player #"])
                p2 = int(row["Black Player #"])
                s = float(row["Predicted"])
                m = row["Month #"]
                self.mock_out[(p1, p2, m)] = s

        def GetMatchScore(self, p1, p2, m):

            return self.mock_out[(p1, p2, m)]

    mocked = Mock(csv_list)
    res = EvaluateModel(mocked, csv_list)

    nose.tools.assert_almost_equals(res, 0.68, 2)