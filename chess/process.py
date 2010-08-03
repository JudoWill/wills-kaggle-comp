from __future__ import division
import csv
from math import sqrt
from collections import defaultdict



class Player():
    def __init__(self, rank = 0.5):
        self.rank = rank

    def match(self, bp, score):
        """Updates the rank based on the rank of Black-Player and the score"""

        if score > 0.5 and bp.rank > self.rank:
            #already correct
            return
        elif score < 0.5 and self.rank > bp.rank:
            #already correct
            return
        else:
            self.rank = max(self.rank+(score - bp.rank - self.rank)/2, 0)



    def get_match_score(self, bp):
        """Returns the 'score' of this player vs. Black-Player"""

        return max(0, self.rank-bp.rank)


def TrainModel(csv_gen, default_rank = 0.5):
    """Trains the model based on receiving a 'csv-generator' from the rows"""

    player_dict = defaultdict(lambda : Player(rank = default_rank))

    for row in csv_gen:
        p1 = int(row["White Player #"])
        p2 = int(row["Black Player #"])
        s = float(row["Score"])

        player_dict[p1].match(player_dict[p2], s)

    return player_dict


def EvaluateModel(model_dict, csv_gen):
    """Performs the model evaluation based on the Kaggle rules"""

    predicted_agg = defaultdict(float)
    correct_agg = defaultdict(float)

    for row in csv_gen:
        p1 = int(row["White Player #"])
        p2 = int(row["Black Player #"])
        s = float(row["Score"])
        score = model_dict[p1].get_match_score(model_dict[p2])
        predicted_agg[(row['Month #'], p1, p2)] += score
        predicted_agg[(row['Month #'], p2)] += score

        correct_agg[(row['Month #'], p1)] += s
        correct_agg[(row['Month #'], p2)] += s

    mse = 0.0
    for key in correct_agg.keys():
        mse += (predicted_agg[key] - correct_agg[key])**2

    return sqrt(mse)

def WritePrediction(model_dict, csv_gen, out_handle):

    out_handle.write('"Month #","White Player #","Black Player #","Score"\n')
    for row in csv_gen:
        p1 = int(row["White Player #"])
        p2 = int(row["Black Player #"])
        m = row['Month #']
        score = model_dict[p1].get_match_score(model_dict[p2])
        out_handle.write('%s,%i,%i,%f\n' % (m, p1, p2, score))



if __name__ == '__main__':


    INIT_DATA_FILE = 'initial-data/training_data.csv'
    TEST_DATA_FILE = 'initial-data/test_data.csv'
    OUTFILE = 'results.csv'


    with open(INIT_DATA_FILE) as handle:
        train_rows = list(csv.DictReader(handle))

    ntrain = int(0.7*len(train_rows))
    
    model = TrainModel(train_rows[:ntrain], default_rank = 0.1)
    
    print EvaluateModel(model, train_rows[ntrain+1:])

    rmodel = TrainModel(train_rows)

    with open(TEST_DATA_FILE) as thandle:
        csv_gen = csv.DictReader(thandle)
        with open(OUTFILE, 'w') as ohandle:
            WritePrediction(rmodel, csv_gen, ohandle)






