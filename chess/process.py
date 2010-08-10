from __future__ import division
import csv
from math import sqrt
from collections import defaultdict
from operator import attrgetter
from types import ListType
import random, optparse


class Player():
    def __init__(self, pid, rank = 0):
        self.rank = rank
        self.pid = pid
        self.wins = list()
        self.loses = list()

    def match(self, bp, score):
        """Updates the rank based on the rank of Black-Player and the score"""

        if score > 0 and bp.rank > self.rank:
        #already correct
            return
        elif score < 0 and self.rank > bp.rank:
        #already correct
            return
        else:
            self.rank = max(self.rank+(score - bp.rank - self.rank)/2, 0)

        if score < 0:
            self.wins.append(bp.pid)
            bp.loses.append(self.pid)
        elif score > 0:
            self.loses.append(bp.pid)
            bp.wins.append(self.pid)



class PlayerDict():

    def __init__(self, default_rank = 0):
        self.pdict = {}
        self.default_rank = default_rank
        self.match_list = []
        self.ranked_list = []
        self.score = None

    def __getitem__(self, pid):

        if pid in self.pdict:
            return self.pdict[pid]
        else:
            nplayer = Player(pid, rank = self.default_rank)
            self.pdict[pid] = nplayer
            return nplayer

    def GetMatchScore(self, wid, bid, month):

        p1 = self[wid]
        p2 = self[bid]

        return OutTreatScore(p1.rank - p2.rank)

    def PerformMatch(self, wid, bid, score):

        self[wid].match(self[bid], score)
        self.match_list += [(wid, bid, score)]

    def PlayerCmp(self, wplayer, bplayer):

        value = 0

        bplayer_wins = len([i for i in bplayer.wins if i == wplayer.pid])
        wplayer_wins = len([i for i in wplayer.wins if i == bplayer.pid])

        try:
            win_adjust = (bplayer_wins - wplayer_wins)/(bplayer_wins+wplayer_wins)
        except ZeroDivisionError:
            win_adjust = 0
        value += win_adjust

        rank_adjust = wplayer.rank - bplayer.rank
        value += rank_adjust

        if value > 0:
            return 1
        elif value < 0:
            return -1
        else:
            return 0

    def SetRanks(self, num_iter = 5):

        self.ranked_list = self.pdict.values()
        self.ranked_list.sort(key = attrgetter('rank'))

        for _ in range(num_iter):
            self.ranked_list.sort(cmp = self.PlayerCmp)
            self._adjust_ranks()

    def _adjust_ranks(self):

        c=0
        while c < len(self.ranked_list)-2:
            it1 = self.ranked_list[c]
            it2 = self.ranked_list[c+1]

            if it2.rank > it1.rank:
                adj = (it2.rank-it1.rank)/2
                it2.rank -= adj
                it1.rank += adj
            c += 1

    def EvaluateModel(self, csv_gen):

        self.score = EvaluateModel(self, csv_gen)
        print self.score

def TrainTestInds(nitems, frac = 0.6):
    train = []
    test = []
    for item in nitems:
        if random.random() < frac:
            train.append(item)
        else:
            test.append(item)

    return train, test


def TrainModel(csv_gen, num_models = 10, default_rank = 0):
    """Trains the model based on receiving a 'csv-generator' from the rows"""

    model_list = []
    for i in range(num_models):

        player_dict = PlayerDict(default_rank = default_rank)
        train, test = TrainTestInds(csv_gen)
        for row in train:
            p1 = int(row["White Player #"])
            p2 = int(row["Black Player #"])
            s = InTreatScore(float(row["Score"]))
            player_dict.PerformMatch(p1, p2, s)

        player_dict.SetRanks()
        player_dict.EvaluateModel(test)
        model_list.append(player_dict)


    return model_list


def BayesComb(prior, models, w, b, month):
    val = 0
    model_scores = map(lambda x: x.score, models)
    max_score = sum(model_scores)
    props = map(lambda x: x/max_score, model_scores)
    for model, evidence in zip(models, props):
        val += model.GetMatchScore(w,b, month)*evidence
    return val

def EvaluateModel(model_dict, csv_gen):
    """Performs the model evaluation based on the Kaggle rules"""



    islist = type(model_dict) is ListType
    predicted_agg = defaultdict(float)
    correct_agg = defaultdict(float)

    for row in csv_gen:
        p1 = int(row["White Player #"])
        p2 = int(row["Black Player #"])
        s = float(row["Score"])
        m = row["Month #"]
        if islist:
            score = BayesComb(0.5, model_dict, p1, p2, m)
        else:
            score = model_dict.GetMatchScore(p1, p2, m)


        predicted_agg[(m, p1)] += score
        predicted_agg[(m, p2)] += 1-score

        correct_agg[(m, p1)] += s
        correct_agg[(m, p2)] += 1-s
    
    mse = 0.0
    for key in correct_agg.keys():
        mse += (predicted_agg[key] - correct_agg[key])**2

    return sqrt(mse/len(correct_agg.keys()))

def WritePrediction(model_dict, csv_gen, out_handle):

    out_handle.write('"Month #","White Player #","Black Player #","Score"\n')
    for row in csv_gen:
        p1 = int(row["White Player #"])
        p2 = int(row["Black Player #"])
        m = row['Month #']
        score = BayesComb(0.5, model_dict, p1, p2, m)
        out_handle.write('%s,%i,%i,%f\n' % (m, p1, p2, score))

def InTreatScore(inscore):
    return (inscore - 0.5)*2

def OutTreatScore(inscore):
    return (inscore/2)+0.5


if __name__ == '__main__':



    parser = optparse.OptionParser()
    parser.add_option('-r', '--run', dest = 'run',
                      action = 'store_true', default = False)
    (options, args) = parser.parse_args()


    INIT_DATA_FILE = 'initial-data/training_data.csv'
    TEST_DATA_FILE = 'initial-data/test_data.csv'
    OUTFILE = 'results.csv'


    with open(INIT_DATA_FILE) as handle:
        train_rows = list(csv.DictReader(handle))

    ntrain = int(0.7*len(train_rows))
    model = TrainModel(train_rows[:ntrain], default_rank = 0.5)


    print 'real val', EvaluateModel(model, train_rows[ntrain+1:])

    if options.run:
        rmodel = TrainModel(train_rows)

        with open(TEST_DATA_FILE) as thandle:
            csv_gen = csv.DictReader(thandle)
            with open(OUTFILE, 'w') as ohandle:
                WritePrediction(rmodel, csv_gen, ohandle)






