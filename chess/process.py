from __future__ import division
import csv
from math import sqrt, log, exp
from collections import defaultdict, deque
from operator import attrgetter
from types import ListType
import random, optparse
from itertools import combinations, izip, product, repeat, imap, chain
from multiprocessing import Pool
from bisect import insort, bisect
import numpy


class Player():
    def __init__(self, pid, rank = 0):
        self.rank = rank
        self.pid = pid
        self.wins = list()
        self.loses = list()

    def match(self, bp, score, weight = 1):
        """Updates the rank based on the rank of Black-Player and the score"""

        if score > 0 and bp.rank > self.rank:
        #already correct
            return
        elif score < 0 and self.rank > bp.rank:
        #already correct
            return
        else:
            w = log(weight)
            nw = 1/(1+exp(-w))
            move = score - self.rank - bp.rank
            scaled_move = w*nw*move
            #print 'move', self.rank, bp.rank, weight, w, nw, move, scaled_move
            self.rank += scaled_move
            bp.rank -= scaled_move


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
        self.all_scores = None
        self.bins = None
        self.tot = None

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
        score = p1.rank - p2.rank

        return Logit(score)

    def GenerateLikelihood(self):

        def Score():
            for p1, p2 in combinations(self.pdict.itervalues(), 2):
                yield abs(p1.rank-p2.rank)


        bin_scores, self.bins = numpy.histogram(numpy.fromiter(Score(), numpy.float),
                                                bins = numpy.linspace(0,1,num = 50))

        self.tot = numpy.sum(bin_scores)
        self.all_scores = bin_scores.cumsum()


    def GetLikelihood(self, val):

        if self.all_scores is None:
            self.GenerateLikelihood()

        spot = numpy.searchsorted(self.bins, val)

        return self.all_scores[spot-1]/self.tot


    def PerformMatch(self, wid, bid, score, weight = 1):

        self[wid].match(self[bid], score, weight = weight)
        self.match_list += [(wid, bid, score, weight)]

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

    def EvaluateModel(self, csv_gen, check_vote = False):
        self.rmse = EvaluateModel(self, csv_gen, check_vote = check_vote)
        self.score = 0.5+1/(1+exp(self.rmse))
        #print self.rmse, self.score

def Logit(v):
    try:
        r = 1/(1+exp(-v))
    except OverflowError:
        if v > 0:
            r = 1
        else:
            r = 0
    return r

def TrainTestInds(nitems, frac = 0.7):
    train = []
    test = []
    for item in nitems:
        if random.random() < frac:
            train.append(item)
        else:
            test.append(item)

    return train, test


def TrainModel(csv_gen, **kwargs):
    """Trains the model based on receiving a 'csv-generator' from the rows"""

    def WeightMatches(models, csv_gen, check_vote = False):
        for row in csv_gen:
            p1 = int(row["White Player #"])
            p2 = int(row["Black Player #"])
            s = float(row["Score"])
            m = row["Month #"]
            t_score = BayesComb(0.5, models, p1, p2, m,
                                check_vote = check_vote)

            row['weight'] = 10**abs(s-t_score)

            yield row

    def TrainSingle(train, test, default_rank = 0.5, check_vote = True):
        player_dict = PlayerDict(default_rank = default_rank)
        for row in train:
            p1 = int(row["White Player #"])
            p2 = int(row["Black Player #"])
            s = InTreatScore(float(row["Score"]))
            weight = row.get('weight', 1)

            player_dict.PerformMatch(p1, p2, s, weight = weight)

        player_dict.SetRanks()
        player_dict.EvaluateModel(test, check_vote = check_vote)
        #player_dict.GenerateLikelihood()

        return player_dict
        


    default_rank = kwargs.get('default_rank', 0)
    seed_frac = kwargs.get('seed_frac', 0.3)
    rep_frac = kwargs.get('rep_frac', 0.2)
    check_train = kwargs.get('check_train', False)
    check_indiv = kwargs.get('check_indiv', True)

    model_list = []

    train, test = TrainTestInds(csv_gen, frac = seed_frac)
    model_list.append(TrainSingle(train, test, check_vote = check_indiv,
                                  default_rank = default_rank))
    
    while len(test) > 1000:
    #for i in range(num_models):
        train, test = TrainTestInds(test, frac = rep_frac)
        gen = WeightMatches(model_list, train, check_vote = check_train)
        model_list.append(TrainSingle(gen, test, check_vote = check_indiv,
                                      default_rank = default_rank))

    train, test = TrainTestInds(test, frac = rep_frac)
    gen = WeightMatches(model_list, train, check_vote = check_train)
    model_list.append(TrainSingle(gen, test, check_vote = check_indiv,
                                      default_rank = default_rank))

    return model_list


def BayesComb(prior, models, w, b, month, check_vote = False):



    if check_vote:
        pos_vote = []
        neg_vote = []

        for model in models:
            score = model.GetMatchScore(w,b, month)
            if score >= 0.5:
                pos_vote += [model]
            elif score < 0.5:
                neg_vote += [model]

        models = max(neg_vote, pos_vote)
        

    scores = []
    likes = []
    for model in models:
        scores += [model.GetMatchScore(w,b, month)]
        likes += [model.score]

    tot = sum(likes)
    evi = map(lambda x: x/tot, likes)
    res = sum(map(lambda x,y: x*y, scores, evi))


    return res

def EvaluateModel(model_dict, csv_gen, check_vote = False):
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
            score = BayesComb(0.5, model_dict, p1, p2, m, check_vote = check_vote)
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

def Linker(input):
    group, fields, train_rows, test_rows = input
    pdict = dict(zip(chain.from_iterable(fields), group))

    rmodel = TrainModel(chain.from_iterable(train_rows), **pdict)
    val = EvaluateModel(rmodel, chain.from_iterable(test_rows))

    return pdict, val

    
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
                  'check_train', 'check_indiv')
        default_ranks = map(lambda x: x/10, range(10))
        seed_fracs = map(lambda x: x/10, range(1,8))
        rep_fracs = map(lambda x: x/10, range(1,8))
        check_trains = (True, False)
        check_indivs = (True, False)
        bval = 100
        pool = Pool(processes = 3)
        group_gen = product(default_ranks, seed_fracs, rep_fracs,
                            check_trains, check_indivs)
        map_iter = izip(group_gen, repeat((fields,)),
                        repeat((train_rows[:ntrain],)),
                        repeat((train_rows[ntrain+1:],)))
        for pdict, val in pool.imap(Linker, map_iter, chunksize = 10):

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






