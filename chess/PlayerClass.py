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

from utils import *


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
            try:
                nw = 1/(1+exp(-w))
            except OverflowError:
                if w > 0:
                    nw = 1
                else:
                    nw = 0
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
        try:
            self.score = 0.5+1/(1+exp(self.rmse))
        except OverflowError:
            self.score = 0.5
        #print self.rmse, self.score





def BayesComb(prior, models, w, b, month, check_vote = False, prior_score = 0.5):



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
    scores += [prior]
    likes += [prior_score]

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
    try:
        rmse = sqrt(mse/len(correct_agg.keys()))
    except ZeroDivisionError:
        rmse = 1000
    return rmse



def TrainModel(csv_gen, **kwargs):
    """Trains the model based on receiving a 'csv-generator' from the rows"""

    def WeightMatches(models, csv_gen, check_vote = False,
                      prior = 0.5, prior_score = 0.5):
        for row in csv_gen:
            p1 = int(row["White Player #"])
            p2 = int(row["Black Player #"])
            s = float(row["Score"])
            m = row["Month #"]
            t_score = BayesComb(prior, models, p1, p2, m,
                                check_vote = check_vote,
                                prior_score = prior_score)

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



    default_rank = kwargs.pop('default_rank', 0)
    seed_frac = kwargs.pop('seed_frac', 0.3)
    rep_frac = kwargs.pop('rep_frac', 0.2)
    prior = kwargs.pop('prior', 0.5)
    prior_score = kwargs.pop('prior_score', 0.5)
    check_train = kwargs.pop('check_train', False)
    check_indiv = kwargs.pop('check_indiv', True)

    if len(kwargs) > 0:
        raise TypeError, 'Unrecognized kwargs: %s' % (','.join(kwargs.keys()))

    model_list = []

    train, test = TrainTestInds(csv_gen, frac = seed_frac)
    model_list.append(TrainSingle(train, test, check_vote = check_indiv,
                                  default_rank = default_rank))
    c = 0
    while len(test) > 1000 and c < 50:
    #for i in range(num_models):
        c += 1
        train, test = TrainTestInds(test, frac = rep_frac)
        gen = WeightMatches(model_list, train, check_vote = check_train,
                            prior = prior, prior_score = prior_score)
        model_list.append(TrainSingle(gen, test, check_vote = check_indiv,
                                      default_rank = default_rank))

    train, test = TrainTestInds(test, frac = rep_frac)
    gen = WeightMatches(model_list, train, check_vote = check_train,
                        prior = prior, prior_score = prior_score)
    model_list.append(TrainSingle(gen, test, check_vote = check_indiv,
                                      default_rank = default_rank))

    return model_list