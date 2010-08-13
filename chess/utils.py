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


def Logit(v):
    try:
        r = 1/(1+exp(-v))
    except OverflowError:
        if v > 0:
            r = 1
        else:
            r = 0
    return r

def InTreatScore(inscore):
    return (inscore - 0.5)*2

def OutTreatScore(inscore):
    return (inscore/2)+0.5


def TrainTestInds(nitems, frac = 0.7):
    train = []
    test = []
    for item in nitems:
        if random.random() < frac:
            train.append(item)
        else:
            test.append(item)

    return train, test