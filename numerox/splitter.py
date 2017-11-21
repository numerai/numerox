import sys

import numpy as np
from sklearn.model_selection import KFold


# simple splitters ----------------------------------------------------------

class Splitter(object):
    "Base class used by simple splitters"

    def __init__(self, data):
        self.data = data
        self.count = 0

    def __iter__(self):
        return self

    def reset(self):
        self.count = 0

    def __next__(self):
        # py3 compat
        return self.next()

    def __repr__(self):
        msg = ""
        splitter = self.__class__.__name__
        msg += splitter + "(data)"
        return msg


class TournamentSplitter(Splitter):
    "Single split of data into train, tournament"

    def next(self):
        if self.count > 0:
            raise StopIteration
        self.count += 1
        return self.data['train'], self.data['tournament']


class ValidationSplitter(Splitter):
    "Single split of data into train, validation"

    def next(self):
        if self.count > 0:
            raise StopIteration
        self.count += 1
        return self.data['train'], self.data['validation']


class CheatSplitter(Splitter):
    "Single split of data into train+validation, tournament"

    def next(self):
        if self.count > 0:
            raise StopIteration
        self.count += 1
        dfit = self.data.region_isin(['train', 'validation'])
        dpredict = self.data['validation']
        return dfit, dpredict


# complicated splitters -----------------------------------------------------

class Splitter2(object):
    "Base class used by more complicated splitters"

    def __iter__(self):
        return self

    def reset(self):
        self.count = 0

    def __repr__(self):
        msg = ""
        splitter = self.__class__.__name__
        msg += splitter + "(data, "
        for name, value in self.p.items():
            if name != 'data':
                msg += name + "=" + str(value) + ", "
        msg = msg[:-2]
        msg += ")"
        return msg


class SplitSplitter(Splitter2):
    "Single fit-predict split of data"

    def __init__(self, data, fit_fraction, seed=0, train_only=True):
        self.p = {'data': data,
                  'fit_fraction': fit_fraction,
                  'seed': seed,
                  'train_only': train_only}
        self.count = 0

    def next(self):
        if self.count > 0:
            raise StopIteration
        data = self.p['data']
        if self.p['train_only']:
            data = data['train']
        eras = data.unique_era()
        rs = np.random.RandomState(self.p['seed'])
        rs.shuffle(eras)
        nfit = int(self.p['fit_fraction'] * eras.size + 0.5)
        data_fit = data.era_isin(eras[:nfit])
        data_predict = data.era_isin(eras[nfit:])
        self.count += 1
        return data_fit, data_predict

    __next__ = next  # py3 compat


class CVSplitter(Splitter2):
    "K-fold cross validation fit-predict splits across train eras"

    def __init__(self, data, kfold=5, seed=0, train_only=True):
        self.p = {'data': data,
                  'kfold': kfold,
                  'seed': seed,
                  'train_only': train_only}
        self.count = 0
        self.eras = None
        self.cv = None

    def next(self):
        if self.count >= self.p['kfold']:
            raise StopIteration
        data = self.p['data']
        if self.count == 0:
            if self.p['train_only']:
                data = data['train']
            self.eras = data.unique_era()
            cv = KFold(n_splits=self.p['kfold'], random_state=self.p['seed'],
                       shuffle=True)
            self.cv = cv.split(self.eras)
        if sys.version_info[0] == 2:
            fit_index, predict_index = self.cv.next()
        else:
            fit_index, predict_index = self.cv.__next__()
        era_fit = [self.eras[i] for i in fit_index]
        era_predict = [self.eras[i] for i in predict_index]
        dfit = data.era_isin(era_fit)
        dpredict = data.era_isin(era_predict)
        self.count += 1
        return dfit, dpredict

    __next__ = next  # py3 compat
