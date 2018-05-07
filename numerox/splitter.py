import sys

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import numerox as nx


class Splitter(object):
    "Base class used by data splitters; cannot be used as a splitter by itself"

    def __init__(self, data):
        self.data = data
        self.max_count = 0
        self.reset()

    def reset(self):
        self.count = 0

    def __iter__(self):
        return self

    def next(self):
        if self.count > self.max_count:
            raise StopIteration
        tup = self.next_split()
        self.count += 1
        return tup

    # py3 compat
    def __next__(self):
        return self.next()  # pragma: no cover

    def __repr__(self):
        msg = ""
        splitter = self.__class__.__name__
        if not hasattr(self, 'p'):
            msg += splitter + "(data)"
        else:
            splitter = self.__class__.__name__
            msg += splitter + "(data, "
            for name, value in self.p.items():
                if name != 'data':
                    msg += name + "=" + str(value) + ", "
            msg = msg[:-2]
            msg += ")"
        return msg


class TournamentSplitter(Splitter):
    "Single split of data into train, tournament"

    def next_split(self):
        return self.data['train'], self.data['tournament']


class ValidationSplitter(Splitter):
    "Single split of data into train, validation"

    def next_split(self):
        return self.data['train'], self.data['validation']


class CheatSplitter(Splitter):
    "Single split of data into train+validation, tournament"

    def next_split(self):
        dfit = self.data.region_isin(['train', 'validation'])
        dpre = self.data['validation']
        return dfit, dpre


class SplitSplitter(Splitter):
    "Single fit-predict split of data"

    def __init__(self, data, fit_fraction, seed=0, train_only=True):
        self.p = {'data': data,
                  'fit_fraction': fit_fraction,
                  'seed': seed,
                  'train_only': train_only}
        self.max_count = 0
        self.reset()

    def next_split(self):
        data = self.p['data']
        if self.p['train_only']:
            data = data['train']
        eras = data.unique_era()
        rs = np.random.RandomState(self.p['seed'])
        rs.shuffle(eras)
        nfit = int(self.p['fit_fraction'] * eras.size + 0.5)
        dfit = data.era_isin(eras[:nfit])
        dpre = data.era_isin(eras[nfit:])
        return dfit, dpre


class CVSplitter(Splitter):
    "K-fold cross validation fit-predict splits across train eras"

    def __init__(self, data, kfold=5, seed=0, train_only=True):
        self.p = {'data': data,
                  'kfold': kfold,
                  'seed': seed,
                  'train_only': train_only}
        self.eras = None
        self.cv = None
        self.max_count = kfold
        self.reset()

    def next_split(self):
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
            fit_index, predict_index = self.cv.__next__()  # pragma: no cover
        era_fit = [self.eras[i] for i in fit_index]
        era_predict = [self.eras[i] for i in predict_index]
        dfit = data.era_isin(era_fit)
        dpre = data.era_isin(era_predict)
        return dfit, dpre


class LoocvSplitter(Splitter):
    "Leave one (era) out cross validation fit-predict splits across train eras"

    def __init__(self, data, train_only=True):
        if train_only:
            data = data['train']
        self.p = {'data': data}
        self.eras = data.unique_era()
        self.max_count = self.eras.size - 1
        self.reset()

    def next_split(self):
        data = self.p['data']
        era = self.eras[self.count]
        dfit = data.era_isnotin([era])
        dpre = data.era_isin([era])
        return dfit, dpre


class IgnoreEraCVSplitter(Splitter):
    "K-fold cross validation fit-predict splits ignoring eras and balancing y"

    def __init__(self, data, kfold=5, seed=0, train_only=True):
        self.p = {'data': data,
                  'kfold': kfold,
                  'seed': seed,
                  'train_only': train_only}
        self.cv = None
        self.max_count = kfold
        self.reset()

    def next_split(self):
        data = self.p['data']
        if self.count == 0:
            if self.p['train_only']:
                data = data['train']
            cv = StratifiedKFold(n_splits=self.p['kfold'],
                                 random_state=self.p['seed'],
                                 shuffle=True)
            self.cv = cv.split(data.x, data.y)
        if sys.version_info[0] == 2:
            fit_index, pre_index = self.cv.next()
        else:
            fit_index, pre_index = self.cv.__next__()  # pragma: no cover
        dfit = nx.Data(data.df.take(fit_index))
        dpre = nx.Data(data.df.take(pre_index))
        return dfit, dpre


class RollSplitter(Splitter):
    "Roll forward through consecutive eras to generate fit, train splits"

    def __init__(self, data, fit_window, predict_window, step,
                 train_only=True):
        self.p = {'data': data,
                  'fit_window': fit_window,
                  'predict_window': predict_window,
                  'step': step,
                  'train_only': train_only}
        self.eras = None
        self.cv = None
        self.max_count = np.inf  # prevent Splitter for stoping iteration
        self.reset()

    def next_split(self):
        data = self.p['data']
        if self.count == 0:
            if self.p['train_only']:
                data = data['train']
            self.eras = data.unique_era()
        f_idx1 = self.count * self.p['step']
        f_idx2 = f_idx1 + self.p['fit_window']
        p_idx1 = f_idx2
        p_idx2 = p_idx1 + self.p['predict_window']
        nera = self.eras.size
        if p_idx2 > nera:
            raise StopIteration
        era_fit = []
        era_pre = []
        for i in range(nera):
            n_ifs = 0
            if i >= f_idx1 and i < f_idx2:
                era_fit.append(self.eras[i])
                n_ifs += 1
            if i >= p_idx1 and i < p_idx2:
                era_pre.append(self.eras[i])
                n_ifs += 1
            if n_ifs > 1:
                raise RuntimeError("RollSplitter bug!")  # pragma: no cover
        dfit = data.era_isin(era_fit)
        dpre = data.era_isin(era_pre)
        return dfit, dpre
