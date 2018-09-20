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


class FlipSplitter(Splitter):
    "Fit on validation data, evaluate on train data"

    def next_split(self):
        return self.data['validation'], self.data['train']


class ValidationSplitter(Splitter):
    "Single split of data into train, validation"

    def next_split(self):
        return self.data['train'], self.data['validation']


class CheatSplitter(Splitter):
    "Single split of data into train+validation, tournament"

    def next_split(self):
        dfit = self.data.region_isin(['train', 'validation'])
        dpre = self.data['tournament']
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

    def __init__(self, data, tournament, kfold=5, seed=0, train_only=True):
        self.p = {'data': data,
                  'tournament': tournament,
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
            y = data.y[self.p['tournament']]
            self.cv = cv.split(data.x, y)
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


class ConsecutiveCVSplitter(Splitter):
    "K-fold CV where folds contain mostly consecutive eras"

    def __init__(self, data, kfold=5, seed=0, train_only=True):
        self.p = {'data': data,
                  'kfold': kfold,
                  'seed': seed,
                  'train_only': train_only}
        self.eras = None
        self.cv = None
        self.max_count = kfold - 1
        self.reset()

    def next_split(self):
        if not nx.isint(self.p['seed']):
            raise ValueError("`seed` must be an integer")
        data = self.p['data']
        if self.count == 0:
            if self.p['train_only']:
                data = data['train']
            eras = data.unique_era()
            n = int(eras.size / self.p['kfold'])
            idx = self.p['seed'] % n
            if idx == 0:
                era0 = eras[idx:n]
                eras = eras[n:]
            else:
                era0 = np.concatenate((eras[idx:n], eras[-idx:]))
                eras = np.concatenate((eras[:idx], eras[n:-idx]))
            eras = np.array_split(eras, self.p['kfold'] - 1)
            eras.append(era0)
            self.eras = eras
        fit_eras = []
        for i, e in enumerate(self.eras):
            if i != self.count:
                fit_eras.append(e)
        fit_eras = np.concatenate(fit_eras)
        dfit = data.era_isin(fit_eras)
        pre_eras = self.eras[self.count]
        dpre = data.era_isin(pre_eras)
        return dfit, dpre


class CustomCVSplitter(Splitter):
    "Cross validation over folds given by user as list of data objects"

    def __init__(self, data_list):
        ids = []
        if len(data_list) < 2:
            msg = '`data_list` must contain at least two data objects'
            raise ValueError(msg)
        for d in data_list:
            if not isinstance(d, nx.Data):
                msg = '`data_list` must be a list of nx.Data objects'
                raise ValueError(msg)
            ids.extend(d.ids.tolist())
        if len(ids) != len(set(ids)):
            raise ValueError('ids of predict data objects overlap')
        self.p = {'data_list': data_list}
        self.max_count = len(data_list) - 1
        self.reset()

    def next_split(self):
        data_list = self.p['data_list']
        dpre = data_list[self.count]
        idx = list(range(len(data_list)))
        idx.remove(self.count)
        dfit = data_list[idx[0]]
        for ix in idx[1:]:
            dfit += data_list[ix]
        return dfit, dpre


class CustomSplitter(Splitter):
    "Fit/predict splits given by user as list of data tuples"

    def __init__(self, data_list):
        ids = []
        for dt in data_list:
            if len(dt) != 2:
                msg = 'elements in `data_list` must have length of two'
                raise ValueError(msg)
            isd0 = isinstance(dt[0], nx.Data)
            isd1 = isinstance(dt[1], nx.Data)
            if not isd0 or not isd1:
                msg = '`data_list` must be a list of nx.Data object tuples'
                raise ValueError(msg)
            ids.extend(dt[1].ids.tolist())
        if len(ids) != len(set(ids)):
            raise ValueError('ids of predict data objects overlap')
        self.p = {'data_list': data_list}
        self.max_count = len(data_list) - 1
        self.reset()

    def next_split(self):
        data_tuple = self.p['data_list'][self.count]
        dfit = data_tuple[0]
        dpre = data_tuple[1]
        return dfit, dpre
