import os
import zipfile

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import ks_2samp

import numerox as nx
from numerox.metrics import metrics_per_era
from numerox.metrics import metrics_per_name
from numerox.metrics import concordance
from numerox.metrics import LOGLOSS_BENCHMARK

HDF_PREDICTION_KEY = 'numerox_prediction'
EXAMPLE_PREDICTIONS = 'example_predictions.csv'

ORIGINALITY_CORR_LTE = 0.95
ORIGINALITY_KS_GT = 0.03
CONSISTENCY_GTE = 0.58
CONCORDANCE_LT = 0.12


class Prediction(object):

    def __init__(self, df=None):
        self.df = df

    @property
    def names(self):
        "List (copy) of names in prediction object"
        if self.df is None:
            return []
        return self.df.columns.tolist()

    def rename(self, mapper):
        """
        Rename prediction name(s).

        Parameters
        ----------
        mapper : {dict-like, str}
            You can rename using a dictionary with old name as key, new as
            value. Or, if the prediction contains a single name, then `mapper`
            can be a string containing the new name.

        Returns
        -------
        renamed : Prediction
            A copy of the prediction with renames names.
        """
        if self.df is None:
            raise ValueError("Cannot rename an empty prediction")
        if nx.isstring(mapper):
            if self.shape[1] != 1:
                raise ValueError("prediction must contain a single name")
            mapper = {self.names[0]: mapper}
        df = self.df.rename(columns=mapper, copy=True)
        return Prediction(df)

    def drop(self, name):
        "Drop name (str) or names (e.g. a list of names) from prediction"
        if self.df is None:
            raise ValueError("Cannot drop a name from an empty prediction")
        df = self.df.drop(columns=name)
        return Prediction(df)

    @property
    def ids(self):
        "View of ids as a numpy str array"
        if self.df is None:
            return np.array([], dtype=str)
        return self.df.index.values

    @property
    def y(self):
        "View of y as a 2d numpy float array"
        if self.df is None:
            raise ValueError("prediction is empty")
        return self.df.values

    def ynew(self, y_array):
        "Copy of prediction but with prediction.y=`y_array`"
        if self.df is None:
            raise ValueError("prediction is empty")
        if y_array.shape != self.shape:
            msg = "`y_array` must have the same shape as prediction"
            raise ValueError(msg)
        df = pd.DataFrame(data=y_array,
                          index=self.df.index.copy(deep=True),
                          columns=self.df.columns.copy())
        return Prediction(df)

    def iter(self):
        "Yield a prediction object with only one model at a time"
        for name in self.names:
            yield self[name]

    def merge_arrays(self, ids, y, name):
        "Merge numpy arrays `ids` and `y` with name `name`"
        df = pd.DataFrame(data={name: y}, index=ids)
        prediction = Prediction(df)
        return self.merge(prediction)

    def merge(self, prediction):
        "Merge prediction"
        return merge_predictions([self, prediction])

    def save(self, path_or_buf, compress=True, mode='w'):
        """
        Save prediction as an hdf archive.

        Raises a ValueError if the prediction is empty.

        Parameters
        ----------
        path_or_buf : {str, HDFStore}
            Full path filename (string) or HDFStore object.
        compress : bool, optional
            Whether or not to compress the archive. The default (True) is to
            compress.
        mode : str, optional
            The save mode. By default ('w') the archive is overwritten if it
            exists and created if not. With mode 'a' the prediction is
            appended to the archive (the archive must already exist and it
            must contain a prediction object).

        Returns
        -------
        None
        """
        if self.df is None:
            raise ValueError("Prediction object is empty; nothing to save")
        if mode not in ('w', 'a'):
            raise ValueError("`mode` must be 'w' or 'a'")
        if mode == 'a':
            p = nx.load_prediction(path_or_buf)
            self = p.merge(self)
        if compress:
            self.df.to_hdf(path_or_buf, HDF_PREDICTION_KEY,
                           complib='zlib', complevel=4)
        else:
            self.df.to_hdf(path_or_buf, HDF_PREDICTION_KEY)

    def to_csv(self, path_or_buf=None, decimals=6, verbose=False):
        "Save a csv file of predictions; predictin must contain only one name"
        if self.shape[1] != 1:
            raise ValueError("prediction must contain a single name")
        df = self.df.iloc[:, 0].to_frame('probability')
        df.index.rename('id', inplace=True)
        float_format = "%.{}f".format(decimals)
        df.to_csv(path_or_buf, float_format=float_format)
        if verbose:
            print("Save {}".format(path_or_buf))

    def summary(self, data, round_output=True):
        "Performance summary of prediction object that contains a single name"

        if self.shape[1] != 1:
            raise ValueError("prediction must contain a single name")

        # metrics
        metrics, regions = metrics_per_era(data, self, region_as_str=True)
        metrics = metrics.drop(['era', 'name'], axis=1)

        # additional metrics
        region_str = ', '.join(regions)
        nera = metrics.shape[0]
        logloss = metrics['logloss']
        consis = (logloss < LOGLOSS_BENCHMARK).mean()
        sharpe = (LOGLOSS_BENCHMARK - logloss).mean() / logloss.std()

        # summary of metrics
        m1 = metrics.mean(axis=0).tolist() + ['region', region_str]
        m2 = metrics.std(axis=0).tolist() + ['eras', nera]
        m3 = metrics.min(axis=0).tolist() + ['sharpe', sharpe]
        m4 = metrics.max(axis=0).tolist() + ['consis', consis]
        data = [m1, m2, m3, m4]

        # make dataframe
        columns = metrics.columns.tolist() + ['stats', '']
        df = pd.DataFrame(data=data,
                          index=['mean', 'std', 'min', 'max'],
                          columns=columns)

        # make output (optionally) pretty
        if round_output:
            round_dict = {'logloss': 6, 'auc': 4, 'acc': 4, 'ystd': 4}
            df = df.round(decimals=round_dict)

        return df

    def metrics_per_era(self, data, metrics=['logloss', 'auc', 'acc', 'ystd'],
                        era_as_str=True):
        "DataFrame containing given metrics versus era (as index)"
        metrics, regions = metrics_per_era(data, self, columns=metrics,
                                           era_as_str=era_as_str)
        metrics.index = metrics['era']
        metrics = metrics.drop(['era'], axis=1)
        return metrics

    def performance(self, data, era_as_str=True, region_as_str=True,
                    columns=['logloss', 'auc', 'acc', 'ystd', 'sharpe',
                             'consis'], sort_by='logloss'):
        df, info = metrics_per_name(data,
                                    self,
                                    columns=columns,
                                    era_as_str=era_as_str,
                                    region_as_str=region_as_str)
        if sort_by in columns:
            if sort_by == 'logloss':
                df = df.sort_values(by='logloss', ascending=True)
            elif sort_by == 'auc':
                df = df.sort_values(by='auc', ascending=False)
            elif sort_by == 'acc':
                df = df.sort_values(by='acc', ascending=False)
            elif sort_by == 'ystd':
                df = df.sort_values(by='ystd', ascending=False)
            elif sort_by == 'sharpe':
                df = df.sort_values(by='sharpe', ascending=False)
            elif sort_by == 'consis':
                by = ['consis']
                ascending = [False]
                if 'logloss' in df:
                    by.append('logloss')
                    ascending.append('True')
                df = df.sort_values(by=by, ascending=ascending)
            else:
                raise ValueError("`sort_by` name not recognized")
        return df

    def dominance(self, data, sort_by='logloss'):
        "Mean (across eras) of fraction of models bested per era"
        columns = ['logloss', 'auc', 'acc']
        mpe, regions = metrics_per_era(data, self, columns=columns)
        dfs = []
        for i, col in enumerate(columns):
            pivot = mpe.pivot(index='era', columns='name', values=col)
            names = pivot.columns.tolist()
            a = pivot.values
            n = a.shape[1] - 1.0
            if n == 0:
                raise ValueError("Must have at least two names")
            m = []
            for j in range(pivot.shape[1]):
                if col == 'logloss':
                    z = (a[:, j].reshape(-1, 1) < a).sum(axis=1) / n
                else:
                    z = (a[:, j].reshape(-1, 1) > a).sum(axis=1) / n
                m.append(z.mean())
            df = pd.DataFrame(data=m, index=names, columns=[col])
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        df = df.sort_values([sort_by], ascending=[False])
        return df

    def concordance(self, data):
        "Less than 0.12 is passing; data should be the full dataset."
        return concordance(data, self)

    def correlation(self, name=None):
        "Correlation of predictions; by default reports given for each model"
        if name is None:
            names = self.names
        else:
            names = [name]
        z = self.df.values
        znames = self.names
        idx = np.isfinite(z.sum(axis=1))
        z = z[idx]
        z = (z - z.mean(axis=0)) / z.std(axis=0)
        for name in names:
            print(name)
            idx = znames.index(name)
            corr = np.dot(z[:, idx], z) / z.shape[0]
            index = (-corr).argsort()
            for ix in index:
                zname = znames[ix]
                if name != zname:
                    print("   {:.4f} {}".format(corr[ix], zname))

    def originality(self, submitted_names):
        "Which models are original given the models already submitted?"

        # predictions of models already submitted
        ys = self.df[submitted_names].values

        # models that have not been submitted; we will report on these
        names = self.names
        names = [m for m in names if m not in submitted_names]

        # originality
        columns = ['corr', 'corrTF', 'ks', 'ksTF', 'original']
        df = pd.DataFrame(index=names, columns=columns)
        for name in names:
            corr = -np.inf
            ks = np.inf
            corrTF = True
            ksTF = True
            y = self.df[name].values
            for i in range(ys.shape[1]):
                c = np.abs(pearsonr(y, ys[:, i])[0])
                if c > corr:
                    corr = c
                if corrTF and c > ORIGINALITY_CORR_LTE:
                    corrTF = False
                k = ks_2samp(y, ys[:, i])[0]
                if k < ks:
                    ks = k
                if ksTF and k <= ORIGINALITY_KS_GT:
                    ksTF = False
            df.loc[name, 'corr'] = corr
            df.loc[name, 'ks'] = ks
            df.loc[name, 'corrTF'] = corrTF
            df.loc[name, 'ksTF'] = ksTF
            df.loc[name, 'original'] = corrTF and ksTF

        return df

    def check(self, data):
        "Run Numerai checks; must contain a model named 'example_predictions'"

        if 'example_predictions' not in self:
            msg = "must contain a model named 'example_predictions'"
            raise ValueError(msg)
        eidx = self.names.index('example_predictions')
        yex = self.y[:, eidx]

        names = list(self.names)
        names.remove('example_predictions')

        df_dict = {}
        columns = ['validation', 'test', 'live', 'all', 'pass']
        data = data.loc[self.ids]
        regions = data.region
        for name in names:
            print(name)
            df = pd.DataFrame(columns=columns)
            idx = self.names.index(name)
            y = self.y[:, idx]
            for region in ('validation', 'test', 'live', 'all'):
                if region == 'all':
                    yi = y
                    yexi = yex
                else:
                    idx = regions == region
                    yi = y[idx]
                    yexi = yex[idx]
                df.loc['corr', region] = pearsonr(yi, yexi)[0]
                df.loc['rcorr', region] = spearmanr(yi, yexi)[0]
                df.loc['min', region] = yi.min()
                df.loc['max', region] = yi.max()
                maz = np.abs((yi - yi.mean()) / yi.std()).max()
                df.loc['maz', region] = maz

            df.loc['corr', 'pass'] = (df.loc['corr'][:-1] >= 0.2).all()
            df.loc['rcorr', 'pass'] = (df.loc['rcorr'][:-1] >= 0.2).all()
            df.loc['min', 'pass'] = (df.loc['min'][:-1] >= 0.3).all()
            df.loc['max', 'pass'] = (df.loc['max'][:-1] <= 0.7).all()
            df.loc['maz', 'pass'] = (df.loc['maz'][:-1] <= 15).all()

            print(df)

            df_dict[name] = df

        return df_dict

    def compare(self, data, prediction):
        "Compare performance of predictions with the same names"
        cols = ['logloss1', 'logloss2', 'win1',
                'corr', 'maxdiff', 'ystd1', 'ystd2']
        comp = pd.DataFrame(columns=cols)
        names = []
        for name in self.names:
            if name in prediction:
                names.append(name)
        if len(names) == 0:
            return comp
        ids = data.ids
        df1 = self.loc[ids]
        df2 = prediction.loc[ids]
        p1 = self[names]
        p2 = prediction[names]
        m1 = p1.metrics_per_era(data, metrics=['logloss', 'auc', 'ystd'],
                                era_as_str=False)
        m2 = p2.metrics_per_era(data, metrics=['logloss', 'auc', 'ystd'],
                                era_as_str=False)
        for name in names:

            m1i = m1[m1.name == name]
            m2i = m2[m2.name == name]

            if (m1i.index != m2i.index).any():
                raise IndexError("Can only handle aligned eras")

            logloss1 = m1i.logloss.mean()
            logloss2 = m2i.logloss.mean()
            win1 = (m1i.logloss < m2i.logloss).mean()

            y1 = df1[name].y.reshape(-1)
            y2 = df2[name].y.reshape(-1)

            corr = np.corrcoef(y1, y2)[0, 1]
            maxdiff = np.abs(y1 - y2).max()
            ystd1 = y1.std()
            ystd2 = y2.std()

            m = [logloss1, logloss2, win1, corr, maxdiff, ystd1, ystd2]
            comp.loc[name] = m

        return comp

    def copy(self):
        "Copy of prediction"
        if self.df is None:
            return Prediction(None)
        # df.copy(deep=True) doesn't copy index. So:
        df = self.df
        df = pd.DataFrame(df.values.copy(),
                          df.index.copy(deep=True),
                          df.columns.copy())
        return Prediction(df)

    def hash(self):
        """
        Hash of prediction object.

        The hash is unlikely to be the same across different computers (OS,
        Python version, etc). But should be constant on the same system.
        """
        b = self.df.values.tobytes(order='A')
        h = hash(b)
        return h

    def __getitem__(self, name):
        "Prediction indexing is by model name(s)"
        if nx.isstring(name):
            p = Prediction(self.df[name].to_frame(name))
        else:
            p = Prediction(self.df[name])
        return p

    def __setitem__(self, name, prediction):
        "Add (or replace) a prediction by name"
        if prediction.df.shape[1] != 1:
            raise ValueError("Can only insert a single model at a time")
        prediction.df.columns = [name]
        self.df = self.merge(prediction).df

    @property
    def loc(self):
        "indexing by row ids"
        return Loc(self)

    def __add__(self, prediction):
        "Merge predictions"
        return self.merge(prediction)

    def __iadd__(self, prediction):
        "Merge predictions"
        return self.merge(prediction)

    def __contains__(self, name):
        "Is `name` already in prediction? True or False"
        return name in self.df

    def __eq__(self, prediction):
        "Check if prediction objects are equal or not; order matters"
        if self.df is None and prediction.df is None:
            return True
        return self.df.equals(prediction.df)

    @property
    def size(self):
        if self.df is None:
            return 0
        return self.df.size

    @property
    def shape(self):
        if self.df is None:
            return (0, 0)
        return self.df.shape

    def __len__(self):
        "Number of rows"
        if self.df is None:
            return 0
        return self.df.__len__()

    def __repr__(self):
        shape = self.shape
        if shape[1] == 0:
            frac_miss = 0.0
        else:
            frac_miss = self.df.isna().mean().mean()
        fmt = 'Prediction({} rows x {} names; {:.4f} missing)'
        return fmt.format(shape[0], shape[1], frac_miss)


class Loc(object):
    "Utility class for the loc method."

    def __init__(self, prediction):
        self.prediction = prediction

    def __getitem__(self, index):
        return Prediction(self.prediction.df.loc[index])


def load_prediction(filename):
    "Load prediction object from hdf archive"
    df = pd.read_hdf(filename, key=HDF_PREDICTION_KEY)
    return Prediction(df)


def load_prediction_csv(filename, name=None):
    "Load prediction object from a Numerai csv (text) tournament file"
    df = pd.read_csv(filename, index_col='id')
    if df.shape[1] != 1:
        raise ValueError("csv file must contain on column of predictions")
    if name is None:
        name = os.path.split(filename)[-1]
        if name.endswith('.csv'):
            name = name[:-4]
    df.columns = [name]
    return Prediction(df)


def load_example_predictions(data_zip):
    "Load example predictions from Numerai zip archive"
    zf = zipfile.ZipFile(data_zip)
    df = pd.read_csv(zf.open(EXAMPLE_PREDICTIONS), header=0, index_col=0)
    df.columns = ['example_predictions']
    p = nx.Prediction(df)
    return p


def merge_predictions(prediction_list):
    """
    Merge a list of predictions.

    Raises ValueError on overlapping predictions (same model name and same
    row id).
    """
    p = prediction_list[0].copy()
    for i in range(1, len(prediction_list)):
        pi = prediction_list[i]
        for name in pi.names:
            p = _merge_predictions(p, pi[name])
    return p


def _merge_predictions(prediction1, prediction2):
    "Merge a possibly multi-name prediction1 with a single-name prediction2"
    if prediction2.shape[1] != 1:
        raise ValueError("`prediction2` must contain a single name")
    name = prediction2.names[0]
    if prediction1.df is None:
        # empty prediction
        df = prediction2.df
    elif name not in prediction1:
        # inserting predictions from a model not already in report
        df = pd.merge(prediction1.df, prediction2.df, how='outer',
                      left_index=True, right_index=True)
    else:
        # add more ys from a model whose name already exists
        y = prediction1.df[name]
        y = y.dropna()
        s = prediction2.df.iloc[:, 0]
        s = s.dropna()
        s = pd.concat([s, y], join='outer', ignore_index=False,
                      verify_integrity=True)
        dfnew = s.to_frame(name)
        df = pd.merge(prediction1.df, dfnew, how='outer', on=name,
                      left_index=True, right_index=True)
        df[name] = dfnew
    return Prediction(df)
