import os
import zipfile
import warnings

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr

import numerox as nx
from numerox.metrics import metrics_per_era
from numerox.metrics import metrics_per_name
from numerox.metrics import concordance
from numerox.metrics import LOGLOSS_BENCHMARK
from numerox.metrics import add_split_pairs
from numerox.util import is_none_slice

HDF_PREDICTION_KEY = 'numerox_prediction'
EXAMPLE_PREDICTIONS = 'example_predictions_target_{}.csv'

ORIGINALITY_CORR_LTE = 0.95
ORIGINALITY_KS_GT = 0.03
CONSISTENCY_GTE = 0.58
CONCORDANCE_LT = 0.12


class Prediction(object):

    def __init__(self, df=None):
        self.df = df

    # ids -------------------------------------------------------------------

    @property
    def ids(self):
        "View of ids as a numpy str array"
        if self.df is None:
            return np.array([], dtype=str)
        return self.df.index.values

    @property
    def loc(self):
        "indexing by row ids"
        return Loc(self)

    # y ---------------------------------------------------------------------

    @property
    def y(self):
        "View of y as a 2d numpy float array"
        if self.df is None:
            raise ValueError("prediction is empty")
        return self.df.values

    @property
    def y_df(self):
        "Copy of predictions, y, as a dataframe"
        return self.df.copy()

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

    def y_correlation(self):
        "Correlation matrix of y's (predictions) as dataframe"
        return self.df.corr()

    def correlation(self, pair=None, as_str=True):
        "Correlation of predictions; by default reports given for each model"
        if pair is None:
            pairs = self.pairs(as_str)
        else:
            pairs = [(pair[0], nx.tournament_str(pair[1]))]
        z = self.df.values
        zpairs = self.pairs(as_str)
        idx = np.isfinite(z.sum(axis=1))
        z = z[idx]
        z = (z - z.mean(axis=0)) / z.std(axis=0)
        for pair in pairs:
            print('{}, {}'.format(*pair))
            idx = zpairs.index(pair)
            corr = np.dot(z[:, idx], z) / z.shape[0]
            index = (-corr).argsort()
            for ix in index:
                zpair = zpairs[ix]
                if pair != zpair:
                    print("   {:.4f} {}, {}".format(corr[ix], *zpair))

    # name ------------------------------------------------------------------

    def names(self):
        "List (copy) of names in prediction object"
        pairs = self.pairs()
        names = []
        for n, t in pairs:
            if n not in names:
                names.append(n)
        return names

    def name_isin(self, name):
        "Is name in Prediction object? True or False."
        names = self.names()
        return name in names

    def drop_name(self, name):
        "Drop name or list of names from prediction; return copy"
        if self.df is None:
            raise ValueError("Cannot drop a name from an empty prediction")
        pair = self.pairs_with_name(name, as_str=False)
        df = self.df.drop(columns=pair)
        return Prediction(df)

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
        names = self.names()
        df = self.df.copy()
        if nx.isstring(mapper):
            if len(names) != 1:
                msg = 'prediction contains more than one name; use dict mapper'
                raise ValueError(msg)
            pairs = self.pairs(as_str=False)
            pairs = [(mapper, t) for n, t in pairs]
        elif isinstance(mapper, dict):
            prs = self.pairs(as_str=False)
            pairs = []
            for pr in prs:
                if pr[0] in mapper:
                    pr = (mapper[pr[0]], pr[1])
                pairs.append(pr)
        df.columns = pairs
        return Prediction(df)

    # tournament ------------------------------------------------------------

    def tournaments(self, as_str=True):
        "List (copy) of tournaments in prediction object"
        pairs = self.pairs(as_str=False)
        tournaments = [t for n, t in pairs]
        tournaments = sorted(set(tournaments))
        if as_str:
            tournaments = [nx.tournament_str(t) for t in tournaments]
        return tournaments

    def tournament_isin(self, tournament):
        "Is tournament in Prediction object? True or False."
        tournaments = self.tournaments(as_str=False)
        tournament = nx.tournament_int(tournament)
        return tournament in tournaments

    def drop_tournament(self, tournament):
        "Drop tournament or list of tournaments from prediction; return copy"
        if self.df is None:
            msg = "Cannot drop a tournament from an empty prediction"
            raise ValueError(msg)
        pair = self.pairs_with_tournament(tournament, as_str=False)
        df = self.df.drop(columns=pair)
        return Prediction(df)

    # pair ------------------------------------------------------------------

    def pairs(self, as_str=True):
        "List (copy) of (name, tournament) tuple in prediction object"
        if self.df is None:
            return list()
        pairs = self.df.columns.tolist()
        if as_str:
            pairs = [(n, nx.tournament_str(t)) for n, t in pairs]
        return pairs

    def pairs_df(self):
        "Bool dataframe with names as index and tournaments as columns"
        names = self.names()
        tourns = nx.tournament_all()
        df = pd.DataFrame(index=names, columns=tourns)
        for name in names:
            for tourn in tourns:
                df.loc[name, tourn] = (name, tourn) in self
        return df

    def pair_isin(self, pair):
        "Is (name, tournament) tuple in Prediction object? True or False."
        return pair in self

    def pairs_with_name(self, name, as_str=True):
        "List of pairs with given `name`; `name` can be str or list of str."
        if isinstance(name, list):
            names = name
        elif nx.isstring(name):
            names = [name]
        else:
            raise ValueError('`name` must be str or list')
        prs = self.pairs(as_str)
        pairs = []
        for pr in prs:
            if pr[0] in names:
                pairs.append(pr)
        return pairs

    def pairs_with_tournament(self, tournament, as_str=True):
        "List of pairs; `tournament` can be int, str, or list"
        if isinstance(tournament, list):
            tournaments = [nx.tournament_int(t) for t in tournament]
        elif nx.isstring(tournament):
            tournaments = [nx.tournament_int(tournament)]
        elif nx.isint(tournament):
            tournaments = [tournament]
        else:
            raise ValueError('`tournament` must be int, str or list')
        prs = self.pairs(as_str=False)
        pairs = []
        for pr in prs:
            if pr[1] in tournaments:
                if as_str:
                    pr = (pr[0], nx.tournament_str(pr[1]))
                pairs.append(pr)
        return pairs

    def pairs_split(self, as_str=True):
        "Split pairs into two lists: names and tournaments"
        pairs = self.pairs(as_str)
        name, tournament = zip(*pairs)
        return name, tournament

    def __contains__(self, pair):
        "Is `pair` already in prediction? True or False"
        pair = self.make_pair(*pair)
        return pair in self.df

    def make_pair(self, name, tournament):
        "Combine `name` and `tournament` into a pair (dataframe column name)"
        if not nx.isstring(name):
            raise ValueError("`name` must be a string")
        return (name, nx.tournament_int(tournament))

    def drop_pair(self, pair):
        "Drop pair (tuple) or list of pairs from prediction; return copy"
        if self.df is None:
            raise ValueError("Cannot drop a pair from an empty prediction")
        if isinstance(pair, list):
            pairs = [self.make_pair(*p) for p in pair]
        elif isinstance(pair, tuple):
            pairs = [self.make_pair(*pair)]
        pairs0 = self.pairs(as_str=False)
        for p in pairs:
            if p not in pairs0:
                raise ValueError('cannot drop pair that does not exist')
        df = self.df.drop(columns=pairs)
        return Prediction(df)

    # merge -----------------------------------------------------------------

    def merge_arrays(self, ids, y, name, tournament):
        "Merge numpy arrays `ids` and `y` with name `name`"
        pair = self.make_pair(name, tournament)
        df = pd.DataFrame(data=y, columns=[pair], index=ids)
        prediction = Prediction(df)
        return self.merge(prediction)

    def merge(self, prediction):
        "Merge prediction"
        return merge_predictions([self, prediction])

    def __add__(self, prediction):
        "Merge predictions"
        return self.merge(prediction)

    def __iadd__(self, prediction):
        "Merge predictions"
        return self.merge(prediction)

    # io --------------------------------------------------------------------

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
        with warnings.catch_warnings():
            # pytables warns (through pandas) that pairs will be pickled which
            # is a performance hit but there are never a large number of models
            # and hence pairs in a prediction object
            warnings.simplefilter("ignore")
            if compress:
                self.df.to_hdf(path_or_buf, HDF_PREDICTION_KEY,
                               complib='zlib', complevel=4)
            else:
                self.df.to_hdf(path_or_buf, HDF_PREDICTION_KEY)

    def to_csv(self, path_or_buf, decimals=6, verbose=False):
        "Save a csv file of predictions; prediction must contain only one pair"
        if self.shape[1] != 1:
            raise ValueError("prediction must contain a single pair")
        tourn = self.tournaments(as_str=True)[0]
        df = self.df.iloc[:, 0].to_frame('probability_' + tourn)
        df.index.rename('id', inplace=True)
        float_format = "%.{}f".format(decimals)
        df.to_csv(path_or_buf, float_format=float_format)
        if verbose:
            print("Save {}".format(path_or_buf))

    # metrics ---------------------------------------------------------------

    def summary(self, data, tournament=None, round_output=True):
        "Performance summary of prediction object that contains a single pair"

        if self.shape[1] != 1:
            raise ValueError("prediction must contain a single pair")

        # metrics
        metrics, regions = metrics_per_era(data, self, tournament,
                                           region_as_str=True,
                                           split_pairs=False)
        metrics = metrics.drop(['era', 'pair'], axis=1)

        # additional metrics
        region_str = ', '.join(regions)
        nera = metrics.shape[0]
        logloss = metrics['logloss']
        consis = (logloss < LOGLOSS_BENCHMARK).mean()

        # summary of metrics
        if tournament is None:
            t_str = self.tournaments(as_str=True)[0]
        else:
            t_str = nx.tournament_str(tournament)
        m1 = metrics.mean(axis=0).tolist() + ['tourn', t_str]
        m2 = metrics.std(axis=0).tolist() + ['region', region_str]
        m3 = metrics.min(axis=0).tolist() + ['eras', nera]
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

    def summaries(self, data, tournament=None, round_output=True,
                  display=True):
        "Dictionary of performance summaries of predictions"
        df_dict = {}
        for pair in self.pairs(as_str=False):
            df_dict[pair] = self[pair].summary(data, tournament,
                                               round_output=round_output)
            if display:
                print('{}, {}'.format(pair[0], nx.tournament_str(pair[1])))
                print(df_dict[pair])
        return df_dict

    def metric_per_era(self, data, tournament=None, metric='logloss',
                       era_as_str=True, split_pairs=True):
        "DataFrame containing given metric versus era (as index)"
        df = self.metrics_per_era(data, tournament=tournament,
                                  metrics=[metric],
                                  era_as_str=True, split_pairs=True)
        df = df.pivot(columns='pair', values=metric)
        df.columns.name = None
        return df

    def metrics_per_era(self, data, tournament=None,
                        metrics=['logloss', 'auc', 'acc', 'ystd'],
                        era_as_str=True, split_pairs=True):
        "DataFrame containing given metrics versus era (as index)"
        metrics, regions = metrics_per_era(data, self, tournament,
                                           columns=metrics,
                                           era_as_str=era_as_str)
        metrics.index = metrics['era']
        metrics = metrics.drop(['era'], axis=1)
        if split_pairs:
            pair = metrics['pair']
            metrics = metrics.drop('pair', axis=1)
            metrics.insert(0, 'pair', pair)
        else:
            metrics = metrics.drop('pair', axis=1)
        return metrics

    def metric_per_tournament(self, data, metric='logloss'):
        "DataFrame containing given metric versus tournament"
        dfs = []
        for t_int, t_name in nx.tournament_iter():
                df, info = metrics_per_name(data, self, t_int,
                                            columns=[metric],
                                            split_pairs=False)
                df.columns = [t_name]
                dfs.append(df)
        df = pd.concat(dfs, axis=1)
        df.insert(df.shape[1], 'mean', df.mean(axis=1))
        df = df.sort_values('mean')
        return df

    def performance(self, data, tournament=None, era_as_str=True,
                    region_as_str=True,
                    columns=['logloss', 'auc', 'acc', 'ystd', 'sharpe',
                             'consis'], sort_by='logloss'):
        df, info = metrics_per_name(data,
                                    self,
                                    tournament,
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
                raise ValueError("`sort_by` method not recognized")
        return df

    def performance_mean(self, data, mean_of='name', era_as_str=True,
                         region_as_str=True,
                         columns=['logloss', 'auc', 'acc', 'ystd', 'sharpe',
                                  'consis'], sort_by='logloss'):
        "Mean performance averaged across names (default) or tournaments,"
        df = self.performance(data, era_as_str=era_as_str,
                              region_as_str=region_as_str, columns=columns)
        if mean_of == 'name':
            g = df.groupby('name')
            df = g.mean()
            c = g.count()
            df.insert(0, 'N', c['tournament'])
        elif mean_of == 'tournament':
            g = df.groupby('tournament')
            df = g.mean()
            c = g.count()
            df.insert(0, 'N', c['name'])
        else:
            raise ValueError("`across` must be 'name' or 'tournament'")
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
                raise ValueError("`sort_by` method not recognized")
        return df

    def dominance(self, data, tournament=None, sort_by='logloss'):
        "Mean (across eras) of fraction of models bested per era"
        columns = ['logloss', 'auc', 'acc']
        mpe, regions = metrics_per_era(data, self, tournament, columns=columns)
        dfs = []
        for i, col in enumerate(columns):
            pivot = mpe.pivot(index='era', columns='pair', values=col)
            pairs = pivot.columns.tolist()
            a = pivot.values
            n = a.shape[1] - 1.0
            if n == 0:
                raise ValueError("Must have at least two pairs")
            m = []
            for j in range(pivot.shape[1]):
                if col == 'logloss':
                    z = (a[:, j].reshape(-1, 1) < a).sum(axis=1) / n
                else:
                    z = (a[:, j].reshape(-1, 1) > a).sum(axis=1) / n
                m.append(z.mean())
            df = pd.DataFrame(data=m, index=pairs, columns=[col])
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        df = add_split_pairs(df)
        df = df.sort_values([sort_by], ascending=[False])
        return df

    def concordance(self, data):
        "Less than 0.12 is passing; data should be the full dataset."
        return concordance(data, self)

    def check(self, data, verbose=True):
        """
        Run Numerai upload checks.

        Parameters
        ----------
        data : nx.Data
            Data object of Numerai dataset.
        verbose : bool
            By default, True, output is printed to stdout.

        Returns
        -------
        check : dict
            A dictionary where the keys are the (name, tournament) pairs and
            the values are Pandas DataFrames that contain the results of the
            checks.
        """

        # calc example predictions
        example_y = {}
        for tournament in self.tournaments(as_str=False):
            ep = nx.production(nx.example_predictions(), data,
                               tournament=tournament, verbosity=0)
            ep = ep.loc[self.ids]
            example_y[tournament] = ep.y[:, 0]

        df_dict = {}
        columns = ['validation', 'test', 'live', 'all', 'pass']
        data = data.loc[self.ids]
        regions = data.region
        pairs = list(self.pairs(as_str=False))

        # check each model, tournament pair
        for pair in pairs:
            print('{}, {}'.format(pair[0], nx.tournament_str(pair[1])))
            df = pd.DataFrame(columns=columns)
            idx = pairs.index(pair)
            y = self.y[:, idx]
            for region in ('validation', 'test', 'live', 'all'):
                yexi = example_y[pair[1]]
                if region == 'all':
                    yi = y
                else:
                    idx = regions == region
                    yi = y[idx]
                    yexi = yexi[idx]
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

            df_dict[pair] = df

        return df_dict

    def compare(self, data, prediction, tournament=None):
        "Compare performance of predictions with the same names"
        pairs = []
        for pair in self.pairs(as_str=False):
            if pair in prediction:
                pairs.append(pair)
        cols = ['logloss1', 'logloss2', 'win1',
                'corr', 'maxdiff', 'ystd1', 'ystd2']
        comp = pd.DataFrame(columns=cols, index=pairs)
        if len(pairs) == 0:
            return comp
        ids = data.ids
        df1 = self.loc[ids]
        df2 = prediction.loc[ids]
        p1 = self[pairs]
        p2 = prediction[pairs]
        m1 = p1.metrics_per_era(data, tournament, metrics=['logloss'],
                                era_as_str=False)
        m2 = p2.metrics_per_era(data, tournament, metrics=['logloss'],
                                era_as_str=False)
        for i, pair in enumerate(pairs):

            m1i = m1[(m1.name == pair[0]) &
                     (m1.tournament == nx.tournament_str(pair[1]))]
            m2i = m2[(m2.name == pair[0]) &
                     (m2.tournament == nx.tournament_str(pair[1]))]

            if (m1i.index != m2i.index).any():
                raise IndexError("Can only handle aligned eras")

            logloss1 = m1i.logloss.mean()
            logloss2 = m2i.logloss.mean()
            win1 = (m1i.logloss < m2i.logloss).mean()

            y1 = df1[pair].y.reshape(-1)
            y2 = df2[pair].y.reshape(-1)

            corr = np.corrcoef(y1, y2)[0, 1]
            maxdiff = np.abs(y1 - y2).max()
            ystd1 = y1.std()
            ystd2 = y2.std()

            m = [logloss1, logloss2, win1, corr, maxdiff, ystd1, ystd2]
            comp.iloc[i] = m

        comp = add_split_pairs(comp)

        return comp

    # indexing --------------------------------------------------------------

    def __getitem__(self, index):
        "Prediction indexing is by model pair(s)"
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("When indexing by tuple must be length 2")
            if isinstance(index[0], slice):
                if not is_none_slice(index[0]):
                    raise IndexError("Slces must be slice(None, None, None,)")
                pairs1 = self.pairs(as_str=False)
            elif nx.isstring(index[0]):
                pairs1 = self.pairs_with_name(index[0], as_str=False)
            else:
                raise IndexError("indexing method not recognized")
            if isinstance(index[1], slice):
                if not is_none_slice(index[1]):
                    raise IndexError("Slces must be slice(None, None, None,)")
                pairs2 = self.pairs(as_str=False)
            elif nx.isint(index[1]):
                pairs2 = self.pairs_with_tournament(index[1], as_str=False)
            elif nx.isstring(index[1]):
                pairs2 = self.pairs_with_tournament(index[1], as_str=False)
            else:
                raise IndexError("indexing method not recognized")
            pairs = []
            for pair in pairs1:
                if pair in pairs2:
                    pairs.append(pair)
            p = Prediction(pd.DataFrame(data=self.df[pairs]))
        elif nx.isstring(index):
            pairs = self.pairs_with_name(index, as_str=False)
            p = Prediction(self.df[pairs])
        else:
            # assume an iterable of tuple pairs
            idx = []
            for ix in index:
                if len(ix) != 2:
                    msg = "Expecting list of tuple pairs with length 2"
                    raise IndexError(msg)
                idx.append((ix[0], nx.tournament_int(ix[1])))
            p = Prediction(self.df[idx])
        return p

    def __setitem__(self, index, prediction):
        "Add (or replace) a prediction by pair"
        if prediction.df.shape[1] != 1:
            raise ValueError("Can only insert a single model at a time")
        if not isinstance(index, tuple):
            raise IndexError('`index` must be a tuple pair')
        if len(index) != 2:
            raise IndexError('`index` must be a tuple pair of length 2')
        prediction.df.columns = [index]
        self.df = self.merge(prediction).df

    # utilities -------------------------------------------------------------

    def iter(self):
        "Yield a prediction object with only one model at a time"
        for pair in self.pairs(as_str=False):
            yield self[pair]

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
        df = self.pairs_df()
        df = df.replace(to_replace=True, value='x')
        df = df.replace(to_replace=False, value='')
        return df.to_string(index=True)


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
        raise ValueError("csv file must contain one column of predictions")
    tournament = nx.tournament_int(df.columns[0].split('_')[-1])
    if name is None:
        name = os.path.split(filename)[-1]
        if name.endswith('.csv'):
            name = name[:-4]
    df.columns = [(name, tournament)]
    return Prediction(df)


def load_example_predictions(data_zip, tournament):
    "Load example predictions from Numerai zip archive"
    zf = zipfile.ZipFile(data_zip)
    tourn_name = nx.tournament_str(tournament)
    filename = EXAMPLE_PREDICTIONS.format(tourn_name)
    df = pd.read_csv(zf.open(filename), header=0, index_col=0)
    df.columns = ['example_predictions_{}'.format(tourn_name)]
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
        for pair in pi.pairs(as_str=False):
            p = _merge_predictions(p, pi[pair])
    return p


def _merge_predictions(prediction1, prediction2):
    "Merge a possibly multi-name prediction1 with a single-name prediction2"
    if prediction2.shape[1] != 1:
        raise ValueError("`prediction2` must contain a single name")
    pair = prediction2.pairs(as_str=False)[0]
    if prediction1.df is None:
        # empty prediction
        df = prediction2.df
    elif pair not in prediction1:
        # inserting predictions from a model not already in report
        df = pd.merge(prediction1.df, prediction2.df, how='outer',
                      left_index=True, right_index=True)
    else:
        # add more ys from a model whose name already exists
        y = prediction1.df[pair]
        y = y.dropna()
        s = prediction2.df.iloc[:, 0]
        s = s.dropna()
        s = pd.concat([s, y], join='outer', ignore_index=False,
                      verify_integrity=True)
        dfnew = pd.DataFrame(s, columns=[pair])
        df = pd.merge(prediction1.df, dfnew, how='outer', on=[pair],
                      left_index=True, right_index=True)
        df[pair] = dfnew
    return Prediction(df)
