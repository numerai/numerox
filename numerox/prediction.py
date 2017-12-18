import pandas as pd
import numpy as np

from numerox.metrics import metrics_per_era

HDF_PREDICTION_KEY = 'numerox_prediction'


class Prediction(object):

    def __init__(self, df=None):
        self.df = df

    @property
    def ids(self):
        "View of ids as a numpy str array"
        return self.df.index.values

    @property
    def yhat(self):
        "View of yhat as a 1d numpy float array"
        return self.df['yhat'].values

    def yhatnew(self, y_array):
        "Copy of prediction but with prediction.yhat=`y_array`"
        if y_array.shape[0] != len(self):
            msg = "`y_array` must have the same number of rows as prediction"
            raise ValueError(msg)
        if y_array.ndim != 1:
            raise ValueError("`y_array` must be 1 dimensional")
        df = pd.DataFrame(data=np.empty((y_array.shape[0],), dtype=np.float64),
                          index=self.df.index.copy(deep=True),
                          columns=['yhat'])
        df['yhat'] = y_array
        return Prediction(df)

    def append(self, ids, yhat):
        df = pd.DataFrame(data={'yhat': yhat}, index=ids)
        if self.df is None:
            df.index.rename('ids', inplace=True)
        else:
            try:
                df = pd.concat([self.df, df], verify_integrity=True)
            except ValueError:
                # pandas doesn't raise expected IndexError and for our large
                # number of y, the id overlaps that it prints can be very long
                raise IndexError("Overlap in ids found")
        self.df = df

    def to_csv(self, path_or_buf=None, decimals=6, verbose=False):
        "Save a csv file of predictions for later upload to Numerai"
        df = self.df.copy()
        df.index.rename('id', inplace=True)
        df.rename(columns={'yhat': 'probability'}, inplace=True)
        float_format = "%.{}f".format(decimals)
        df.to_csv(path_or_buf, float_format=float_format)
        if verbose:
            print("Save {}".format(path_or_buf))

    def save(self, path_or_buf, compress=True):
        "Save prediction as an hdf archive; raises if nothing to save"
        if self.df is None:
            raise ValueError("Prediction object is empty; nothing to save")
        if compress:
            self.df.to_hdf(path_or_buf, HDF_PREDICTION_KEY,
                           complib='zlib', complevel=4)
        else:
            self.df.to_hdf(path_or_buf, HDF_PREDICTION_KEY)

    def consistency(self, data):
        "Consistency over eras in `data`"
        logloss = self.metrics_per_era(data, metrics=['logloss'])
        c = (logloss.values < np.log(2)).mean()
        return c

    def metrics_per_era(self, data, metrics=['logloss'], era_as_str=True):
        "DataFrame containing given metrics versus era"
        metrics, regions = metrics_per_era(data, self, columns=metrics,
                                           era_as_str=era_as_str)
        metrics.index = metrics['era']
        metrics = metrics.drop(['era', 'model'], axis=1)
        return metrics

    def performance(self, data):
        df = self.performance_df(data)
        df = df.round(decimals={'logloss': 6, 'auc': 4, 'acc': 4, 'ystd': 4})
        with pd.option_context('display.colheader_justify', 'left'):
            print(df.to_string(index=True))

    def performance_df(self, data):

        # metrics
        metrics, regions = metrics_per_era(data, self, region_as_str=True)
        metrics = metrics.drop(['era', 'model'], axis=1)

        # additional metrics
        region_str = ', '.join(regions)
        nera = metrics.shape[0]
        logloss = metrics['logloss']
        consis = (logloss < np.log(2)).mean()
        sharpe = (np.log(2) - logloss).mean() / logloss.std()

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

        return df

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

    @property
    def size(self):
        if self.df is None:
            return 0
        return self.df.size

    @property
    def shape(self):
        if self.df is None:
            return tuple()
        return self.df.shape

    def __len__(self):
        "Number of rows"
        if self.df is None:
            return 0
        return self.df.__len__()

    def column_list(self):
        "Return column names of dataframe as a list"
        return self.df.columns.tolist()

    def __add__(self, other_prediction):
        "Concatenate two prediction objects that have no overlap in ids"
        return concat_prediction([self, other_prediction])

    def __repr__(self):
        if self.df is None:
            return ''
        t = []
        fmt = '{:<10}{:>13.6f}'
        y = self.df.yhat
        t.append(fmt.format('mean', y.mean()))
        t.append(fmt.format('std', y.std()))
        t.append(fmt.format('min', y.min()))
        t.append(fmt.format('max', y.max()))
        t.append(fmt.format('rows', len(self.df)))
        t.append(fmt.format('nulls', y.isnull().sum()))
        return '\n'.join(t)


def load_prediction(file_path):
    "Load prediction object from hdf archive; return Prediction"
    df = pd.read_hdf(file_path, key=HDF_PREDICTION_KEY)
    return Prediction(df)


def concat_prediction(predictions):
    "Concatenate list-like of prediction objects; ids must not overlap"
    dfs = [d.df for d in predictions]
    try:
        df = pd.concat(dfs, verify_integrity=True, copy=True)
    except ValueError:
        # pandas doesn't raise expected IndexError and for our large data
        # object, the id overlaps that it prints can be very long so
        raise IndexError("Overlap in ids found")
    return Prediction(df)
