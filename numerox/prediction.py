import pandas as pd
import numpy as np

from numerox.metrics import calc_metrics

HDF_PREDICTION_KEY = 'numerox_prediction'


class Prediction(object):

    def __init__(self, df=None):
        self.df = df

    @property
    def ids(self):
        "Copy of ids as a numpy str array or None is empty"
        if self.df is None:
            return None
        return self.df.index.values.astype(str)

    @property
    def yhat(self):
        "Copy of yhat as a 1d numpy array or None is empty"
        if self.df is None:
            return None
        return self.df['yhat'].values.copy()

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

    def to_csv(self, path_or_buf=None, decimals=8):
        "Save a csv file of predictions for later upload to Numerai"
        float_format = "%.{}f".format(decimals)
        return self.df.to_csv(path_or_buf, float_format=float_format)

    def save(self, path_or_buf, compress=True):
        "Save prediction as an hdf archive; raises if nothing to save"
        if self.df is None:
            raise ValueError("Prediction object is empty; nothing to save")
        if compress:
            self.df.to_hdf(path_or_buf, HDF_PREDICTION_KEY,
                           complib='zlib', complevel=4)
        else:
            self.df.to_hdf(path_or_buf, HDF_PREDICTION_KEY)

    def performance(self, data):
        metrics = calc_metrics(data, self.df)
        metrics = metrics['yhat']
        regions = data.unique_region().tolist()
        regions = ', '.join(regions)
        print("      logloss   auc     acc     ystd")
        fmt = "{:<4}  {:8.6f}  {:6.4f}  {:6.4f}  {:6.4f}{extra}"
        extra = "  |  {:<7}  {:<}".format('region', regions)
        print(fmt.format('mean', *metrics.mean(axis=0), extra=extra))
        extra = "  |  {:<7}  {:<}".format('eras', metrics.shape[0])
        print(fmt.format('std', *metrics.std(axis=0), extra=extra))
        consistency = (metrics['logloss'] < np.log(2)).mean()
        extra = "  |  {:<7}  {:<.4f}".format('consis', consistency)
        print(fmt.format('min', *metrics.min(axis=0), extra=extra))
        prctile = np.percentile(metrics['logloss'], 75)
        extra = "  |  {:<7}  {:<.4f}".format('75th', prctile)
        print(fmt.format('max', *metrics.max(axis=0), extra=extra))

    def copy(self):
        "Copy of prediction"
        if self.df is None:
            return Prediction(None)
        return Prediction(self.df.copy(deep=True))

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

    def _column_list(self):
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


if __name__ == '__main__':
    import numerox as nx
    data = nx.load_data('/data/nx/numerai_dataset_20171024.hdf')
    p = nx.load_prediction('/data/nx/pred/extratrees_nfeature2.pred')
    p.performance(data['validation'])
    """
    model = nx.model.logistic()
    prediction1 = nx.backtest(model, data, verbosity=1)
    prediction2 = nx.production(model, data)
    """
    """
    prediction = prediction1 + prediction2
    print prediction
    prediction.performance(data)
    prediction.save('/data/nx/pred/logistic_1e-5.pred')
    """

    """
    for c in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        print c
        model = nx.model.logistic(c)
        prediction1 = nx.backtest(model, data, verbosity=1)
        prediction2 = nx.production(model, data)
        prediction = prediction1 + prediction2
        prediction.save('/data/nx/pred/logistic_{:.0e}.pred'.format(c))

    for n in (2, 3, 5, 7):
        model = nx.model.extratrees(nfeatures=n)
        prediction1 = nx.backtest(model, data, verbosity=1)
        prediction2 = nx.production(model, data)
        prediction = prediction1 + prediction2
        prediction.save('/data/nx/pred/extratrees_nfeature{}.pred'.format(n))
    """
