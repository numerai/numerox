import pandas as pd
import numpy as np

from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

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
        metrics = calc_metrics(data, self)
        for region in metrics:
            metric = metrics[region]
            if metric is None:
                continue
            print("      logloss   auc     acc     ystd")
            fmt = "{:<4}  {:.6f}  {:.4f}  {:.4f}  {:.4f}{extra}"
            extra = "  |  {:<7}  {:<}".format('region', region)
            print(fmt.format('mean', *metric.mean(axis=0), extra=extra))
            extra = "  |  {:<7}  {:<}".format('eras', metric.shape[0])
            print(fmt.format('std', *metric.std(axis=0), extra=extra))
            consistency = (metric['logloss'] < np.log(2)).mean()
            extra = "  |  {:<7}  {:<.4f}".format('consis', consistency)
            print(fmt.format('min', *metric.min(axis=0), extra=extra))
            prctile = np.percentile(metric['logloss'], 75)
            extra = "  |  {:<7}  {:<.4f}".format('75th', prctile)
            print(fmt.format('max', *metric.max(axis=0), extra=extra))

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


def calc_metrics(data, prediction):

    # merge prediction with data (remove features x)
    yhat_df = prediction.df.dropna()
    data = data.region_isin(['train', 'validation'])  # speed optimzation
    data_df = data.df[['era', 'region', 'y']]  # speed optimzation
    df = pd.merge(data_df, yhat_df, left_index=True, right_index=True,
                  how='right')

    metrics = {'train': None, 'validation': None}
    for region in metrics:

        # pull out region
        idx = df.region.isin([region])
        df_region = df[idx]
        if len(df_region) == 0:
            continue

        # calc metrics for each era
        unique_eras = df_region.era.unique()
        ys = df_region[['y', 'yhat']]
        metric = []
        for era in unique_eras:
            idx = df_region.era.isin([era])
            ys_era = ys[idx]
            y = ys_era['y'].values
            yhat = ys_era['yhat'].values
            m = _calc_metrics_1era(y, yhat)
            metric.append(m)
        metric = np.array(metric)

        # jam into a dataframe
        columns = ['logloss', 'auc', 'acc', 'ystd']
        metric = pd.DataFrame(metric, columns=columns, index=unique_eras)

        metrics[region] = metric

    return metrics


def _calc_metrics_1era(y, yhat):
    "standard metrics for `yhat` given actual outcome `y`"
    m = []
    m.append(log_loss(y, yhat))
    m.append(roc_auc_score(y, yhat))
    yh = np.zeros(yhat.size)
    yh[yhat >= 0.5] = 1
    m.append(accuracy_score(y, yh))
    m.append(yhat.std())
    return m


if __name__ == '__main__':
    import numerox as nx
    data = nx.load_data('/data/nx/numerai_dataset_20171024.hdf')
    model = nx.model.logistic()
    prediction1 = nx.backtest(model, data, verbosity=1)
    prediction2 = nx.production(model, data)
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
