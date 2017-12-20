import os
import glob

import pandas as pd
import numpy as np

from numerox.prediction import load_prediction
from numerox.metrics import metrics_per_era
from numerox.metrics import metrics_per_model


class Report(object):

    def __init__(self, df=None):
        self.df = df

    @property
    def models(self):
        return self.df.columns.tolist()

    def append_prediction(self, prediction, model_name):
        df = prediction.df
        df = df.rename(columns={'yhat': model_name})
        dfs = [self.df, df]
        df = pd.concat(dfs, axis=1, verify_integrity=True, copy=False)
        self.df = df

    def append_prediction_dict(self, prediction_dict):
        dfs = []
        for model in prediction_dict:
            df = prediction_dict[model].df
            df.rename(columns={'yhat': model}, inplace=True)
            dfs.append(df)
        df = pd.concat(dfs, axis=1, verify_integrity=True, copy=False)
        self.df = df

    def performance_per_era(self, data, model_name):
        print(model_name)
        df = self.df[model_name].to_frame(model_name)
        df = metrics_per_era(data, Report(df))[model_name]
        df = df.round(decimals={'logloss': 6, 'auc': 4, 'acc': 4, 'ystd': 4})
        with pd.option_context('display.colheader_justify', 'left'):
            print(df.to_string())

    def performance(self, data, sort_by='logloss'):
        df, info = self.performance_df(data)
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
            df = df.sort_values(by=['consis', 'logloss'],
                                ascending=[False, True])
        else:
            raise ValueError("`sort_by` name not recognized")
        df = df.round(decimals={'logloss': 6, 'auc': 4, 'acc': 4, 'ystd': 4,
                                'sharpe': 4, 'consis': 4})
        info_str = ', '.join(info['region']) + '; '
        info_str += '{} eras'.format(len(info['era']))
        print(info_str)
        with pd.option_context('display.colheader_justify', 'left'):
            print(df.to_string(index=True))

    def performance_df(self, data, era_as_str=True, region_as_str=True):
        cols = ['logloss', 'auc', 'acc', 'ystd', 'sharpe', 'consis']
        metrics, info = metrics_per_model(data,
                                          self,
                                          columns=cols,
                                          era_as_str=era_as_str,
                                          region_as_str=region_as_str)
        return metrics, info

    def dominance(self, data, sort_by='logloss'):
        "Mean (across eras) of fraction of models bested per era"
        df = self.dominance_df(data)
        df = df.sort_values([sort_by], ascending=[False])
        df = df.round(decimals=4)
        with pd.option_context('display.colheader_justify', 'left'):
            print(df.to_string(index=True))

    def dominance_df(self, data):
        "Mean (across eras) of fraction of models bested per era"
        columns = ['logloss', 'auc', 'acc']
        mpe, regions = metrics_per_era(data, self, columns=columns)
        dfs = []
        for i, col in enumerate(columns):
            pivot = mpe.pivot(index='era', columns='model', values=col)
            models = pivot.columns.tolist()
            a = pivot.values
            n = a.shape[1] - 1.0
            if n == 0:
                raise ValueError("Must have at least two models")
            m = []
            for j in range(pivot.shape[1]):
                if col == 'logloss':
                    z = (a[:, j].reshape(-1, 1) < a).sum(axis=1) / n
                else:
                    z = (a[:, j].reshape(-1, 1) > a).sum(axis=1) / n
                m.append(z.mean())
            df = pd.DataFrame(data=m, index=models, columns=[col])
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        return df

    def correlation(self, model_name=None):
        "Correlation of predictions; by default reports given for each model"
        if model_name is None:
            models = self.models
        else:
            models = [model_name]
        z = self.df.values
        zmodels = self.models
        idx = np.isfinite(z.sum(axis=1))
        z = z[idx]
        z = (z - z.mean(axis=0)) / z.std(axis=0)
        for model in models:
            print(model)
            idx = zmodels.index(model)
            corr = np.dot(z[:, idx], z) / z.shape[0]
            index = (-corr).argsort()
            for ix in index:
                zmodel = zmodels[ix]
                if model != zmodel:
                    print("   {:.4f} {}".format(corr[ix], zmodel))


def load_report(prediction_dir, extension='pred'):
    "Load Prediction objects (hdf) in `prediction_dir`; return Report object"
    original_dir = os.getcwd()
    os.chdir(prediction_dir)
    predictions = {}
    try:
        for filename in glob.glob("*{}".format(extension)):
            prediction = load_prediction(filename)
            model = filename[:-len(extension) - 1]
            predictions[model] = prediction
    finally:
        os.chdir(original_dir)
    report = Report()
    report.append_prediction_dict(predictions)
    return report
