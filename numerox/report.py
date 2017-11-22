import os
import glob

import pandas as pd
import numpy as np

from numerox.prediction import load_prediction
from numerox.metrics import metrics_per_era


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
        df = self.performance_df(data)
        if sort_by == 'logloss':
            df = df.sort_values(by='logloss', ascending=True)
        elif sort_by == 'auc':
            df = df.sort_values(by='auc', ascending=False)
        elif sort_by == 'acc':
            df = df.sort_values(by='acc', ascending=False)
        elif sort_by == 'ystd':
            df = df.sort_values(by='ystd', ascending=False)
        elif sort_by == 'consis':
            df = df.sort_values(by=['consis', 'logloss'],
                                ascending=[False, True])
        else:
            raise ValueError("`sort_by` name not recognized")
        df = df.round(decimals={'logloss': 6, 'auc': 4, 'acc': 4, 'ystd': 4,
                                'consis': 4})
        with pd.option_context('display.colheader_justify', 'left'):
            print(df.to_string(index=False))

    def performance_df(self, data):

        # calc performance
        metrics = metrics_per_era(data, self)
        regions = data.unique_region().tolist()
        models = list(metrics.keys())
        nera = metrics[models[0]].shape[0]
        regera = ', '.join(regions) + '; %d' % nera + ' eras'

        # create dataframe of performance
        cols = ['logloss', 'auc', 'acc', 'ystd', 'consis', '(%s)' % regera]
        df = pd.DataFrame(columns=cols)
        for i, model in enumerate(models):
            metric_df = metrics[model]
            metric = metric_df.mean(axis=0).tolist()
            consis = (metric_df['logloss'] < np.log(2)).mean()
            metric.extend([consis, model])
            df.loc[i] = metric

        return df


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
