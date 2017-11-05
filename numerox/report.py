import os
import glob

import pandas as pd
import numpy as np

from numerox.prediction import load_prediction
from numerox.metrics import calc_metrics


class Report(object):

    def __init__(self, df=None):
        self.df = df

    @staticmethod
    def from_data_predictions(prediction_dict):
        dfs = []
        for model in prediction_dict:
            df = prediction_dict[model].df
            df.rename(columns={'yhat': model}, inplace=True)
            dfs.append(df)
        df = pd.concat(dfs, axis=1, verify_integrity=True, copy=False)
        return Report(df)

    def performance(self, data, sort_by='logloss'):
        metrics = calc_metrics(data, self.df)
        regions = data.unique_region().tolist()
        nera = metrics[metrics.keys()[0]].shape[0]
        regera = ', '.join(regions) + '; %d' % nera + ' eras'
        print("logloss   auc     acc     ystd    consis  (%s)" % regera)
        fmt = "{:.6f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {model:<}"
        for model in metrics:
            metric_df = metrics[model]
            metric = metric_df.mean(axis=0)
            metric['consis'] = (metric_df['logloss'] < np.log(2)).mean()
            print(fmt.format(*metric, model=model))


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
    report = Report.from_data_predictions(predictions)
    return report


if __name__ == '__main__':
    import numerox as nx
    data = nx.load_data('/data/nx/numerai_dataset_20171024.hdf')
    report = nx.report.load_report('/data/nx/pred')
    report.performance(data['train'])
