import os
import glob

import pandas as pd

from numerox.prediction import load_prediction


class Report(object):

    def __init__(self, df=None):
        self.df = df

    @staticmethod
    def from_data_predictions(data, prediction_dict):
        dfs = []
        for model in prediction_dict:
            df = prediction_dict[model].df
            df.rename(columns={'yhat': model}, inplace=True)
            dfs.append(df)
        df = pd.concat(dfs, axis=1, verify_integrity=True, copy=False)
        ery = data.df[['era', 'region', 'y']]
        df = pd.merge(ery, df, left_index=True, right_index=True, how='right')
        return Report(df)

    def performance(self):
        pass


def load_report(data, prediction_dir, extension='pred'):
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
    report = Report.from_data_predictions(data, predictions)
    return report
