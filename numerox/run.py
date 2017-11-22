import os
import time
import pprint

from numerox import Prediction, TournamentSplitter, CVSplitter


def production(model, data, verbosity=2):
    "Fit a model with train data; make prediction on tournament data"
    splitter = TournamentSplitter(data)
    prediction = run(model, splitter, verbosity=verbosity)
    return prediction


def backtest(model, data, kfold=5, seed=0, verbosity=2):
    "K-fold cross validation of model through train data"
    splitter = CVSplitter(data, kfold=kfold, seed=seed, train_only=True)
    prediction = run(model, splitter, verbosity)
    return prediction


def run(model, splitter, verbosity=2):
    "Run a single model through a data splitter"
    t0 = time.time()
    if verbosity > 2:
        print(splitter)
    if verbosity > 0:
        pprint.pprint(model)
    data = None
    prediction = Prediction()
    for data_fit, data_predict in splitter:
        ids, yhat = model.fit_predict(data_fit, data_predict)
        prediction.append(ids, yhat)
        if data is None:
            data = data_predict.copy()
        else:
            data = data + data_predict
        if verbosity > 1:
            prediction.performance(data.region_isnotin(['test', 'live']))
    if verbosity == 1:
        prediction.performance(data.region_isnotin(['test', 'live']))
    if verbosity > 2:
        minutes = (time.time() - t0) / 60
        print('Done in {:.2f} minutes'.format(minutes))
    return prediction


class Runner(object):
    """
    Run one or more models through a data splitter

    `run_list` is a list of dictionaries with keys 'model' and optionally
    'prediction_file' for location to save hdf5 prediction and `cvs_file` for
    location to save cvs file. If `save_dir` is not None then it will be
    prepended to `prediction_file` and `csv_file`.
    """

    def __init__(self, run_list, splitter, save_dir=None, verbosity=2):
        self.run_list = run_list
        self.splitter = splitter
        self.save_dir = save_dir
        self.verbosity = verbosity

    def run(self):
        "run through all models without yielding prediction object"
        for prediction in self.run_iter():
            pass

    def run_iter(self):
        "Generator that runs through all models and yields prediction object"
        # possible future feature: upload if r['upload_token'] is not None?
        for r in self.run_list:
            self.splitter.reset()
            prediction = run(r['model'], self.splitter, self.verbosity)
            if 'prediction_file' in r and r['prediction_file'] is not None:
                filename = r['prediction_file']
                if self.save_dir is not None:
                    filename = os.path.join(self.save_dir, filename)
                prediction.save(filename)
            if 'csv_file' in r and r['csv_file'] is not None:
                filename = r['csv_file']
                if self.save_dir is not None:
                    filename = os.path.join(self.save_dir, filename)
                prediction.to_csv(filename)
            yield prediction
