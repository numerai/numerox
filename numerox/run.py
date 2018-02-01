import time
import pprint

from numerox import Prediction, TournamentSplitter, CVSplitter


def production(model, data, name=None, verbosity=2):
    "Fit a model with train data; make prediction on tournament data"
    splitter = TournamentSplitter(data)
    prediction = run(model, splitter, name, verbosity=verbosity)
    return prediction


def backtest(model, data, name=None, kfold=5, seed=0, verbosity=2):
    "K-fold cross validation of model through train data"
    splitter = CVSplitter(data, kfold=kfold, seed=seed, train_only=True)
    prediction = run(model, splitter, name, verbosity)
    return prediction


def run(model, splitter, name=None, verbosity=2):
    "Run a single model through a data splitter"
    t0 = time.time()
    if name is None:
        name = model.__class__.__name__
    else:
        if verbosity > 2:
            print(name)
    if verbosity > 2:
        print(splitter)
    if verbosity > 0:
        pprint.pprint(model)
    data = None
    prediction = Prediction()
    for data_fit, data_predict in splitter:
        if verbosity > 0:
            if data is None:
                data = data_predict.copy()
            else:
                data = data + data_predict
        # the following line of code hides from your model the y
        # that you are trying to predict to prevent accidental cheating
        data_predict = data_predict.y_to_nan()
        ids, yhat = model.fit_predict(data_fit, data_predict)
        prediction = prediction.merge_arrays(ids, yhat, name)
        if verbosity > 1:
            print(prediction.summary(data.region_isnotin(['test', 'live'])))
    if verbosity == 1:
        print(prediction.summary(data.region_isnotin(['test', 'live'])))
    if verbosity > 1:
        minutes = (time.time() - t0) / 60
        print('Done in {:.2f} minutes'.format(minutes))
    return prediction
