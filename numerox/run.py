import time
import pprint

import numerox as nx


def production(model, data, tournament, name=None, verbosity=2):
    "Fit a model with train data; make prediction on tournament data"
    splitter = nx.TournamentSplitter(data)
    prediction = run(model, splitter, tournament, name, verbosity=verbosity)
    return prediction


def backtest(model, data, tournament, name=None, kfold=5, seed=0, verbosity=2):
    "K-fold cross validation of model through train data"
    splitter = nx.CVSplitter(data, kfold=kfold, seed=seed, train_only=True)
    prediction = run(model, splitter, tournament, name, verbosity)
    return prediction


def run(model, splitter, tournament, name=None, verbosity=2):
    "Run a single model through a data splitter"
    t0 = time.time()
    if name is None:
        if nx.isint(tournament):
            postfix = '_t' + str(tournament)
        else:
            postfix = '_' + tournament
        name = model.__class__.__name__ + postfix
    else:
        if verbosity > 2:
            print(name)
    if verbosity > 2:
        print(splitter)
    if verbosity > 0:
        pprint.pprint(model)
    data = None
    prediction = nx.Prediction()
    for data_fit, data_predict in splitter:
        if verbosity > 0:
            if data is None:
                data = data_predict.copy()
            else:
                data = data + data_predict
        # the following line of code hides from your model the y
        # that you are trying to predict to prevent accidental cheating
        data_predict = data_predict.y_to_nan()
        ids, yhat = model.fit_predict(data_fit, data_predict, tournament)
        prediction = prediction.merge_arrays(ids, yhat, name)
        if verbosity > 1:
            print(prediction.summary(data.region_isnotin(['test', 'live']),
                                     tournament))
    if verbosity == 1:
        print(prediction.summary(data.region_isnotin(['test', 'live']),
                                 tournament))
    if verbosity > 1:
        minutes = (time.time() - t0) / 60
        print('Done in {:.2f} minutes'.format(minutes))
    return prediction
