import pprint

from numerox import Prediction, tournament_splitter, cv_splitter


def run(model, splitter, verbosity=2):
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
    return prediction


def production(model, data, verbosity=2):
    splitter = tournament_splitter(data)
    prediction = run(model, splitter, verbosity=verbosity)
    return prediction


def backtest(model, data, kfold=5, seed=0, verbosity=2):
    splitter = cv_splitter(data, kfold=kfold, seed=seed)
    prediction = run(model, splitter, verbosity)
    return prediction
