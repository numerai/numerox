import time
import pprint

import numerox as nx


def production(model, data, tournament=None, verbosity=2):
    """
    Fit a model with train data; make prediction on tournament data.

    Parameters
    ----------
    model : nx.Model, list, tuple
        Prediction model. Can be a list or tuple of prediction models. Model
        names must be unique.
    data : nx.Data
        The data to run the model through.
    tournament : {None, int, str, list, tuple}, optional
        The tournament(s) to run the model through. By default (None) the
        model is run through all five tournaments. If a list or tuple of
        tournaments is given then it must must not contain duplicate
        tournaments.
    verbosity : int, optional
        An integer that determines verbosity. Zero is silent.

    Returns
    -------
    p : nx.Prediction
        A prediction object containing the predictions of the specified
        model/tournament pairs.

    """
    splitter = nx.TournamentSplitter(data)
    prediction = run(model, splitter, tournament, verbosity=verbosity)
    return prediction


def backtest(model, data, tournament=None, kfold=5, seed=0, verbosity=2):
    """
    K-fold cross validation of model over the train data.

    Parameters
    ----------
    model : nx.Model, list, tuple
        Prediction model. Can be a list or tuple of prediction models. Model
        names must be unique.
    data : nx.Data
        The data to run the model through.
    tournament : {None, int, str, list, tuple}, optional
        The tournament(s) to run the model through. By default (None) the
        model is run through all five tournaments. If a list or tuple of
        tournaments is given then it must must not contain duplicate
        tournaments.
    verbosity : int, optional
        An integer that determines verbosity. Zero is silent.

    Returns
    -------
    p : nx.Prediction
        A prediction object containing the predictions of the specified
        model/tournament pairs.

    """
    splitter = nx.CVSplitter(data, kfold=kfold, seed=seed, train_only=True)
    prediction = run(model, splitter, tournament, verbosity)
    return prediction


def run(model, splitter, tournament=None, verbosity=2):
    """
    Run a model/tournament pair (or pairs) through a data splitter.

    Parameters
    ----------
    model : nx.Model, list, tuple
        Prediction model to run through the splitter. Can be a list or tuple
        of prediction models. Model names must be unique.
    splitter : nx.Splitter
        An iterator of fit/predict data pairs.
    tournament : {None, int, str, list, tuple}, optional
        The tournament(s) to run the model through. By default (None) the
        model is run through all five tournaments. If a list or tuple of
        tournaments is given then it must must not contain duplicate
        tournaments.
    verbosity : int, optional
        An integer that determines verbosity. Zero is silent.

    Returns
    -------
    p : nx.Prediction
        A prediction object containing the predictions of the specified
        model/tournament pairs.

    """

    # make list of models
    if isinstance(model, nx.Model):
        models = [model]
    elif isinstance(model, list) or isinstance(model, tuple):
        models = model
    else:
        raise ValueError('`model` must be a model, list, or tuple of models')
    names = [m.name for m in models]
    if len(names) != len(set(names)):
        raise ValueError('`model` cannot contain duplicate names')

    # make list of tournaments
    if tournament is None:
        tournaments = nx.tournament_all()
    elif nx.isint(tournament) or nx.isstring(tournament):
        tournaments = [tournament]
    elif isinstance(tournament, list) or isinstance(tournament, tuple):
        tournaments = tournament
    else:
        msg = '`tournament` must be an integer, string, list, tuple, or None.'
        raise ValueError(msg)
    tournaments = [nx.tournament_str(t) for t in tournaments]
    if len(tournaments) != len(set(tournaments)):
        raise ValueError('`tournament` cannot contain duplicates')

    # loop over all model/tournament pairs
    p = nx.Prediction()
    for m in models:
        for t in tournaments:
            splitter.reset()
            p += run_one(m, splitter, t, verbosity=verbosity)
    splitter.reset()

    return p


def run_one(model, splitter, tournament, verbosity=2):
    "Run a single model through a data splitter for a single tournament"
    t0 = time.time()
    name = model.name
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
        prediction = prediction.merge_arrays(ids, yhat, name, tournament)
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
