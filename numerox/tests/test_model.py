import numerox as nx
from numerox.model import HAS_XGBOOST


def get_models():
    models = [nx.logistic(), nx.extratrees(), nx.randomforest()]
    if HAS_XGBOOST:
        models.append(nx.xgboost())
    return models


def test_model_repr():
    "Make sure Model.__repr__ runs"
    for model in get_models():
        model.__repr__()


def test_model_hash():
    "Make sure Model.hash runs"
    d = nx.testing.micro_data()
    dt = d['train']
    dp = d['tournament']
    for model in get_models():
        model.hash(dt, dp)


def test_model_run():
    "Make sure models run"
    d = nx.play_data()
    d_fit = d['train']
    d_predict = d['tournament']
    for model in get_models():
        model.fit_predict(d_fit, d_predict)


def test_pipeline():
    d = nx.play_data()
    d_fit = d['train']
    d_predict = d['tournament']
    model = nx.pipeline([
        nx.logistic()
    ])
    model.fit_predict(d_fit, d_predict)
