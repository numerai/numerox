from numerox.model import logistic, extratrees
from numerox.testing import micro_data


def test_model_repr():
    "Make sure Model.__repr__ runs"
    models = [logistic(), extratrees()]
    for model in models:
        model.__repr__()


def test_model_run():
    "Make sure models run"
    d = micro_data()
    d_fit = d['train']
    d_predict = d['tournament']
    models = [logistic(), extratrees(nfeatures=2)]
    for model in models:
        model.fit_predict(d_fit, d_predict)
