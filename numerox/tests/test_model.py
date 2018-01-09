import numerox as nx


def get_models():
    models = [nx.logistic(), nx.extratrees(), nx.randomforest(),
              nx.mlpc(), nx.logisticPCA()]
    return models


def test_model_repr():
    "Make sure Model.__repr__ runs"
    for model in get_models():
        model.__repr__()


def test_model_run():
    "Make sure models run"
    d = nx.play_data()
    d_fit = d['train']
    d_predict = d['tournament']
    for model in get_models():
        model.fit_predict(d_fit, d_predict)
