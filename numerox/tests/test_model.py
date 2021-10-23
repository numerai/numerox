from nose.tools import ok_
from nose.tools import assert_raises

import numerox as nx


def get_models():

    models = [nx.linear(),
              nx.ridge_mean(),
              nx.extratrees(),
              nx.randomforest(),
              nx.mlpc(),
              nx.linearPCA(),
              nx.example_predictions(),
              nx.fifty()]

    return models


def test_model_repr():
    """Make sure Model.__repr__ runs"""
    for model in get_models():
        model.__repr__()

    # model without self.p
    class test_model(nx.Model):
        def __init__(self):
            pass

    model = test_model()
    model.__repr__()

def test_model_run():
    """Make sure models run"""
    d = nx.play_data()
    dfit = d['train']
    dpre = d['tournament']
    for model in get_models():
        model.fit_predict(dfit, dpre, tournament=8)


def test_model_rename():
    """Test renaming a model"""
    model = nx.linear()
    ok_(model.name == 'linear', 'wrong name')
    model.rename('LR')
    ok_(model.name == 'LR', 'wrong name')
    model = model.rename('logreg')
    ok_(model.name == 'logreg', 'wrong name')
    ok_(model.__repr__().startswith('logreg'), 'wrong name')
    model = nx.linear()
    ok_(model.rename(None).name == 'linear', 'wrong name')
    assert_raises(ValueError, model.rename, 1)
