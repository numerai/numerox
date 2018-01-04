import tempfile

import numpy as np
import pandas as pd
from nose.tools import ok_
from nose.tools import assert_raises

import numerox as nx
from numerox import testing
from numerox.testing import assert_data_equal as ade


def test_prediction_roundtrip():
    "save/load roundtrip shouldn't change prediction"
    p = testing.micro_prediction()
    with tempfile.NamedTemporaryFile() as temp:
        p.save(temp.name)
        p2 = nx.load_prediction(temp.name)
        ade(p, p2, "prediction corrupted during roundtrip")


def test_prediction_copies():
    "prediction properties should be copies"
    p = testing.micro_prediction()
    ok_(testing.shares_memory(p, p), "looks like shares_memory failed")
    ok_(testing.shares_memory(p, p.ids), "p.ids should be a view")
    ok_(testing.shares_memory(p, p.yhat), "p.yhat should be a view")
    ok_(not testing.shares_memory(p, p.copy()), "should be a copy")


def test_data_properties():
    "prediction properties should not be corrupted"

    d = testing.micro_data()
    p = nx.Prediction()
    p.merge_arrays(d.ids, d.y, 'model1')
    p.merge_arrays(d.ids, d.y, 'model2')

    ok_((p.ids == p.df.index).all(), "ids is corrupted")
    ok_((p.ids == d.df.index).all(), "ids is corrupted")
    ok_((p.yhat[:, 0] == d.df.y).all(), "yhat is corrupted")
    ok_((p.yhat[:, 1] == d.df.y).all(), "yhat is corrupted")


def test_prediction_add():
    "add two predictions together"

    d = testing.micro_data()
    p1 = nx.Prediction()
    p2 = nx.Prediction()
    d1 = d['train']
    d2 = d['tournament']
    rs = np.random.RandomState(0)
    yhat1 = 0.2 * (rs.rand(len(d1)) - 0.5) + 0.5
    yhat2 = 0.2 * (rs.rand(len(d2)) - 0.5) + 0.5
    p1.merge_arrays(d1.ids, yhat1, 'model1')
    p2.merge_arrays(d2.ids, yhat2, 'model1')

    p = p1 + p2  # just make sure that it runs

    assert_raises(ValueError, p.__add__, p1)
    assert_raises(ValueError, p1.__add__, p1)


def test_prediction_getitem():
    "prediction.__getitem__"
    p = testing.micro_prediction()
    names = ['model2', 'model0']
    p2 = p[names]
    ok_(isinstance(p2, nx.Prediction), 'expecting a prediction')
    ok_(p2.names == names, 'names corrcupted')


def test_prediction_performance_df():
    "make sure prediction.performance_df runs"

    d = testing.micro_data()
    d = d['train'] + d['validation']

    p = nx.Prediction()
    p.merge_arrays(d.ids, d.y, 'model1')
    p.merge_arrays(d.ids, d.y, 'model2')
    p.merge_arrays(d.ids, d.y, 'model3')

    df, info = p.performance_df(d)

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    ok_(isinstance(info, dict), 'expecting a dictionary')


def test_prediction_dominance_df():
    "make sure prediction.dominance_df runs"

    d = nx.play_data()
    d = d['validation']

    p = nx.Prediction()
    p.merge_arrays(d.ids, d.y, 'model1')
    p.merge_arrays(d.ids, d.y, 'model2')
    p.merge_arrays(d.ids, d.y, 'model3')

    df = p.dominance_df(d)

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_prediction_repr():
    "make sure prediction.__repr__() runs"
    p = testing.micro_prediction()
    p.__repr__()
