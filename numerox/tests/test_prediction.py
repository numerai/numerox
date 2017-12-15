import tempfile

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import ok_, assert_raises

import numerox as nx
from numerox import Prediction
from numerox.testing import assert_data_equal as ade
from numerox.testing import (play_data, shares_memory, micro_prediction,
                             micro_data)


def test_prediction_roundtrip():
    "save/load roundtrip shouldn't change prediction"
    d = micro_data()
    m = nx.logistic()
    p = nx.production(m, d, verbosity=0)
    with tempfile.NamedTemporaryFile() as temp:
        p.save(temp.name)
        p2 = nx.load_prediction(temp.name)
        ade(p, p2, "prediction corrupted during roundtrip")


def test_prediction_consistency():
    "make sure prediction.consistency runs"

    d = play_data()
    p = Prediction()

    rs = np.random.RandomState(0)
    yhat = 0.005 * rs.randn(len(d)) + 0.5
    yhat += 0.0005 * (d.y - 0.5)
    p.append(d.ids, yhat)

    c = p.consistency(d['train'])
    ok_(c > 0, "consistency should (likely) be greater than zero")
    ok_(c < 1, "consistency should (likely) be less than one")


def test_prediction_yhatnew():
    "changing preidction.yhat"

    d = play_data()
    p = Prediction()
    p.append(d.ids, d.y)

    y = 2 * p.yhat
    pnew = p.yhatnew(y)
    ok_(not shares_memory(p, pnew), "prediction.yhatnew should return a copy")
    assert_array_equal(pnew.yhat, y, "prediction.yhatneww corrupted values")
    assert_raises(ValueError, p.yhatnew, y[:4])


def test_prediction_copies():
    "prediction properties should be copies"

    d = play_data()
    p = Prediction()
    p.append(d.ids, d.y)

    ok_(shares_memory(p, p), "looks like shares_memory failed")
    ok_(shares_memory(p, p.ids), "p.ids should be a view")
    ok_(shares_memory(p, p.yhat), "p.yhat should be a view")
    ok_(not shares_memory(p, p.copy()), "should be a copy")


def test_data_properties():
    "prediction properties should not be corrupted"

    d = play_data()
    p = Prediction()
    p.append(d.ids, d.y)

    ok_((p.ids == p.df.index).all(), "ids is corrupted")
    ok_((p.ids == d.df.index).all(), "ids is corrupted")
    idx = ~np.isnan(p.df.yhat)
    ok_((p.yhat[idx] == p.df.yhat[idx]).all(), "yhat is corrupted")
    ok_((p.yhat[idx] == d.df.y[idx]).all(), "yhat is corrupted")


def test_prediction_add():
    "add two predictions together"

    d = micro_data()
    p1 = Prediction()
    p2 = Prediction()
    d1 = d['train']
    d2 = d['tournament']
    rs = np.random.RandomState(0)
    yhat1 = 0.2 * (rs.rand(len(d1)) - 0.5) + 0.5
    yhat2 = 0.2 * (rs.rand(len(d2)) - 0.5) + 0.5
    p1.append(d1.ids, yhat1)
    p2.append(d2.ids, yhat2)

    p = p1 + p2  # just make sure that it runs

    assert_raises(IndexError, p.__add__, p1)
    assert_raises(IndexError, p1.__add__, p1)


def test_prediction_repr():
    "make sure prediction.__repr__() runs"
    p = micro_prediction()
    p.__repr__()
