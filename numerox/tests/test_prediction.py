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
    ok_(testing.shares_memory(p, p.y), "p.y should be a view")
    ok_(not testing.shares_memory(p, p.copy()), "should be a copy")


def test_data_properties():
    "prediction properties should not be corrupted"

    d = testing.micro_data()
    p = nx.Prediction()
    p = p.merge_arrays(d.ids, d.y, 'model1')
    p = p.merge_arrays(d.ids, d.y, 'model2')

    ok_((p.ids == p.df.index).all(), "ids is corrupted")
    ok_((p.ids == d.df.index).all(), "ids is corrupted")
    ok_((p.y[:, 0] == d.df.y).all(), "y is corrupted")
    ok_((p.y[:, 1] == d.df.y).all(), "y is corrupted")


def test_prediction_rename():
    "prediction.rename"

    p = testing.micro_prediction()
    rename_dict = {}
    names = []
    original_names = p.names
    for i in range(p.shape[1]):
        key = original_names[i]
        value = 'm_%d' % i
        names.append(value)
        rename_dict[key] = value
    p2 = p.rename(rename_dict)
    ok_(p2.names == names, 'prediction.rename failed')

    p = testing.micro_prediction()
    p = p['model1']
    p2 = p.rename('modelX')
    ok_(p2.names[0] == 'modelX', 'prediction.rename failed')


def test_prediction_drop():
    "prediction.drop"
    p = testing.micro_prediction()
    p = p.drop(['model1'])
    ok_(p.names == ['model0', 'model2'], 'prediction.drop failed')


def test_prediction_add():
    "add two predictions together"

    d = testing.micro_data()
    p1 = nx.Prediction()
    p2 = nx.Prediction()
    d1 = d['train']
    d2 = d['tournament']
    rs = np.random.RandomState(0)
    y1 = 0.2 * (rs.rand(len(d1)) - 0.5) + 0.5
    y2 = 0.2 * (rs.rand(len(d2)) - 0.5) + 0.5
    p1 = p1.merge_arrays(d1.ids, y1, 'model1')
    p2 = p2.merge_arrays(d2.ids, y2, 'model1')

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


def test_prediction_loc():
    "test prediction.loc"
    mp = testing.micro_prediction
    p = mp()
    msg = 'prediction.loc indexing error'
    ade(p.loc[['index1']], mp([1]), msg)
    ade(p.loc[['index4']], mp([4]), msg)
    ade(p.loc[['index4', 'index0']], mp([4, 0]), msg)
    ade(p.loc[['index4', 'index0', 'index2']], mp([4, 0, 2]), msg)


def test_prediction_performance_df():
    "make sure prediction.performance_df runs"

    d = testing.micro_data()
    d = d['train'] + d['validation']

    p = nx.Prediction()
    p = p.merge_arrays(d.ids, d.y, 'model1')
    p = p.merge_arrays(d.ids, d.y, 'model2')
    p = p.merge_arrays(d.ids, d.y, 'model3')

    df, info = p.performance_df(d)

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    ok_(isinstance(info, dict), 'expecting a dictionary')


def test_prediction_dominance_df():
    "make sure prediction.dominance_df runs"

    d = nx.play_data()
    d = d['validation']

    p = nx.Prediction()
    p = p.merge_arrays(d.ids, d.y, 'model1')
    p = p.merge_arrays(d.ids, d.y, 'model2')
    p = p.merge_arrays(d.ids, d.y, 'model3')

    df = p.dominance_df(d)

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_prediction_originality():
    "make sure prediction.originality runs"
    p = testing.micro_prediction()
    df = p.originality(['model1'])
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_prediction_setitem():
    "compare prediction._setitem__ with merge"

    data = nx.play_data()
    p1 = nx.production(nx.logistic(), data, 'model1', verbosity=0)
    p2 = nx.production(nx.logistic(1e-5), data, 'model2',  verbosity=0)
    p3 = nx.production(nx.logistic(1e-6), data, 'model3',  verbosity=0)
    p4 = nx.backtest(nx.logistic(), data, 'model1',  verbosity=0)

    p = nx.Prediction()
    p['model1'] = p1
    p['model2'] = p2
    p['model3'] = p3
    p['model1'] = p4

    pp = nx.Prediction()
    pp = pp.merge(p1)
    pp = pp.merge(p2)
    pp = pp.merge(p3)
    pp = pp.merge(p4)

    pd.testing.assert_frame_equal(p.df, pp.df)

    assert_raises(ValueError, p.__setitem__, 'model1', p1)


def test_prediction_repr():
    "make sure prediction.__repr__() runs"
    p = testing.micro_prediction()
    p.__repr__()
