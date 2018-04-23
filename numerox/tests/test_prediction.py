import os
import tempfile

import numpy as np
import pandas as pd
from nose.tools import ok_
from nose.tools import assert_raises

import numerox as nx
from numerox import testing
from numerox.testing import assert_data_equal as ade

TINY_DATASET_CSV = os.path.join(os.path.dirname(__file__),
                                'tiny_dataset_csv.zip')


def test_empty_prediction():
    "Test handling of empty predictions"
    p = nx.Prediction()
    ok_(p.names == [], "wrong name")
    assert_raises(ValueError, p.rename, 'name')
    assert_raises(ValueError, p.rename, ['name'])
    assert_raises(ValueError, p.drop, 'name')
    assert_raises(ValueError, p.drop, ['name'])
    assert_raises(ValueError, p.save, 'not_used')
    ok_((p.ids == np.array([], dtype=str)).all(), 'empty ids')
    ok_(p.copy() == p, 'empty copy')
    ok_(p.size == 0, 'empty size')
    ok_(p.shape == (0, 0), 'empty shape')
    ok_(len(p) == 0, 'empty length')
    p.__repr__()


def test_prediction_methods():
    "test prediction methods"
    p = nx.testing.micro_prediction()
    ok_(len(p) == 10, "wrong length")
    ok_(p.size == 30, "wrong size")
    ok_(p.shape == (10, 3), "wrong shape")
    ok_(p == p, "not equal")


def test_prediction_roundtrip():
    "save/load roundtrip shouldn't change prediction"
    p = testing.micro_prediction()
    with tempfile.NamedTemporaryFile() as temp:

        p.save(temp.name)
        p2 = nx.load_prediction(temp.name)
        ade(p, p2, "prediction corrupted during roundtrip")

        p.save(temp.name, compress=False)
        p2 = nx.load_prediction(temp.name)
        ade(p, p2, "prediction corrupted during roundtrip")


def test_prediction_save():
    "test prediction.save with mode='a'"
    p = testing.micro_prediction()
    p1 = p['model0']
    p2 = p[['model1', 'model2']]
    with tempfile.NamedTemporaryFile() as temp:
        p1.save(temp.name)
        p2.save(temp.name, mode='a')
        p12 = nx.load_prediction(temp.name)
        ade(p, p12, "prediction corrupted during roundtrip")


def test_prediction_to_csv():
    "make sure prediction.to_csv runs"
    p = testing.micro_prediction()
    with tempfile.NamedTemporaryFile() as temp:
        p['model1'].to_csv(temp.name)
        with testing.HiddenPrints():
            p['model1'].to_csv(temp.name, verbose=True)
        p2 = nx.load_prediction_csv(temp.name, 'model1')
        ade(p2, p['model1'], "prediction corrupted during roundtrip")
    assert_raises(ValueError, p.to_csv, 'unused')


def test_load_example_predictions():
    "test nx.load_example_predictions"
    p = nx.load_example_predictions(TINY_DATASET_CSV)
    ok_(len(p) == 4, "wrong number of rows")
    ok_(p.shape == (4, 1), 'data has wrong shape')
    ok_(np.abs(p.df.iloc[2, 0] - 0.50123) < 1e-8, 'wrong feature value')


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
    assert_raises(ValueError, p.rename, 'modelX')

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


def test_prediction_summary():
    "make sure prediction.summary runs"
    d = testing.micro_data()
    p = testing.micro_prediction()
    df = p['model1'].summary(d)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_prediction_performance():
    "make sure prediction.performance runs"
    d = testing.micro_data()
    p = testing.micro_prediction()
    df = p.performance(d)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    p.performance(d, sort_by='auc')
    p.performance(d, sort_by='acc')
    p.performance(d, sort_by='ystd')
    p.performance(d, sort_by='sharpe')
    p.performance(d, sort_by='consis')


def test_prediction_dominance():
    "make sure prediction.dominance runs"

    d = nx.play_data()
    d = d['validation']

    p = nx.Prediction()
    p = p.merge_arrays(d.ids, d.y, 'model1')
    p = p.merge_arrays(d.ids, d.y, 'model2')
    p = p.merge_arrays(d.ids, d.y, 'model3')

    df = p.dominance(d)

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    assert_raises(ValueError, p['model1'].dominance, d)


def test_prediction_originality():
    "make sure prediction.originality runs"
    p = testing.micro_prediction()
    df = p.originality(['model1'])
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_prediction_correlation():
    "make sure prediction.correlation runs"
    p = testing.micro_prediction()
    with testing.HiddenPrints():
        p.correlation()


def test_prediction_check():
    "make sure prediction.check runs"
    d = nx.play_data()
    p1 = nx.production(nx.logistic(), d, verbosity=0)
    p2 = p1.copy()
    p2 = p2.rename('example_predictions')
    p = p1 + p2
    with testing.HiddenPrints():
        df = p.check(d)
    ok_(isinstance(df, dict), 'expecting a dictionary')
    assert_raises(ValueError, p1.check, d)


def test_prediction_concordance():
    "make sure prediction.concordance runs"
    d = testing.play_data()
    p = nx.production(nx.logistic(), d, 'model1', verbosity=0)
    df = p.concordance(d)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_prediction_compare():
    "make sure prediction.compare runs"
    d = testing.micro_data()
    p = testing.micro_prediction()
    df = p.compare(d, p)
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
    assert_raises(ValueError, p.__setitem__, 'model1', p)


def test_prediction_ynew():
    "test prediction.ynew"
    p = testing.micro_prediction()
    y = p.y.copy()
    y2 = np.random.rand(*y.shape)
    p2 = p.ynew(y2)
    np.testing.assert_array_equal(p2.y, y2, 'prediction.ynew failed')
    assert_raises(ValueError, p.ynew, y2[:3])
    assert_raises(ValueError, p.ynew, y2[:, :2])
    assert_raises(ValueError, p.ynew, y2.reshape(-1))


def test_prediction_iter():
    "test prediction.iter"
    p = testing.micro_prediction()
    names = []
    for pi in p.iter():
        n = pi.names
        ok_(len(n) == 1, 'should only yield a single name')
        names.append(n[0])
    ok_(p.names == names, 'prediction.iter failed')


def test_prediction_repr():
    "make sure prediction.__repr__() runs"
    p = testing.micro_prediction()
    p.__repr__()


def test_data_hash():
    "test prediction.hash"
    p = testing.micro_prediction()
    ok_(p.hash() == p.hash(), "prediction.hash not reproduceable")
    p2 = nx.Prediction(p.df[::2])
    ok_(p2.hash() == p2.hash(), "prediction.hash not reproduceable")


def test_merge_predictions():
    "test merge_predictions"

    p = testing.micro_prediction()
    assert_raises(ValueError, nx.merge_predictions, [p, p])

    p2 = nx.merge_predictions([p, nx.Prediction()])
    ade(p2, p, 'corruption of merge predictions')

    p1 = testing.micro_prediction([0, 1, 2, 3, 4])
    p2 = testing.micro_prediction([5, 6, 7, 8, 9])
    p12 = nx.merge_predictions([p1, p2])
    ade(p12, p, 'corruption of merge predictions')

    p1 = testing.micro_prediction([0, 1, 2, 3])
    p2 = testing.micro_prediction([4, 5, 6])
    p3 = testing.micro_prediction([7, 8, 9])
    p123 = nx.merge_predictions([p1, p2, p3])
    ade(p123, p, 'corruption of merge predictions')

    p1 = testing.micro_prediction([9, 4, 3, 2])
    p2 = testing.micro_prediction([1, 8, 7])
    p3 = testing.micro_prediction([6, 5, 0])
    p123 = nx.merge_predictions([p1, p2, p3])
    ade(p123, p, 'corruption of merge predictions')

    p1 = testing.micro_prediction([0, 1, 2, 3, 4])
    p11 = p1[['model0', 'model1']]
    p12 = p1['model2']
    p2 = testing.micro_prediction([5, 6, 7, 8, 9])
    p21 = p2['model0']
    p22 = p2[['model1', 'model2']]
    p12 = nx.merge_predictions([p11, p21, p22, p12])
    ade(p12, p, 'corruption of merge predictions')
