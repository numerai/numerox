import os

from nose.tools import ok_
from nose.tools import assert_raises
from nose.tools import raises

import numpy as np
import pandas as pd

import numerox as nx
from numerox.testing import assert_data_equal as ade

TINY_DATASET_CSV = os.path.join(os.path.dirname(__file__),
                                'tiny_dataset_csv.zip')


def test_empty_prediction():
    "Test handling of empty predictions"
    p = nx.Prediction()
    ok_(p.names() == [], "wrong name")
    assert_raises(ValueError, p.rename, 'name')
    assert_raises(ValueError, p.rename, ['name'])
    assert_raises(ValueError, p.save, 'not_used')
    assert_raises(ValueError, p.drop_name, 'knn')
    assert_raises(ValueError, p.drop_tournament, 'bernie')
    assert_raises(ValueError, p.drop_pair, ('knn', 1))
    ok_((p.ids == np.array([], dtype=str)).all(), 'empty ids')
    ok_(p.copy() == p, 'empty copy')
    ok_(p.size == 0, 'empty size')
    ok_(p.shape == (0, 0), 'empty shape')
    ok_(len(p) == 0, 'empty length')
    p.__repr__()


@raises(ValueError)
def test_emtpy_y_raises():
    p = nx.Prediction()
    p.y


def test_prediction_methods():
    "test prediction methods"
    p = nx.testing.micro_prediction()
    ok_(len(p) == 10, "wrong length")
    ok_(p.size == 40, "wrong size")
    ok_(p.shape == (10, 4), "wrong shape")
    ok_(p == p, "not equal")
    ok_(p.name_isin('model1'), 'wrong output')
    ok_(not p.name_isin('modelX'), 'wrong output')
    ok_(p.tournament_isin('bernie'), 'wrong output')
    ok_(not p.tournament_isin('ken'), 'wrong output')
    ok_(p.pair_isin(('model1', 'bernie')), 'wrong output')
    ok_(('model1', 'bernie') in p, 'wrong output')
    ok_(('model1', 1) in p, 'wrong output')
    ok_(('model0', 'bernie') not in p, 'wrong output')


def test_prediction_roundtrip():
    "save/load roundtrip shouldn't change prediction"
    p = nx.testing.micro_prediction()
    path = None
    try:
        path = nx.testing.create_tempfile('numerox.h5')

        p.save(path)
        p2 = nx.load_prediction(path)
        ade(p, p2, "prediction corrupted during roundtrip")

        p.save(path, compress=False)
        p2 = nx.load_prediction(path)
    finally:
        nx.testing.delete_tempfile(path)
    ade(p, p2, "prediction corrupted during roundtrip")


def test_prediction_save():
    "test prediction.save with mode='a'"
    p = nx.testing.micro_prediction()
    p1 = p[('model0', 2)]
    p2 = p[[('model1', 1), ('model2', 3), ('model0', 5)]]
    path = None
    try:
        path = nx.testing.create_tempfile('numerox.h5')
        p1.save(path)
        p2.save(path, mode='a')
        p12 = nx.load_prediction(path)
    finally:
        nx.testing.delete_tempfile(path)
    ade(p, p12, "prediction corrupted during roundtrip")


def test_prediction_to_csv():
    "make sure prediction.to_csv runs"
    p = nx.testing.micro_prediction()
    path = None
    try:
        path = nx.testing.create_tempfile('numerox.h5')
        p[('model1', 1)].to_csv(path)
        with nx.testing.HiddenPrints():
            p[('model1', 1)].to_csv(path, verbose=True)
        p2 = nx.load_prediction_csv(path, 'model1')
        nx.load_prediction_csv(path)
    finally:
        nx.testing.delete_tempfile(path)
    ade(p2, p[('model1', 1)], "prediction corrupted during roundtrip")
    assert_raises(ValueError, p.to_csv, 'unused', 2)
    assert_raises(ValueError, p.to_csv, 'model1', 99)


def test_load_example_predictions():
    "test nx.load_example_predictions"
    p = nx.load_example_predictions(TINY_DATASET_CSV, tournament='elizabeth')
    p = nx.load_example_predictions(TINY_DATASET_CSV, tournament=1)
    ok_(len(p) == 6, "wrong number of rows")
    ok_(p.shape == (6, 1), 'data has wrong shape')
    ok_(np.abs(p.df.iloc[2, 0] - 0.50397) < 1e-8, 'wrong feature value')


def test_prediction_pairs_with_name():
    "test p.pairs_with_name"
    p = nx.testing.micro_prediction()
    pairs = p.pairs_with_name('model0')
    pairs0 = [('model0', 'elizabeth'), ('model0', 'charles')]
    ok_(pairs == pairs0, 'wrong pairs')
    pairs = p.pairs_with_name('model0', as_str=False)
    pairs0 = [('model0', 2), ('model0', 5)]
    ok_(pairs == pairs0, 'wrong pairs')
    pairs = p.pairs_with_name(['model0', 'model2'])
    pairs0 = [('model0', 'elizabeth'), ('model2', 'jordan'),
              ('model0', 'charles')]
    ok_(pairs == pairs0, 'wrong pairs')
    assert_raises(ValueError, p.pairs_with_name, None)


def test_prediction_pairs_with_tournament():
    "test p.pairs_with_tournament"
    p = nx.testing.micro_prediction()
    pairs = p.pairs_with_tournament('bernie')
    pairs0 = [('model1', 'bernie')]
    ok_(pairs == pairs0, 'wrong pairs')
    pairs = p.pairs_with_tournament('bernie', as_str=False)
    pairs0 = [('model1', 1)]
    ok_(pairs == pairs0, 'wrong pairs')
    pairs = p.pairs_with_tournament([2, 'charles'])
    pairs0 = [('model0', 'elizabeth'), ('model0', 'charles')]
    ok_(pairs == pairs0, 'wrong pairs')
    assert_raises(ValueError, p.pairs_with_tournament, None)


def test_iprediction_pairs_split():
    "test prediction.pairs_split"
    p = nx.testing.micro_prediction()
    name0 = ('model0', 'model1', 'model2', 'model0')
    tournament0 = ('elizabeth', 'bernie', 'jordan', 'charles')
    name, tournament = p.pairs_split()
    ok_(name == name0, 'wrong names')
    ok_(tournament == tournament0, 'wrong tournaments')


def test_prediction_make_pair():
    "test prediction.make_pair"
    p = nx.testing.micro_prediction()
    pair = p.make_pair('knn', 'bernie')
    ok_(pair == ('knn', 1))
    pair = p.make_pair('knn', 1)
    ok_(pair == ('knn', 1))
    assert_raises(ValueError, p.make_pair, None, 1)


def test_prediction_copies():
    "prediction properties should be copies"
    p = nx.testing.micro_prediction()
    ok_(nx.testing.shares_memory(p, p), "looks like shares_memory failed")
    ok_(nx.testing.shares_memory(p, p.ids), "p.ids should be a view")
    ok_(nx.testing.shares_memory(p, p.y), "p.y should be a view")
    ok_(not nx.testing.shares_memory(p, p.copy()), "should be a copy")
    ok_(not nx.testing.shares_memory(p, p.y_df), "should be a copy")


def test_prediction_properties():
    "prediction properties should not be corrupted"

    d = nx.testing.micro_data()
    p = nx.Prediction()
    p = p.merge_arrays(d.ids, d.y['bernie'], 'model1', 1)
    p = p.merge_arrays(d.ids, d.y['elizabeth'], 'model2', 2)

    ok_((p.ids == p.df.index).all(), "ids is corrupted")
    ok_((p.ids == d.df.index).all(), "ids is corrupted")
    ok_((p.y[:, 0] == d.df.bernie).all(), "y is corrupted")
    ok_((p.y[:, 1] == d.df.elizabeth).all(), "y is corrupted")


def test_prediction_rename():
    "prediction.rename"

    p = nx.testing.micro_prediction()
    rename_dict = {}
    names = []
    original_names = p.names()
    for i in range(len(original_names)):
        key = original_names[i]
        value = 'm_%d' % i
        names.append(value)
        rename_dict[key] = value
    p2 = p.rename(rename_dict)
    ok_(p2.names() == names, 'prediction.rename failed')

    p = nx.testing.micro_prediction()
    assert_raises(ValueError, p.rename, 'modelX')

    p = p[('model1', 1)]
    p2 = p.rename('modelX')
    ok_(p2.names()[0] == 'modelX', 'prediction.rename failed')


def test_prediction_drop_name():
    "prediction.drop_name"

    p = nx.testing.micro_prediction()
    p = p.drop_name('model1')
    prs = [('model0', 'elizabeth'), ('model2', 'jordan'),
           ('model0', 'charles')]
    ok_(p.pairs() == prs, 'p.drop_name failed')

    p = nx.testing.micro_prediction()
    p = p.drop_name(['model1', 'model0'])
    prs = [('model2', 'jordan')]
    ok_(p.pairs() == prs, 'p.drop_name failed')


def test_prediction_drop_tournament():
    "prediction.drop_tournament"

    p = nx.testing.micro_prediction()
    p = p.drop_tournament('bernie')
    prs = [('model0', 'elizabeth'), ('model2', 'jordan'),
           ('model0', 'charles')]
    ok_(p.pairs() == prs, 'p.drop_tournament failed')

    p = nx.testing.micro_prediction()
    p = p.drop_tournament(['bernie', 2, 'jordan'])
    prs = [('model0', 'charles')]
    ok_(p.pairs() == prs, 'p.drop_tournament failed')


def test_prediction_drop_pair():
    "prediction.drop_pair"

    p = nx.testing.micro_prediction()
    p = p.drop_pair(('model1', 1))
    prs = [('model0', 2), ('model2', 3), ('model0', 5)]
    ok_(p.pairs(as_str=False) == prs, 'p.drop_pair failed')

    p = nx.testing.micro_prediction()
    p = p.drop_pair([('model1', 1), ('model0', 2)])
    prs = [('model2', 3), ('model0', 5)]
    ok_(p.pairs(as_str=False) == prs, 'p.drop_pair failed')

    assert_raises(ValueError, p.drop_pair, ('modelX', 1))


def test_prediction_add():
    "add two predictions together"

    d = nx.testing.micro_data()
    p1 = nx.Prediction()
    p2 = nx.Prediction()
    d1 = d['train']
    d2 = d['tournament']
    rs = np.random.RandomState(0)
    y1 = 0.2 * (rs.rand(len(d1)) - 0.5) + 0.5
    y2 = 0.2 * (rs.rand(len(d2)) - 0.5) + 0.5
    p1 = p1.merge_arrays(d1.ids, y1, 'model1', 1)
    p2 = p2.merge_arrays(d2.ids, y2, 'model1', 1)

    p = p1 + p2  # just make sure that it runs

    assert_raises(ValueError, p.__add__, p1)
    assert_raises(ValueError, p1.__add__, p1)


def test_prediction_getitem():
    "prediction.__getitem__"
    p = nx.testing.micro_prediction()
    pairs = [('model2', 3), ('model0', 2)]
    p2 = p[pairs]
    ok_(isinstance(p2, nx.Prediction), 'expecting a prediction')
    ok_(p2.pairs(as_str=False) == pairs, 'pairs corrupted')
    pairs = [('model0', 2), ('model0', 5)]
    p2 = p['model0']
    ok_(p2.pairs(as_str=False) == pairs, 'pairs corrupted')
    p2 = p['model0', :]
    ok_(p2.pairs(as_str=False) == pairs, 'pairs corrupted')
    pairs = [('model1', 1)]
    p2 = p[:, 'bernie']
    ok_(p2.pairs(as_str=False) == pairs, 'pairs corrupted')
    assert_raises(IndexError, p.__getitem__, ('modelX', 1, 'extra'))
    assert_raises(IndexError, p.__getitem__, (1, 1))
    assert_raises(IndexError, p.__getitem__, (slice(1), 1))
    assert_raises(IndexError, p.__getitem__, (1, slice(1)))
    assert_raises(IndexError, p.__getitem__, [('model1', 1, 1)])


def test_prediction_loc():
    "test prediction.loc"
    mp = nx.testing.micro_prediction
    p = mp()
    msg = 'prediction.loc indexing error'
    ade(p.loc[['index1']], mp([1]), msg)
    ade(p.loc[['index4']], mp([4]), msg)
    ade(p.loc[['index4', 'index0']], mp([4, 0]), msg)
    ade(p.loc[['index4', 'index0', 'index2']], mp([4, 0, 2]), msg)


def test_prediction_y_correlation():
    "test prediction.y_correlation"
    p = nx.testing.micro_prediction()
    df = p.y_correlation()
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_prediction_summary():
    "make sure prediction.summary runs"
    d = nx.testing.micro_data()
    p = nx.testing.micro_prediction()
    df = p[('model1', 1)].summary(d, 3)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    assert_raises(ValueError, p.summary, d)
    with nx.testing.HiddenPrints():
        df_dict = p.summaries(d)
    ok_(isinstance(df_dict, dict), 'expecting a dict')


def test_prediction_metric_per_era():
    "make sure prediction.metric_per_era runs"
    d = nx.testing.micro_data()
    p = nx.testing.micro_prediction()
    df = p.metric_per_era(d, metric='logloss')
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    ok_(df.shape[0] == len(d.unique_era()), 'wrong shape')
    ok_(df.shape[1] == len(p.pairs()), 'wrong shape')
    df = p.metric_per_era(d, metric='logloss', split_pairs=False)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_prediction_metrics_per_tournament():
    "make sure prediction.metric_per_era runs"
    d = nx.testing.micro_data()
    p = nx.testing.micro_prediction()
    df = p.metric_per_tournament(d)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    ok_(df.shape[1] == 6, 'expecting 6 columns')


def test_prediction_performance():
    "make sure prediction.performance runs"
    d = nx.testing.micro_data()
    p = nx.testing.micro_prediction()
    df = p.performance(d, 1)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    p.performance(d, 2, sort_by='auc')
    p.performance(d, 'elizabeth', sort_by='auc')
    p.performance(d, 3, sort_by='acc')
    p.performance(d, 4, sort_by='ystd')
    p.performance(d, 5, sort_by='sharpe')
    p.performance(d, 1, sort_by='consis')


def test_prediction_performance_mean():
    "make sure prediction.performance_mean runs"
    d = nx.testing.micro_data()
    p = nx.testing.micro_prediction()
    df = p.performance_mean(d, mean_of='tournament')
    df = p.performance_mean(d, mean_of='name')
    df = p.performance_mean(d)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    p.performance_mean(d, sort_by='auc')
    p.performance_mean(d, sort_by='auc')
    p.performance_mean(d, sort_by='acc')
    p.performance_mean(d, sort_by='ystd')
    p.performance_mean(d, sort_by='sharpe')
    p.performance_mean(d, sort_by='consis')
    assert_raises(ValueError, p.performance_mean, d, 'x values')


def test_prediction_regression():
    "regression test of prediction performance evaluation"
    d = nx.play_data()
    p = nx.production(nx.logistic(), d, tournament=None, verbosity=0)
    for number, name in nx.tournament_iter():
        p2 = nx.production(nx.logistic(), d, tournament=name, verbosity=0)
        df = p.performance_mean(d['validation'], mean_of='tournament')
        logloss1 = df.loc[name]['logloss']
        logloss2 = p2.summary(d['validation']).loc['mean']['logloss']
        diff = np.abs(logloss1 - logloss2)
        msg = 'failed on {}'.format(name)
        ok_(diff < 1e-6, msg)


def test_prediction_dominance():
    "make sure prediction.dominance runs"

    d = nx.play_data()
    d = d['validation']

    p = nx.Prediction()
    p = p.merge_arrays(d.ids, d.y['bernie'], 'model1', 1)
    p = p.merge_arrays(d.ids, d.y['elizabeth'], 'model2', 2)
    p = p.merge_arrays(d.ids, d.y['jordan'], 'model3', 3)

    df = p.dominance(d, 3)
    df = p.dominance(d, 'jordan')

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    assert_raises(ValueError, p[('model1', 1)].dominance, d, 1)


def test_prediction_correlation():
    "make sure prediction.correlation runs"
    p = nx.testing.micro_prediction()
    with nx.testing.HiddenPrints():
        p.correlation()
        p.correlation(('model1', 'bernie'))
        p.correlation(('model1', 1))


def test_prediction_check():
    "make sure prediction.check runs"
    d = nx.play_data()
    p1 = nx.production(nx.logistic(), d, 'ken', verbosity=0)
    p2 = p1.copy()
    p2 = p2.rename('example_predictions')
    p = p1 + p2
    with nx.testing.HiddenPrints():
        df = p.check(d)
    ok_(isinstance(df, dict), 'expecting a dictionary')


def test_prediction_concordance():
    "make sure prediction.concordance runs"
    d = nx.testing.play_data()
    p = nx.production(nx.logistic(), d, 3, verbosity=0)
    df = p.concordance(d)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_prediction_compare():
    "make sure prediction.compare runs"
    d = nx.testing.micro_data()
    p = nx.testing.micro_prediction()
    df = p.compare(d, p, tournament=2)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_prediction_setitem():
    "compare prediction._setitem__ with merge"

    data = nx.play_data()
    p1 = nx.production(nx.logistic(), data, 'bernie', verbosity=0)
    p2 = nx.production(nx.logistic(1e-5), data, 2,  verbosity=0)
    p3 = nx.production(nx.logistic(1e-6), data, 3,  verbosity=0)
    p4 = nx.backtest(nx.logistic(), data, 4,  verbosity=0)

    p = nx.Prediction()
    p[('logistic', 1)] = p1
    p[('logistic', 2)] = p2
    p[('logistic', 3)] = p3
    p[('logistic', 4)] = p4

    pp = nx.Prediction()
    pp = pp.merge(p1)
    pp = pp.merge(p2)
    pp = pp.merge(p3)
    pp = pp.merge(p4)

    pd.testing.assert_frame_equal(p.df, pp.df)

    assert_raises(ValueError, p.__setitem__, ('logistic', 1), p1)
    assert_raises(ValueError, p.__setitem__, ('logistic', 1), p)


def test_prediction_ynew():
    "test prediction.ynew"
    p = nx.testing.micro_prediction()
    y = p.y.copy()
    y2 = np.random.rand(*y.shape)
    p2 = p.ynew(y2)
    np.testing.assert_array_equal(p2.y, y2, 'prediction.ynew failed')
    assert_raises(ValueError, p.ynew, y2[:3])
    assert_raises(ValueError, p.ynew, y2[:, :2])
    assert_raises(ValueError, p.ynew, y2.reshape(-1))
    p = nx.Prediction()
    assert_raises(ValueError, p.ynew, y2)


def test_prediction_y_df():
    "test prediction.y_df"
    p = nx.testing.micro_prediction()
    y = p.y.copy()
    df = p.y_df
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    np.testing.assert_array_equal(df.values, y, 'prediction.ynew failed')


def test_prediction_iter():
    "test prediction.iter"
    p = nx.testing.micro_prediction()
    pairs = []
    for pi in p.iter():
        n = pi.pairs()
        ok_(len(n) == 1, 'should only yield a single name')
        pairs.append(n[0])
    ok_(p.pairs() == pairs, 'prediction.iter failed')


def test_prediction_repr():
    "make sure prediction.__repr__() runs"
    p = nx.testing.micro_prediction()
    p.__repr__()


def test_data_hash():
    "test prediction.hash"
    p = nx.testing.micro_prediction()
    ok_(p.hash() == p.hash(), "prediction.hash not reproduceable")
    p2 = nx.Prediction(p.df[::2])
    ok_(p2.hash() == p2.hash(), "prediction.hash not reproduceable")


def test_merge_predictions():
    "test merge_predictions"

    p = nx.testing.micro_prediction()
    assert_raises(ValueError, nx.merge_predictions, [p, p])

    p2 = nx.merge_predictions([p, nx.Prediction()])
    ade(p2, p, 'corruption of merge predictions')

    p1 = nx.testing.micro_prediction([0, 1, 2, 3, 4])
    p2 = nx.testing.micro_prediction([5, 6, 7, 8, 9])
    p12 = nx.merge_predictions([p1, p2])
    ade(p12, p, 'corruption of merge predictions')

    p1 = nx.testing.micro_prediction([0, 1, 2, 3])
    p2 = nx.testing.micro_prediction([4, 5, 6])
    p3 = nx.testing.micro_prediction([7, 8, 9])
    p123 = nx.merge_predictions([p1, p2, p3])
    ade(p123, p, 'corruption of merge predictions')

    p1 = nx.testing.micro_prediction([9, 4, 3, 2])
    p2 = nx.testing.micro_prediction([1, 8, 7])
    p3 = nx.testing.micro_prediction([6, 5, 0])
    p123 = nx.merge_predictions([p1, p2, p3])
    ade(p123, p, 'corruption of merge predictions')

    p1 = nx.testing.micro_prediction([0, 1, 2, 3, 4])
    p11 = p1[[('model0', 2), ('model1', 1), ('model2', 3)]]
    p12 = p1['model0', 5]
    p2 = nx.testing.micro_prediction([5, 6, 7, 8, 9])
    p21 = p2['model0', 2]
    p22 = p2[[('model1', 1), ('model2', 3), ('model0', 5)]]
    p12 = nx.merge_predictions([p11, p21, p22, p12])
    ade(p12, p, 'corruption of merge predictions')
