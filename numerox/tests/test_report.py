from nose.tools import ok_
from nose.tools import assert_raises

import pandas as pd

import numerox as nx
from numerox.testing import micro_data


def test_report_performance_df():
    "make sure report.performance_df runs"

    d = micro_data()
    d = d['train'] + d['validation']

    p = nx.Prediction()
    p.append(d.ids, d.y)

    r = nx.Report()
    r.append_prediction(p, 'model1')
    r.append_prediction(p, 'model2')
    r.append_prediction(p, 'model3')

    df, info = r.performance_df(d)

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    ok_(isinstance(info, dict), 'expecting a dictionary')


def test_report_getitem():
    "report.__getitem__"

    d = micro_data()
    p = nx.Prediction()
    p.append(d.ids, d.y)

    r = nx.Report()
    r.append_prediction(p, 'model1')
    r.append_prediction(p, 'model2')
    r.append_prediction(p, 'model3')

    r2 = r[['model3', 'model1']]

    ok_(isinstance(r2, nx.Report), 'expecting a report')
    ok_(r2.models == ['model3', 'model1'], 'expecting a dictionary')


def test_report_dominance_df():
    "make sure report.dominance_df runs"

    d = nx.play_data()
    d = d['validation']

    p = nx.Prediction()
    p.append(d.ids, d.y)

    r = nx.Report()
    r.append_prediction(p, 'model1')
    r.append_prediction(p, 'model2')
    r.append_prediction(p, 'model3')

    df = r.dominance_df(d)

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')


def test_report_setitem():
    "report.__setitem__"

    data = nx.play_data()
    p1 = nx.production(nx.logistic(), data, verbosity=0)
    p2 = nx.production(nx.logistic(1e-5), data, verbosity=0)
    p3 = nx.production(nx.logistic(1e-6), data, verbosity=0)
    p4 = nx.backtest(nx.logistic(2e-4), data, verbosity=0)

    r = nx.Report()
    r['model1'] = p1
    r['model2'] = p2
    r['model3'] = p3
    r['model4'] = p4
    r['model1'] = p4

    r2 = nx.Report()
    r2.append_prediction(p1, 'model1')
    r2.append_prediction(p2, 'model2')
    r2.append_prediction(p3, 'model3')
    r2.append_prediction(p4, 'model4')
    r2.append_prediction(p4, 'model1')

    pd.testing.assert_frame_equal(r.df, r2.df)

    assert_raises(ValueError, r.__setitem__, 'model1', p1)


def test_report_originality():
    "make sure report.originality runs"

    d = nx.play_data()
    d = d['validation']

    p = nx.Prediction()
    p.append(d.ids, d.y)

    r = nx.Report()
    r.append_prediction(p, 'model1')
    r.append_prediction(p, 'model2')
    r.append_prediction(p, 'model3')

    df = r.originality(['model1'])

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
