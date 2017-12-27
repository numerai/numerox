from nose.tools import ok_
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
