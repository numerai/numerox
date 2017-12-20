from nose.tools import ok_
import pandas as pd

import numerox as nx
from numerox import Report, Prediction
from numerox.testing import micro_data


def test_report_performance_df():
    "make sure report.performance_df runs"

    d = micro_data()
    d = d['train'] + d['validation']

    p = Prediction()
    p.append(d.ids, d.y)

    r = Report()
    r.append_prediction(p, 'model1')
    r.append_prediction(p, 'model2')
    r.append_prediction(p, 'model3')

    df, info = r.performance_df(d)

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    ok_(isinstance(info, dict), 'expecting a dictionary')


def test_report_dominance_df():
    "make sure report.dominance_df runs"

    d = nx.play_data()
    d = d['validation']

    p = Prediction()
    p.append(d.ids, d.y)

    r = Report()
    r.append_prediction(p, 'model1')
    r.append_prediction(p, 'model2')
    r.append_prediction(p, 'model3')

    df = r.dominance_df(d)

    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
