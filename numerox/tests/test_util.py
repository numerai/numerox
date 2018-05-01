from nose.tools import ok_

import pandas as pd

import numerox as nx


def test_isint():
    "test isint"
    ok_(nx.isint(1))
    ok_(nx.isint(-1))
    ok_(not nx.isint(1.1))
    ok_(not nx.isint('a'))
    ok_(not nx.isint(True))
    ok_(not nx.isint(False))
    ok_(not nx.isint(None))


def test_isstring():
    "test isstring"
    ok_(nx.isstring('1'))
    ok_(nx.isstring("1"))
    ok_(nx.isstring(u'1'))
    ok_(not nx.isstring(1))
    ok_(not nx.isstring(1))
    ok_(not nx.isstring(1.1))
    ok_(not nx.isstring(True))
    ok_(not nx.isstring(False))
    ok_(not nx.isstring(None))


def test_history():
    "make sure history runs"
    df = nx.history()
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
