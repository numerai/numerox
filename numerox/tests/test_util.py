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


def test_flatten_dict():
    "test flatten_dict"
    d = {'a': 1, 'z': {'b': 2, 'c': 3}}
    f = nx.util.flatten_dict(d)
    f0 = {'a': 1, 'b': 2, 'c': 3}
    ok_(isinstance(f, dict), 'expecting a dict')
    ok_(f == f0, 'wrong dict returned')


def test_is_none_splice():
    "test util.is_none_slice"
    ok_(nx.util.is_none_slice(slice(None)))
    ok_(not nx.util.is_none_slice(slice(1, None)))
    ok_(not nx.util.is_none_slice(slice(None, 1)))
    ok_(not nx.util.is_none_slice(slice(None, None, 1)))
