from nose.tools import ok_
from nose.tools import assert_raises

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


def test_tournament():
    "Roundtrip of tournament_int2str and tournament_str2int"
    for i in range(1, 6):
        t = nx.tournament_str2int(nx.tournament_int2str(i))
        ok_(t == i, 'tournament corrupted during round trip')


def test_tournament_int():
    "test tournament_int"
    for t_int, t_str in nx.tournament_iter():
        t_int2 = nx.tournament_int(t_int)
        ok_(t_int2 == t_int, "tournament int do not agree")
        t_int2 = nx.tournament_int(t_str)
        ok_(t_int2 == t_int, "tournament int do not agree")
    assert_raises(ValueError, nx.tournament_int, 0)
    assert_raises(ValueError, nx.tournament_int, 'burn')
    assert_raises(ValueError, nx.tournament_int, None)


def test_tournament_str():
    "test tournament_str"
    for t_int, t_str in nx.tournament_iter():
        t_str2 = nx.tournament_str(t_int)
        ok_(t_str2 == t_str, "tournament str do not agree")
        t_str2 = nx.tournament_str(t_str)
        ok_(t_str2 == t_str, "tournament str do not agree")
    assert_raises(ValueError, nx.tournament_str, 0)
    assert_raises(ValueError, nx.tournament_str, 'burn')
    assert_raises(ValueError, nx.tournament_int, None)


def test_flatten_dict():
    "test flatten_dict"
    d = {'a': 1, 'z': {'b': 2, 'c': 3}}
    f = nx.util.flatten_dict(d)
    f0 = {'a': 1, 'b': 2, 'c': 3}
    ok_(isinstance(f, dict), 'expecting a dict')
    ok_(f == f0, 'wrong dict returned')
