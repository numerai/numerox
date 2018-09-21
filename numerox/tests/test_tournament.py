from nose.tools import ok_
from nose.tools import assert_raises

import numerox as nx
from numerox.tournament import tournament_int2str, tournament_str2int


def test_tournament():
    "Roundtrip of tournament_int2str and tournament_str2int"
    for i in range(1, 6):
        t = tournament_str2int(tournament_int2str(i))
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
    assert_raises(ValueError, nx.tournament_str, 6)
    assert_raises(ValueError, nx.tournament_str, 'burn')
    assert_raises(ValueError, nx.tournament_str, None)


def test_tournament_all():
    "test tournament_all"
    t = ['bernie', 'elizabeth', 'jordan', 'ken', 'charles']
    ok_(nx.tournament_all() == t, 'wrong tournaments')
    ok_(nx.tournament_all(True) == t, 'wrong tournaments')
    t = [1, 2, 3, 4, 5]
    ok_(nx.tournament_all(False) == t, 'wrong tournaments')
