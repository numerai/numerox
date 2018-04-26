from nose.tools import ok_

from numerox.util import isint, isstring


def test_isint():
    "test isint"
    ok_(isint(1))
    ok_(isint(-1))
    ok_(not isint(1.1))
    ok_(not isint('a'))
    ok_(not isint(True))
    ok_(not isint(False))
    ok_(not isint(None))


def test_isstring():
    "test isint"
    ok_(isstring('1'))
    ok_(isstring("1"))
    ok_(isstring(u'1'))
    ok_(not isstring(1))
    ok_(not isstring(1))
    ok_(not isstring(1.1))
    ok_(not isstring(True))
    ok_(not isstring(False))
    ok_(not isstring(None))
