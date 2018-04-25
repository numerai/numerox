from nose.tools import ok_

import numerox as nx


def make_status():
    s = {}
    s['concordance'] = True
    s['consistency'] = 58
    s['originality'] = True
    s['validation_logloss'] = 0.693
    return s


def test_is_controlling_capital():
    "test is_controlling_capital"

    iscc = nx.is_controlling_capital
    msg = 'is_controlling_capital failed'

    s = make_status()
    ok_(iscc(s), msg)

    s = make_status()
    s['concordance'] = None
    ok_(not iscc(s), msg)

    s = make_status()
    s['concordance'] = False
    ok_(not iscc(s), msg)

    s = make_status()
    s['consistency'] = 57
    ok_(not iscc(s), msg)
