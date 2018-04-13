from nose.tools import ok_

import pandas as pd

import numerox as nx
from numerox.numerai import raw_leaderboard_to_df


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


def test_raw_earnings_to_df():
    "make sure raw_earnings_to_df runs"
    e = [{u'LiveLogloss': 0.6920578512962873,
          u'paymentGeneral': {u'nmrAmount': u'0.97', u'usdAmount': u'0.00'},
          u'paymentStaking': None,
          u'stake': {u'value': None},
          u'stakeResolution': None,
          u'username': u'cheat'},
         {u'LiveLogloss': 0.6920714939547946,
          u'paymentGeneral': {u'nmrAmount': u'0.93', u'usdAmount': u'0.00'},
          u'paymentStaking': None,
          u'stake': {u'value': None},
          u'stakeResolution': None,
          u'username': u'lie'},
         {u'LiveLogloss': 0.6920927181513603,
          u'paymentGeneral': {u'nmrAmount': u'0.90', u'usdAmount': u'0.00'},
          u'paymentStaking': None,
          u'stake': {u'value': None},
          u'stakeResolution': None,
          u'username': u'steal'}]
    df = raw_leaderboard_to_df(e, 88)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
