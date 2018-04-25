from nose.tools import ok_

import pandas as pd

from numerox.leaderboard import raw_leaderboard_to_df


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
