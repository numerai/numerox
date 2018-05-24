import datetime

from nose.tools import ok_
import pandas as pd
from numpy import nan

from numerox import testing
from numerox import report


def test_reports():
    "make sure low-level Report code runs"

    # leaderboard
    d = [[90, 'bot1', 0, 0, 0, 0, 0, 0.693, 0, nan, nan],
         [90, 'bot2', 1, 1, 1, 1, 0, 0.692, 1, 0.1, 10],
         [90, 'bot3', 0, 0, 0, 0, 3, 0.697, 3, 0.1, 30],
         [91, 'bot1', 2, 3, 1, 1, 0, 0.691, 0, 0.1, 30],
         [91, 'bot4', 0, 0, 0, 0, 0, 0.691, 0, nan, nan]]
    cols = ['round', 'user', 'usd_main', 'usd_stake', 'nmr_main', 'nmr_stake',
            'nmr_burn', 'live', 's', 'c', 'soc']
    lb = pd.DataFrame(data=d, columns=cols)

    # nmr price
    dd = datetime.date
    d = [[90, dd(2017, 12, 20), 52.12],
         [91, dd(2017, 12, 27), 62.34]]
    nmr_price = pd.DataFrame(data=d, columns=['round', 'date', 'usd'])
    nmr_price = nmr_price.set_index('round')

    # resolution dates
    d = [[89, dd(2017, 12, 13)],
         [90, dd(2017, 12, 20)],
         [91, dd(2017, 12, 27)],
         [92, dd(2018, 1, 4)]]
    resolution_dates = pd.DataFrame(data=d, columns=['round', 'date'])
    resolution_dates = resolution_dates.set_index('round')

    with testing.HiddenPrints():
        for verbose in (True, False):

            df = report.consistency(lb, 0)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.consistency(lb, 1)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.reputation(lb, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.group_consistency(lb, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.group_confidence(lb, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.group_burn(lb, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.ten99(lb, 'bot1', nmr_price, resolution_dates)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.stake(lb, 150.01, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.earn(lb, 150.01, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.burn(lb, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.participation(lb, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.big_staker(lb, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.headcount(lb, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.single_stake_payout(lb, 150.01, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.user_summary(lb, ['bot1'], 89.23, verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.user_nmr(lb, ['bot1'], resolution_dates, verbose=True)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.user_nmr(lb, ['bot4'], resolution_dates, verbose=True)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.user_nmr_tax(lb, ['bot1', 'bot4'], nmr_price,
                                     resolution_dates, price_zero_burns=True,
                                     verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            df = report.user_nmr_tax(lb, ['bot1', 'bot4'], nmr_price,
                                     resolution_dates, price_zero_burns=False,
                                     verbose=verbose)
            ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
            plist = report.user_participation(lb, 'bot3')
            ok_(isinstance(plist, list), 'expecting a list')
