from nose.tools import ok_
import pandas as pd

from numerox import report


def leaderboard():
    d = [[113, 1, True, 'bot1', 0, 0, 0, 0, 0, 0.690, 0.683, 58, 1, 0.2, 10],
         [113, 1, True, 'bot2', 1, 1, 1, 1, 0, 0.692, 0.682, 66, 1, 0.02, 10],
         [113, 1, True, 'bot3', 0, 0, 0, 0, 3, 0.692, 0.682, 75, 3, 0.1, 30],
         [113, 1, True, 'bot4', 2, 3, 1, 1, 0, 0.691, 0.682, 91, 2, 0.1, 30],
         [113, 1, True, 'bot5', 0, 0, 0, 0, 0, 0.691, 0.682, 50, 3, 0.3, 30]]
    cols = ['round', 'tournament', 'resolved', 'user', 'usd_main', 'usd_stake',
            'nmr_main', 'nmr_stake', 'nmr_burn', 'live', 'val', 'consis', 's',
            'c', 'soc']
    lb = pd.DataFrame(data=d, columns=cols)
    return lb


def test_reports():
    "make sure low-level Report code runs"

    data = [[113, None, None, 10, 10, 0.0]]
    cols = ['round', 'open_date', 'resolve_date', 'open_usd', 'resolve_usd',
            'return']
    prices = pd.DataFrame(data=data, columns=cols)
    prices = prices.set_index('round')

    df = report.summary(leaderboard(), prices)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.summary_user(leaderboard(), 'bot1')
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.payout(leaderboard())
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.payout_users(leaderboard(), 'bot1')
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
    df = report.payout_users(leaderboard(), ['bot1', 'bot2'])
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.cutoff(leaderboard())
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.whatif(leaderboard(), ['bot1', 'bot2'], 1, 0.999)
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.dominance(leaderboard(), 'bot1')
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.logloss(leaderboard(), 'bot1')
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.pass_rate(leaderboard())
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.out_of_five(leaderboard())
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.five_star_club(leaderboard())
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.logloss_correlation(leaderboard())
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.friends(leaderboard(), 'bot1')
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')

    df = report.val_v_live_consistency(leaderboard())
    ok_(isinstance(df, pd.DataFrame), 'expecting a dataframe')
