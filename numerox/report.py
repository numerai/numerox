import numpy as np
import pandas as pd

import numerox as nx
from numerox.numerai import year_to_round_range
from numerox.numerai import round_resolution_date
from numerox.metrics import LOGLOSS_BENCHMARK

DEFAULT_FIRST_ROUND = 51


class Report(object):

    def __init__(self, tournament=1):
        self.tournament = tournament
        self.lb = nx.Leaderboard(tournament)

    def ten99(self, user, year=2017):
        "Generate unoffical 1099-MISC report"
        r1, r2 = year_to_round_range(year, self.tournament)
        df = ten99(self.lb[r1:r2], user, year, self.tournament)
        return df

    def consistency(self, round1, round2=None, min_participation_fraction=0.8):
        "User consistency"
        df = consistency(self.lb[round1:round2], min_participation_fraction)
        return df

    def stake(self, round1=61, round2=None, ntop=None):
        "Stake earnings report"
        df = stake(self.lb[round1:round2])
        df = ntopify(df, ntop)
        return df

    def earn(self, round1=61, round2=None, ntop=None):
        "Earnings report"
        df = earn(self.lb[round1:round2])
        df = ntopify(df, ntop)
        return df

    def burn(self, round1=61, round2=None, ntop=None):
        "Burn report"
        df = burn(self.lb[round1:round2])
        df = ntopify(df, ntop)
        return df

    def participation(self, round1=61, round2=None, ntop=None):
        "Participation report"
        df = participation(self.lb[round1:round2])
        df = ntopify(df, ntop)
        return df

    def big_staker(self, round1=61, round2=None, ntop=None):
        "Stake amount (in nmr) report"
        df = big_staker(self.lb[round1:round2])
        df = ntopify(df, ntop)
        return df

    def new_user(self, round1=61, round2=None):
        "Count of new users versus round number"
        df = new_user(self.lb[round1:round2])
        return df

    def user_participation(self, user, round1=61, round2=None):
        "List of rounds user participated in"
        r = user_participation(self.lb[round1:round2], user)
        return r

    def all(self, round1=61, round2=None):

        print_title(self.consistency)
        print(self.consistency(round1, round2))

        print_title(self.stake)
        print(self.stake(round1, round2))

        print_title(self.earn)
        print(self.earn(round1, round2))

        print_title(self.burn)
        print(self.burn(round1, round2))

        print_title(self.participation)
        print(self.participation(round1, round2))

        print_title(self.big_staker)
        print(self.big_staker(round1, round2))

        print_title(self.new_user)
        print(self.new_user(round1, round2))


def consistency(df, min_participation_fraction):
    "User consistency"
    df = df[['user', 'round', 'live']]
    df = df[~df['live'].isna()]
    df = df.drop_duplicates(['round', 'user'])
    df = df.pivot(index='user', columns='round', values='live')
    df = df[df.count(axis=1) >= min_participation_fraction * df.shape[1]]
    nrounds = df.count(axis=1)
    rounds = df.columns.tolist()
    idx1 = [r for r in rounds if r < 102]
    nwins1 = (df[idx1] < np.log(2)).sum(axis=1)
    idx2 = [r for r in rounds if r >= 102]
    nwins2 = (df[idx2] < LOGLOSS_BENCHMARK).sum(axis=1)
    nwins = nwins1 + nwins2
    consistency = pd.DataFrame()
    consistency['rounds'] = nrounds
    consistency['consistency'] = nwins / nrounds
    consistency = consistency.sort_values(['consistency', 'rounds'],
                                          ascending=[False, False])
    return consistency


def ten99(df, user, year, tournament):
    "Generate unoffical 1099-MISC report"
    df = df[df.user == user]
    df = df[['round', 'usd_main', 'usd_stake', 'nmr_main', 'nmr_stake']]
    df = df.set_index('round')
    nmrprice = nx.nmr_resolution_price(tournament=tournament)
    price = []
    for n in df.index:
        if n < 58:
            # nmr not yet traded on bittrex
            p = 0
        else:
            p = nmrprice.loc[n]['usd']
        price.append(p)
    df['nmr_usd'] = price
    total = df['usd_main'].values + df['usd_stake'].values
    nmr = df['nmr_main'].values + df['nmr_stake'].values
    total = total + nmr * df['nmr_usd'].values
    df['total'] = total
    earn = df['usd_main'] + df['nmr_main'] + df['nmr_stake'] + df['usd_stake']
    df = df[earn != 0]  # remove burn only rounds
    date = round_resolution_date(tournament=tournament)
    date = date.loc[df.index]
    df.insert(0, 'date', date)
    df['nmr_usd'] = df['nmr_usd'].round(2)
    df['total'] = df['total'].round(2)
    return df


def stake(df):
    "Earnings report of top stakers"
    price = nx.token_price_data(ticker='nmr')['price']
    t1 = df['round'].min()
    t2 = df['round'].max()
    fmt = "Top stake earners (R{} - R{}) at {:.2f} usd/nmr"
    print(fmt.format(t1, t2, price))
    df = df[['user', 'usd_stake', 'nmr_stake', 'nmr_burn']]
    df = df.groupby('user').sum()
    nmr = df['nmr_stake'] - df['nmr_burn']
    df['profit_usd'] = df['usd_stake'] + price * nmr
    df = df.sort_values('profit_usd', ascending=False)
    df = df.round()
    cols = ['usd_stake', 'nmr_stake', 'nmr_burn', 'profit_usd']
    df[cols] = df[cols].astype(int)
    return df


def earn(df):
    "Report on top earners"
    price = nx.token_price_data(ticker='nmr')['price']
    t1 = df['round'].min()
    t2 = df['round'].max()
    fmt = "Top earners (R{} - R{}) at {:.2f} usd/nmr"
    print(fmt.format(t1, t2, price))
    df = df[['user', 'usd_main', 'usd_stake', 'nmr_main', 'nmr_stake',
            'nmr_burn']]
    df = df.groupby('user').sum()
    profit = df['usd_main'] + df['usd_stake']
    nmr = df['nmr_main'] + df['nmr_stake'] - df['nmr_burn']
    profit += price * nmr
    df['profit_usd'] = profit
    df = df.sort_values('profit_usd', ascending=False)
    df = df.round()
    cols = ['usd_main', 'usd_stake', 'nmr_main', 'nmr_stake', 'nmr_burn',
            'profit_usd']
    df[cols] = df[cols].astype(int)
    return df


def burn(df):
    "Report on top burners"
    t1 = df['round'].min()
    t2 = df['round'].max()
    fmt = "Top burners (R{} - R{})"
    print(fmt.format(t1, t2))
    df = df[['user', 'nmr_burn']]
    df = df.groupby('user').sum()
    df = df.sort_values('nmr_burn', ascending=False)
    df = df.round()
    df = df.astype(int)
    return df


def participation(df):
    "Report on participation"
    t1 = df['round'].min()
    t2 = df['round'].max()
    fmt = "Participation (R{} - R{})"
    print(fmt.format(t1, t2))
    df = df[['user', 'round']]
    gb = df.groupby('user')
    # users appear twice in R44 so use nunique instead of count
    df_count = gb.nunique()
    df_count = df_count.rename({'round': 'count'}, axis='columns')
    df_first = gb.min()
    df_first = df_first.rename({'round': 'first'}, axis='columns')
    df_last = gb.max()
    df_last = df_last.rename({'round': 'last'}, axis='columns')
    df = pd.concat([df_count, df_first, df_last], axis=1)
    df['skipped'] = df['last'] - df['first'] + 1 - df['count']
    df = df.sort_values(['count', 'skipped'], ascending=[False, True])
    df = df.drop(['user'], axis=1)
    return df


def big_staker(df):
    "Report on big stakers"
    t1 = df['round'].min()
    t2 = df['round'].max()
    fmt = "Big stakers (in units of NMR) (R{} - R{})"
    print(fmt.format(t1, t2))
    pool = df['usd_stake'].abs() + df['nmr_burn'].abs() != 0
    df = df[['user', 's']]
    df.insert(2, 'pool', df['s'] * pool)
    df = df[df['s'] > 0]
    gb = df.groupby('user')
    df_sum = gb.sum()
    df_min = gb['s'].min()
    df_max = gb['s'].max()
    df_med = gb['s'].median()
    df_num = gb['s'].count()
    df_min = df_min.rename('min')
    df_max = df_max.rename('max')
    df_med = df_med.rename('median')
    df_num = df_num.rename('nstake')
    df = pd.concat([df_sum['s'].rename('sum'), df_max, df_med, df_min, df_num],
                   axis=1)
    df['aggressiveness'] = df_sum['pool'] / df_sum['s']
    df = df.sort_values(['sum', 'aggressiveness'], ascending=[False, False])
    return df


def new_user(df):
    "Count of new users versus round number"
    t1 = df['round'].min()
    t2 = df['round'].max()
    fmt = "Count of new users (R{} - R{})"
    print(fmt.format(t1, t2))
    df = df[['user', 'round']]
    df_first = df.groupby('user').min()
    df_first = df_first.rename({'round': 'first'}, axis='columns')
    data = []
    for r in range(t1, t2 + 1):
        n = (df_first['first'] == r).sum()
        data.append((r, n))
    df = pd.DataFrame(data=data, columns=['round', 'count'])
    df = df.set_index('round')
    return df


def user_participation(df, user):
    "List of rounds user participated in"
    df = df[['user', 'round']]
    idx = df['user'] == user
    if idx.sum() == 0:
        return []
    df = df[idx]
    # users appear twice in R44 so use unique
    r = df['round'].unique().tolist()
    return r


# ---------------------------------------------------------------------------
# utility functions

def ntopify(df, ntop):
    "Select top N (ntop > 0) or bottom N (ntop < 0) or all (ntop = None)"
    if ntop is not None:
        if ntop < 0:
            df = df[ntop:]
        else:
            df = df[:ntop]
    return df


def print_title(func):
    print('-' * 70)
    print('\n{}\n'.format(func.__name__.upper()))
