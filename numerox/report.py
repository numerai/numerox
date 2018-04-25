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
        df = stake(self.lb[round1:round2], ntop)
        return df

    def earn(self, round1=61, round2=None, ntop=None):
        "Earnings report"
        df = earn(self.lb[round1:round2], ntop)
        return df


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


def stake(df, ntop):
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
    if ntop is not None:
        if ntop < 0:
            df = df[ntop:]
        else:
            df = df[:ntop]
    df = df.round()
    cols = ['usd_stake', 'nmr_stake', 'nmr_burn', 'profit_usd']
    df[cols] = df[cols].astype(int)
    return df


def earn(df, ntop):
    "Report on top earners"
    price = nx.token_price_data(ticker='nmr')['price']
    t1 = df['round'].min()
    t2 = df['round'].max()
    fmt = "Top earners (R{} - R{}) at {:.2f} usd/nmr"
    print(fmt.format(t1, t2, price))
    df = df.drop('round', axis=1)
    df = df.groupby('user').sum()
    profit = df['usd_main'] + df['usd_stake']
    nmr = df['nmr_main'] + df['nmr_stake'] - df['nmr_burn']
    profit += price * nmr
    df['profit_usd'] = profit
    df = df.sort_values('profit_usd', ascending=False)
    if ntop < 0:
        df = df[ntop:]
    else:
        df = df[:ntop]
    df = df.round()
    cols = ['usd_main', 'usd_stake', 'nmr_main', 'nmr_burn', 'profit_usd']
    df[cols] = df[cols].astype(int)
    return df
