import numpy as np
import pandas as pd

import numerox as nx
from numerox.numerai import year_to_round_range
from numerox.numerai import round_resolution_date
from numerox.metrics import LOGLOSS_BENCHMARK


class Report(object):

    def __init__(self, tournament=1):
        self.tournament = tournament
        self.lb = nx.Leaderboard(tournament)
        self.resolution_date = None

    def ten99(self, user, year=2017):
        "Generate unoffical 1099-MISC report"
        r1, r2 = year_to_round_range(year, self.tournament)
        nmr_price = nx.nmr_resolution_price(tournament=self.tournament)
        resolution_dates = self.get_resolution_dates()
        df = ten99(self.lb[r1:r2], user, nmr_price, resolution_dates)
        return df

    def consistency(self, round1, round2=None, min_participation_fraction=0.8):
        "User consistency"
        df = consistency(self.lb[round1:round2], min_participation_fraction)
        return df

    def reputation(self, round1=51, round2=None, ntop=None):
        "Reputation report"
        df = reputation(self.lb[round1:round2])
        df = ntopify(df, ntop)
        return df

    def stake(self, round1=61, round2=None, ntop=None):
        "Stake earnings report"
        nmr_usd = nx.token_price_data(ticker='nmr')['price']
        df = stake(self.lb[round1:round2], nmr_usd)
        df = ntopify(df, ntop)
        return df

    def earn(self, round1=61, round2=None, ntop=None, use_ints=True):
        "Earnings report"
        nmr_usd = nx.token_price_data(ticker='nmr')['price']
        df = earn(self.lb[round1:round2], nmr_usd)
        df = ntopify(df, ntop)
        if use_ints:
            df = df.round()
            cols = ['usd_main', 'usd_stake', 'nmr_main', 'nmr_stake',
                    'nmr_burn', 'profit_usd']
            df[cols] = df[cols].astype(int)
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

    def headcount(self, round1=61, round2=None):
        "Count of users versus round number"
        df = headcount(self.lb[round1:round2])
        return df

    def single_stake_payout(self, round1=61, round2=None):
        "Largest stake payouts"
        nmr_usd = nx.token_price_data(ticker='nmr')['price']
        df = single_stake_payout(self.lb[round1:round2], nmr_usd)
        return df

    def user_summary(self, users, round1=61, round2=None):
        "Summary report on user(s)"
        nmr_usd = nx.token_price_data(ticker='nmr')['price']
        if nx.isstring(users):
            users = [users]
        df = user_summary(self.lb[round1:round2], users, nmr_usd)
        return df

    def user_nmr(self, users, round1=61, round2=None):
        "User(s) nmr details"
        if nx.isstring(users):
            users = [users]
        resolution_dates = self.get_resolution_dates()
        df = user_nmr(self.lb[round1:round2], users, resolution_dates)
        return df

    def user_nmr_tax(self, users, round1=61, round2=None,
                     price_zero_burns=True):
        """
        User(s) nmr tax details.

        Price of nmr (in usd) before round 58 (i.e before nmr was traded on
        and exchange) is set to 0.

        Price of nmr for burns is optionally set to zero by default.
        """
        if nx.isstring(users):
            users = [users]
        nmr_price = nx.nmr_resolution_price(tournament=self.tournament)
        resolution_dates = self.get_resolution_dates()
        df = user_nmr_tax(self.lb[round1:round2], users, nmr_price,
                          resolution_dates, price_zero_burns)
        return df

    def user_participation(self, user, round1=61, round2=None):
        "List of rounds user participated in"
        r = user_participation(self.lb[round1:round2], user)
        return r

    def group_consistency(self, round1=61, round2=None):
        "Consistency among various groups of users"
        df = group_consistency(self.lb[round1:round2])
        return df

    def group_confidence(self, round1=61, round2=None):
        "Linearly interpolated confidence at prize-pool cutoff"
        df = group_confidence(self.lb[round1:round2])
        return df

    def group_burn(self, round1=61, round2=None):
        "Total NMR burn per round"
        df = group_burn(self.lb[round1:round2])
        return df

    def all(self, round1=61, round2=None):

        print_title(self.consistency)
        print(self.consistency(round1, round2))

        print_title(self.reputation)
        print(self.reputation(round1, round2))

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

        print_title(self.headcount)
        print(self.headcount(round1, round2))

        print_title(self.single_stake_payout)
        print(self.single_stake_payout(round1, round2))

        print_title(self.group_consistency)
        print(self.group_consistency(round1, round2))

        print_title(self.group_confidence)
        print(self.group_confidence(round1, round2))

        print_title(self.group_burn)
        print(self.group_burn(round1, round2))

    def get_resolution_dates(self):
        if self.resolution_date is None:
            dates = round_resolution_date(tournament=self.tournament)
            self.resolution_date = dates
        return self.resolution_date


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


def reputation(df, verbose=True):
    "Reputation report"

    # display round range
    if verbose:
        t1 = df['round'].min()
        t2 = df['round'].max()
        fmt = "Reputation (sorted by points, username) (R{} - R{})"
        print(fmt.format(t1, t2))

    # pass logloss benchmark?
    df = df[['user', 'round', 'live']]
    df = df.drop_duplicates(['round', 'user'])
    df_pass1 = df['live']
    df_pass1 = 1.0 * (df_pass1 < np.log(2))
    df_pass1[df['round'] > 101] = 0
    df_pass2 = df['live']
    df_pass2 = 1.0 * (df_pass2 < LOGLOSS_BENCHMARK)
    df_pass2[df['round'] < 102] = 0
    df_pass = df_pass1 + df_pass2
    df.insert(3, 'pass', df_pass)

    # how many points?
    df_points = df.groupby('user').sum()['pass']

    # how many precious nmr?
    nmr1 = df[df['round'] < 100].groupby('user').sum()['pass']
    nmr2 = df[df['round'] >= 100].groupby('user').sum()['pass']
    nmr2 = 0.1 * nmr2
    df_nmr = nmr1.add(nmr2, fill_value=0)

    # how many rounds?
    df_rounds = df.groupby('user').count()['live']

    # put it all together
    df = pd.concat([df_points, df_nmr, df_rounds], axis=1)
    df.columns = ['points', 'nmr', 'rounds']
    df['index'] = df.index
    df = df.sort_values(['points', 'index'], ascending=[False, True])
    df = df.drop('index', axis=1)

    return df


def group_consistency(df, verbose=True):
    "Consistency among various groups of users"

    # display round range
    if verbose:
        t1 = df['round'].min()
        t2 = df['round'].max()
        fmt = "Group consistency (R{} - R{})"
        print(fmt.format(t1, t2))

    # pass logloss benchmark?
    df = df[['user', 'round', 'live', 's']]
    df_pass1 = df['live']
    df_pass1 = 1.0 * (df_pass1 < np.log(2))
    df_pass1[df['round'] > 101] = 0
    df_pass2 = df['live']
    df_pass2 = 1.0 * (df_pass2 < LOGLOSS_BENCHMARK)
    df_pass2[df['round'] < 102] = 0
    df_pass = df_pass1 + df_pass2
    df.insert(3, 'pass', df_pass)

    # consistency
    df = df.drop_duplicates(['round', 'user'])
    df_overall = df.groupby('round').mean()['pass']
    df_nonstake = df[df['s'] == 0].groupby('round').mean()['pass']
    df_stake = df[df['s'] > 0].groupby('round').mean()['pass']

    # put it all together
    df = pd.concat([df_overall, df_nonstake, df_stake], axis=1)
    df.columns = ['overall', 'nonstake', 'stake']

    return df


def group_confidence(df, verbose=True):
    "Linearly interpolated confidence at prize-pool cutoff"

    # display round range
    t1 = df['round'].min()
    if t1 < 61:
        t1 = 61
    t2 = df['round'].max()
    if t1 < 61:
        t1 = 61
    if verbose:
        fmt = "Linearly interpolated confidence at prize-pool cutoff "
        fmt += "(R{} - R{})"
        print(fmt.format(t1, t2))

    # only keep the data that we need
    df = df[['round', 's', 'c', 'soc', 'nmr_burn']]
    df = df[df.s != 0]

    # loop through each round
    data = []
    for r in range(t1, t2 + 1):

        if r < 78:
            cutoff = 3000
        else:
            cutoff = 6000

        c = [0, 0]
        for i in range(2):
            stakes = df[df['round'] == r]
            if stakes.shape[0] < 2:
                continue
            if i == 1:
                stakes = stakes[stakes.nmr_burn == 0]
            stakes = stakes.sort_values(by='c', ascending=False)
            cumsum = stakes.soc.cumsum(axis=0) - stakes.soc  # dollars above
            if cumsum.max() < cutoff:
                continue
            stakes.insert(4, 'cumsum', cumsum)
            x = stakes['c'].values.astype(np.float64)
            y = stakes['cumsum'].values
            idx = np.isfinite(x + y)
            x = x[idx]
            y = y[idx]
            c[i] = np.interp(cutoff, y, x)

        data.append([r, c[0], c[1]])

    # jam into dataframe
    df = pd.DataFrame(data=data,
                      columns=['round', 'cutoff', 'resolved_cutoff'])
    df = df.set_index('round')

    return df


def group_burn(df, verbose=True):
    "Total NMR burn per round"

    # display round range
    if verbose:
        t1 = df['round'].min()
        t2 = df['round'].max()
        fmt = "NMR burned (R{} - R{})"
        print(fmt.format(t1, t2))

    # nmr burned
    df = df.drop_duplicates(['round', 'user'])
    df = df[['round', 'nmr_burn', 's']]
    df = df.groupby('round').sum()
    df = df[['nmr_burn', 's']]
    df.columns = ['burn', 'staked']
    df.insert(2, 'fraction', df['burn'] / df['staked'])

    return df


def ten99(df, user, nmr_price, resolution_dates):
    "Generate unoffical 1099-MISC report"
    df = df[df.user == user]
    df = df[['round', 'usd_main', 'usd_stake', 'nmr_main', 'nmr_stake']]
    df = df.set_index('round')
    price = []
    for n in df.index:
        if n < 58:
            # nmr not yet traded on bittrex
            p = 0
        else:
            p = nmr_price.loc[n]['usd']
        price.append(p)
    df['nmr_usd'] = price
    total = df['usd_main'].values + df['usd_stake'].values
    nmr = df['nmr_main'].values + df['nmr_stake'].values
    total = total + nmr * df['nmr_usd'].values
    df['total'] = total
    earn = df['usd_main'] + df['nmr_main'] + df['nmr_stake'] + df['usd_stake']
    df = df[earn != 0]  # remove burn only rounds
    date = resolution_dates
    date = date.loc[df.index]
    df.insert(0, 'date', date)
    df['nmr_usd'] = df['nmr_usd'].round(2)
    df['total'] = df['total'].round(2)
    return df


def stake(df, price, verbose=True):
    "Earnings report of top stakers"
    if verbose:
        t1 = df['round'].min()
        t2 = df['round'].max()
        fmt = "Top stake earners (R{} - R{}) at {:.2f} usd/nmr"
        print(fmt.format(t1, t2, price))
    df = df[df.s > 0]
    df = df[['user', 'usd_stake', 'nmr_stake', 'nmr_burn']]
    df = df.groupby('user').sum()
    nmr = df['nmr_stake'] - df['nmr_burn']
    df['profit_usd'] = df['usd_stake'] + price * nmr
    df = df.sort_values('profit_usd', ascending=False)
    df = df.round()
    cols = ['usd_stake', 'nmr_stake', 'nmr_burn', 'profit_usd']
    df[cols] = df[cols].astype(int)
    return df


def earn(df, price, verbose=True):
    "Report on top earners"
    if verbose:
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
    return df


def burn(df, verbose=True):
    "Report on top burners"
    if verbose:
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


def participation(df, verbose=True):
    "Report on participation"
    if verbose:
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


def big_staker(df, verbose=True):
    "Report on big stakers"
    if verbose:
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


def headcount(df, verbose=True):
    "Count of users versus round number"
    t1 = df['round'].min()
    t2 = df['round'].max()
    if verbose:
        fmt = "Count of users (R{} - R{})"
        print(fmt.format(t1, t2))
    df = df[['user', 'round', 's', 'nmr_burn']]
    df2 = df[df.s != 0]
    df3 = df[df.s == 0]
    df4 = df[df.nmr_burn > 0]
    df2_first = df2.groupby('user').min()
    df3_first = df3.groupby('user').min()
    df4_first = df4.groupby('user').min()
    data = []
    for r in range(t1, t2 + 1):
        total = (df['round'] == r).sum()
        stake = (df2['round'] == r).sum()
        nonstake = (df3['round'] == r).sum()
        new_stake = (df2_first['round'] == r).sum()
        new_nonstake = (df3_first['round'] == r).sum()
        new_burn = (df4_first['round'] == r).sum()
        data.append((r, total, stake, nonstake, new_stake, new_nonstake,
                     new_burn))
    cols = ['round', 'total', 'stake', 'nonstake', 'new_stake',
            'new_nonstake', 'new_burn']
    df = pd.DataFrame(data=data, columns=cols)
    df = df.set_index('round')
    return df


def single_stake_payout(df, price, verbose=True):
    "Largest stake payouts"
    t1 = df['round'].min()
    t2 = df['round'].max()
    if verbose:
        fmt = "Largest stake payouts (R{} - R{}) at {:.2f} usd/nmr"
        print(fmt.format(t1, t2, price))
    df = df[['round', 'user', 's', 'c', 'usd_stake', 'nmr_stake', 'nmr_burn']]
    nmr = df['nmr_stake'] - df['nmr_burn']
    profit_usd = df['usd_stake'] + price * nmr
    df.insert(4, 'profit_usd', profit_usd)
    df = df.drop(['usd_stake', 'nmr_stake', 'nmr_burn'], axis=1)
    df = df.sort_values('profit_usd', ascending=False)
    df = df.round()
    df = df.reset_index(drop=True)
    return df


def user_summary(df, users, price, verbose=True):
    "Summary report on user(s)"

    if verbose:
        t1 = df['round'].min()
        t2 = df['round'].max()
        fmt = "User(s) summary (R{} - R{})"
        print(fmt.format(t1, t2))

    idx = df.user.isin(users)
    df = df[idx]

    s = pd.DataFrame(columns=users)

    c = consistency(df, 0)
    s.loc['rounds'] = c['rounds']

    r = reputation(df, verbose=False)
    s.loc['rep_points'] = r['points']
    s.loc['consistency'] = c['consistency']

    e = earn(df, price, verbose=False)
    s.loc['profit_usd'] = e['profit_usd']
    s.loc['nmr_burn'] = e['nmr_burn']

    bs = big_staker(df, verbose=False)
    s.loc['nmr_staked'] = bs['sum']
    s.loc['median_stake'] = bs['median']
    s.loc['max_stake'] = bs['max']

    st = stake(df, price, verbose=False)
    s.loc['stake_profit_usd'] = st['profit_usd']

    d = df[['user']]
    nmr = df['nmr_main'] + df['nmr_stake']
    d.insert(1, 'nmr', nmr)
    nmr = d.groupby('user').sum()['nmr']
    s.loc['nmr_earn'] = nmr

    columns = ['rep_points', 'rounds', 'consistency', 'profit_usd',
               'nmr_staked', 'median_stake', 'max_stake', 'nmr_burn',
               'nmr_earn']
    s = s.loc[columns]

    for row in columns:
        if row != 'consistency':
            s.loc[row] = s.loc[row].fillna(0)

    return s


def user_nmr(df, users, resolution_dates, verbose=True):
    "User(s) nmr details"
    if verbose:
        t1 = df['round'].min()
        t2 = df['round'].max()
        fmt = "User(s) nmr detail (R{} - R{})"
        print(fmt.format(t1, t2))
    idx = df.user.isin(users)
    df = df[idx]
    df = df[['round', 'user', 'nmr_main', 'nmr_stake', 'nmr_burn']]
    idx = df['nmr_main'] + df['nmr_stake'] + df['nmr_burn'].abs() != 0
    df = df[idx]
    df = df.reset_index(drop=True)
    net_nmr = df['nmr_main'] + df['nmr_stake'] - df['nmr_burn']
    df.insert(5, 'net_nmr', net_nmr)
    date = resolution_dates
    date = date.loc[df['round']]
    if df.shape[0] > 0:
        df.insert(0, 'date', date.values)
    return df


def user_nmr_tax(df, users, nmr_price, resolution_dates, price_zero_burns,
                 verbose=True):
    "User(s) nmr tax details"
    if verbose:
        t1 = df['round'].min()
        t2 = df['round'].max()
        fmt = "User(s) nmr tax detail (R{} - R{})"
        print(fmt.format(t1, t2))

    df = user_nmr(df, users, resolution_dates, verbose=False)
    price = []
    for n in df['round']:
        if n < 58:
            # nmr not yet traded on bittrex
            p = 0
        else:
            p = nmr_price.loc[n]['usd']
        price.append(p)
    df['nmr_usd'] = price

    data = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if row['nmr_main'] > 0:
            data.append([row['date'],
                         'nmr',
                         row['nmr_main'],
                         row['nmr_usd'] * row['nmr_main'],
                         'main',
                         row['user']])
        if row['nmr_stake'] > 0:
            data.append([row['date'],
                         'nmr',
                         row['nmr_stake'],
                         row['nmr_usd'] * row['nmr_stake'],
                         'stake',
                         row['user']])
        if row['nmr_burn'] > 0:
            if price_zero_burns:
                value = 0
            else:
                value = -row['nmr_usd'] * row['nmr_burn']
            data.append([row['date'],
                         'nmr',
                         -row['nmr_burn'],
                         value,
                         'burn',
                         row['user']])

    columns = ['date', 'ticker', 'shares', 'usd', 'type', 'acct']
    df = pd.DataFrame(data=data, columns=columns)

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
