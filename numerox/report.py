import numpy as np
import pandas as pd

import numerox as nx
from numerox.metrics import LOGLOSS_BENCHMARK
from numerox.numerai import calc_cutoff


class Report(object):

    def __init__(self):
        self.lb = nx.Leaderboard()

    def summary(self, round1, round2):
        "Round summary"
        lb = self.lb[round1:round2]
        prices = nx.nmr_round_prices()
        df = summary(lb, prices)
        return df

    def summary_user(self, user, round1, round2):
        "Round summary for `user`"
        lb = self.lb[round1:round2]
        df = summary_user(lb, user)
        return df

    def payout(self, round1, round2):
        "NMR and USD payouts per round"
        lb = self.lb[round1:round2]
        df = payout(lb)
        return df

    def payout_users(self, users, round1, round2):
        "NMR and USD payouts per round for given `users`"
        lb = self.lb[round1:round2]
        df = payout_users(lb, users)
        return df

    def cutoff(self, round1, round2, cache_current_round=True):
        "Independent calculation of confidence cutoff"
        crn = nx.get_current_round_number()
        if not cache_current_round:
            if round1 == crn and round2 == crn:
                b = nx.Leaderboard()
                lb = b[crn]
            elif round2 == crn:
                lb = self.lb[round1:round2 - 1]
                b = nx.Leaderboard()
                lb2 = b[crn]
                lb = pd.concat([lb, lb2])
            else:
                lb = self.lb[round1:round2]
        else:
            lb = self.lb[round1:round2]
        df = cutoff(lb)
        return df

    def whatif(self, users, s, c, round1, round2):
        """
        Profit if `users` had staked `s` and `c` in every tournament.

        Earnings are left in NMR instead of splitting the NMR earnings into
        NMR and USD.

        """
        if nx.isint(round1):
            if round1 < 113:
                raise ValueError("`round1` must start at at least 113")
        lb = self.lb[round1:round2]
        df = whatif(lb, users, s, c)
        return df

    def dominance(self, user, round1, round2):
        "Fraction of users that `user` beats in terms of live logloss."
        lb = self.lb[round1:round2]
        df = dominance(lb, user)
        return df

    def logloss(self, user, round1, round2):
        "Live logloss for `user`"
        lb = self.lb[round1:round2]
        df = logloss(lb, user)
        return df

    def pass_rate(self, round1, round2):
        "Fraction of users who beat benchmark in each round"
        lb = self.lb[round1:round2]
        df = pass_rate(lb)
        return df

    def out_of_five(self, round1, round2):
        "Fraction of users that get, e.g., 3/5 in a round"
        lb = self.lb[round1:round2]
        df = out_of_five(lb)
        return df

    def five_star_club(self, round1):
        "Users who beat benchmark in all 5 tournaments sorted by mean logloss"
        lb = self.lb[round1]
        df = five_star_club(lb)
        return df

    def logloss_correlation(self, round1, round2):
        """
        Mean correlation of a users live logloss to all other users.

        Only those that have submitted in every tournament are considered.

        """
        lb = self.lb[round1:round2]
        df = logloss_correlation(lb)
        return df

    def friends(self, user, round1, round2):
        """
        Correlation of live logloss of each user to a given `user` and
        Euclidean distance.

        Only those that have submitted in every tournament are considered. So
        given `user` must have submitted in every tournament.

        """

        lb = self.lb[round1:round2]
        df = friends(lb, user)
        return df

    def val_v_live_consistency(self, round1, round2):
        "Live consistency versus validation consistency"
        lb = self.lb[round1:round2]
        df = val_v_live_consistency(lb)
        return df


def summary(lb, prices):
    "Round summary"

    rounds = np.sort(lb['round'].unique())
    df = pd.DataFrame(columns=rounds)

    pr = pass_rate(lb)
    df.loc['pass rate'] = pr['all']

    five = out_of_five(lb)
    df.loc['mean/5'] = five['mean/5']
    mode = five.iloc[:, 1:-1].idxmax(axis=1).tolist()[:-1]
    try:
        mode = [int(m[0]) for m in mode]
    except TypeError:
        mode = np.nan
    df.loc['mode/5'] = mode
    df.loc['fraction 5/5'] = five['5/5']

    co = cutoff(lb)
    df.loc['cutoff'] = co['mean']

    pay = payout(lb)
    df.loc['staked > cutoff'] = pay['staked_above_cutoff']
    df.loc['burned'] = pay['burned_nmr']
    df.loc['usd+nmr in nmr'] = pay['total_payout_in_nmr']
    rp = pay['total_payout_in_nmr'] - pay['burned_nmr']
    rp = rp / pay['staked_above_cutoff']
    df.loc['stake profit/nmr (nmr)'] = rp

    rounds = np.sort(lb['round'].unique())
    users = []
    peruser = []
    for r in rounds:
        d = lb[lb['round'] == r]
        n = d.shape[0]
        nu = len(d['user'].unique())
        users.append(nu)
        peruser.append(1.0 * n / nu)
    p = prices.loc[rounds]

    df.loc['stake profit/nmr (usd)'] = rp * p['resolve_usd']
    sp = p['open_usd'] / p['resolve_usd'] - 1.0
    df.loc['sell profit/nmr (nmr)'] = sp
    df.loc['sell profit/nmr (usd)'] = sp * p['resolve_usd']
    df.loc['open price'] = p['open_usd']
    df.loc['resolve price'] = p['resolve_usd']
    df.loc['nmr return'] = p['return']
    df.loc['users'] = users
    df.loc['tourneys/user'] = peruser

    df = df.round(2)

    return df


def summary_user(lb, user):
    "Round summary for `user`"

    rounds = np.sort(lb['round'].unique())
    df = pd.DataFrame(columns=rounds)

    idx = lb.user == user
    if idx.sum() == 0:
        df.loc['pass'] = '0/0'
        for item in ('logloss', 'dominance', 'correlation'):
            df.loc[item] = np.nan
        for item in ('staked', 'burned', 'nmr earn', 'usd earn'):
            df.loc[item] = 0
    else:
        d = lb[idx]
        sums = []
        subs = []
        for r in rounds:
            live = d[d['round'] == r]['live']
            if live.shape[0] == 0:
                s = np.nan
            else:
                s = (live < LOGLOSS_BENCHMARK).sum()
            subs.append(live.size)
            sums.append(s)
        p = [str(i) + '/' + str(j) for i, j in zip(sums, subs)]
        df.loc['pass'] = p

        ll = logloss(lb, user)
        df.loc['logloss'] = ll['mean']

        do = dominance(lb, user)
        df.loc['dominance'] = do['mean']

        corr = []
        for r in rounds:
            dr = friends(lb[lb['round'] == r], user)
            c = dr['correlation'].mean()
            corr.append(c)
        df.loc['correlation'] = corr

        pay = payout_users(lb, user)
        df.loc['staked'] = pay['nmr_staked']
        df.loc['burned'] = pay['nmr_burn']
        df.loc['nmr earn'] = pay['nmr_earn']
        df.loc['usd earn'] = pay['usd_earn']

    return df


def payout(lb):
    "NMR and USD payouts per round"
    cols = ['staked_nmr', 'staked_above_cutoff', 'burned_nmr',
            'nmr_payout', 'usd_payout', 'total_payout_in_nmr']
    df = pd.DataFrame(columns=cols)
    rounds = np.sort(lb['round'].unique())
    lb.insert(0, 'pass', lb['live'] < LOGLOSS_BENCHMARK)
    for r in rounds:
        d = lb[lb['round'] == r]
        if r > 112:
            nmr_cut = 0
            nmr_cut_pass = 0
            for t in nx.tournament_all(as_str=False):
                dt = d[d.tournament == t]
                cutoff, ignore = calc_cutoff(dt)
                nmr_cut += dt[dt.c >= cutoff].sum()['s']
                nmr_cut_pass += dt[(dt.c >= cutoff) & (dt['pass'])].sum()['s']
        else:
            nmr_cut = np.nan
        if cutoff == 0:
            total = np.nan
        else:
            total = nmr_cut_pass * (1.0 - cutoff) / cutoff
        ds = d.sum()
        pay = [ds['s'], nmr_cut, ds['nmr_burn'], ds['nmr_stake'],
               ds['usd_stake'], total]
        df.loc[r] = pay
    fraction = df['burned_nmr'] / df['staked_above_cutoff']
    df.insert(3, 'fraction_burned', fraction)
    df.loc['mean'] = df.mean()
    df = df.round(2)
    return df


def payout_users(lb, users):
    "NMR and USD payouts per round for given `users`"
    if isinstance(users, list):
        pass
    elif nx.isstring(users):
        users = [users]
    else:
        raise ValueError("`users` must be str or list (of str)")
    cols = ['nmr_staked', 'nmr_burn', 'nmr_earn', 'usd_earn']
    df = pd.DataFrame(columns=cols)
    rounds = np.sort(lb['round'].unique())
    for r in rounds:
        d = lb[lb['round'] == r]
        idx = d.user.isin(users)
        d = d[idx]
        ds = d.sum()
        pay = [ds['s'], ds['nmr_burn'], ds['nmr_stake'], ds['usd_stake']]
        df.loc[r] = pay
    df.loc['total'] = df.sum()
    return df


def cutoff(lb):
    "Independent calculation of confidence cutoff"
    cols = nx.tournament_all(as_str=True)
    df = pd.DataFrame(columns=cols)
    rounds = np.sort(lb['round'].unique())
    for r in rounds:
        d = lb[lb['round'] == r]
        if r > 112:
            cut = []
            for t in nx.tournament_all(as_str=False):
                dt = d[d.tournament == t]
                cutoff, ignore = calc_cutoff(dt)
                cut.append(cutoff)
        else:
            cut = [np.nan] * 5
        df.loc[r] = cut
    df['mean'] = df.mean(axis=1)
    df.loc['mean'] = df.mean()
    return df


def whatif(lb, users, s, c):
    """
    Profit if `users` had staked `s` and `c` in every tournament.

    Earnings are left in NMR instead of splitting the NMR earnings into
    NMR and USD.

    """
    if isinstance(users, list):
        pass
    elif nx.isstring(users):
        users = [users]
    else:
        raise ValueError("`users` must be str or list (of str)")
    cols = ['nmr_staked', 'nmr_burn', 'nmr_earn', 'nmr_net']
    df = pd.DataFrame(columns=cols)
    lb.insert(0, 'pass', lb['live'] < LOGLOSS_BENCHMARK)
    rounds = np.sort(lb['round'].unique())
    for r in rounds:
        d = lb[lb['round'] == r]
        if r > 112:
            staked = 0
            burn = 0
            earn = 0
            for t in nx.tournament_all(as_str=False):
                dt = d[d.tournament == t]
                if dt.shape[0] > 0:
                    cutoff, ignore = calc_cutoff(dt)
                    if c >= cutoff:
                        idx = dt.user.isin(users)
                        dti = dt[idx]
                        idx = dti['pass']
                        nwin = idx.sum()
                        nlos = (~idx & (dti['live'].notna())).sum()
                        p = (1.0 - cutoff) / cutoff
                        burn += nlos * s
                        earn += nwin * s * p
                        staked += idx.size * s
            net = earn - burn
            df.loc[r] = [staked, burn, earn, net]
        else:
            raise ValueError("`round1` must start at at least 113")
    df.loc['total'] = df.sum()
    return df


def dominance(lb, user):
    "Fraction of users that `user` beats in terms of live logloss."
    cols = nx.tournament_all()
    df = pd.DataFrame(columns=cols)
    lb = lb[['user', 'round', 'tournament', 'live']]
    rounds = np.sort(lb['round'].unique())
    for r in rounds:
        d = lb[lb['round'] == r]
        dom = []
        for t in nx.tournament_all(as_str=False):
            dt = d[d.tournament == t]
            dt = dt[dt.live.notna()]
            if user in dt.user.values:
                dm = (dt[dt.user == user].live.iloc[0] < dt.live).mean()
                dom.append(dm)
            else:
                dom.append(np.nan)
        df.loc[r] = dom
    df['mean'] = df.mean(axis=1)
    df.loc['mean'] = df.mean()
    return df


def logloss(lb, user):
    "Live logloss for `user`"
    cols = nx.tournament_all()
    df = pd.DataFrame(columns=cols)
    lb = lb[['user', 'round', 'tournament', 'live']]
    rounds = np.sort(lb['round'].unique())
    for r in rounds:
        d = lb[lb['round'] == r]
        dom = []
        for t in nx.tournament_all(as_str=False):
            dt = d[d.tournament == t]
            if user in dt.user.values:
                dm = dt[dt.user == user].live.iloc[0]
                dom.append(dm)
            else:
                dom.append(np.nan)
        df.loc[r] = dom
    df['mean'] = df.mean(axis=1)
    df.loc['mean'] = df.mean()
    return df


def pass_rate(lb):
    "Fraction of users who beat benchmark in each round"
    cols = ['all', 'stakers', 'nonstakers', 'above_cutoff', 'below_cutoff']
    df = pd.DataFrame(columns=cols)
    rounds = np.sort(lb['round'].unique())
    for r in rounds:
        d = lb[(lb['round'] == r) & (lb.live.notna())]
        d.insert(0, 'pass', d['live'] < LOGLOSS_BENCHMARK)
        pr_all = d['pass'].mean()
        pr_stakers = d[d['s'] > 0]['pass'].mean()
        pr_nonstakers = d[d['s'] == 0]['pass'].mean()
        if r > 112:
            nabove = 0
            nbelow = 0
            pabove = 0
            pbelow = 0
            for t in nx.tournament_all(as_str=False):
                dt = d[d.tournament == t]
                cutoff, ignore = calc_cutoff(dt)
                nabove += dt[dt.c > cutoff].shape[0]
                nbelow += dt[dt.c < cutoff].shape[0]
                pabove += dt[(dt.c > cutoff) & (dt['pass'])].shape[0]
                pbelow += dt[(dt.c < cutoff) & (dt['pass'])].shape[0]
            if nabove == 0:
                pr_above = np.nan
            else:
                pr_above = 1.0 * pabove / nabove
            if nbelow == 0:
                pr_below = np.nan
            else:
                pr_below = 1.0 * pbelow / nbelow
        else:
            pr_above = np.nan
            pr_below = np.nan
        df.loc[r] = [pr_all, pr_stakers, pr_nonstakers, pr_above, pr_below]
    df.loc['mean'] = df.mean()
    return df


def out_of_five(lb):
    "Fraction of users that get, e.g., 3/5 in a round"
    cols = ['N', '0/5', '1/5', '2/5', '3/5', '4/5', '5/5', 'mean/5']
    df = pd.DataFrame(columns=cols)
    rounds = np.sort(lb['round'].unique())
    nan = np.nan
    for r in rounds:
        d = lb[lb['round'] == r]
        if not d['resolved'].any():
            fraction = [nan, nan, nan, nan, nan, nan, nan, nan]
        else:
            idx = (d.groupby('user').count()['round'] == 5)
            idx = idx[idx]
            idx = d.user.isin(idx.index)
            d = d[idx]
            d['pass'] = d['live'] < LOGLOSS_BENCHMARK
            s = d.groupby('user').sum()
            rep = s.groupby('pass').count()
            rep = rep['round'].to_frame('count')
            count = rep['count'].sum()
            fraction = 1.0 * rep['count'] / count
            if fraction.size != 6:
                # TODO: should handle case where some are missing
                fraction = np.array([np.nan] * 6)
            mean = np.dot(fraction, np.array([0, 1, 2, 3, 4, 5]))
            fraction = fraction.tolist()
            fraction.insert(0, count)
            fraction.insert(7, mean)
        df.loc[r] = fraction
    df.loc['mean'] = df.mean()
    df['N'] = df['N'].astype(int)
    return df


def five_star_club(lb):
    "Users who beat benchmark in all 5 tournaments sorted by mean logloss"
    lb['pass'] = lb['live'] < LOGLOSS_BENCHMARK
    s = lb.groupby('user').sum()
    df = s[s['pass'] == 5]
    df = df[['live']] / 5
    df.columns = ['mean_logloss']
    df = df.sort_values('mean_logloss')
    return df


def logloss_correlation(lb):
    """
    Mean correlation of a users live logloss to all other users.

    Only those that have submitted in every tournament are considered.

    """
    lb = lb[lb.resolved]
    lb.insert(0, 'rt', lb['round'] * 10 + lb['tournament'])
    lb = lb[['user', 'rt', 'live']]
    lb = lb.set_index('user')
    lb = lb.pivot(columns='rt', values='live')
    lb = lb.dropna()
    corr = lb.T.corr()
    df = (corr.sum(axis=1) - 1) / (corr.shape[1] - 1)
    df = df.sort_values()
    df = df.to_frame('mean_correlation')
    return df


def friends(lb, user):
    """
    Correlation of live logloss of each user to a given `user` and
    Euclidean distance.

    Only those that have submitted in every tournament are considered. So
    given `user` must have submitted in every tournament.

    """

    lb.insert(0, 'rt', lb['round'] * 10 + lb['tournament'])
    lb = lb[['user', 'rt', 'live']]
    lb = lb.set_index('user')
    lb = lb.pivot(columns='rt', values='live')
    lb = lb.dropna(axis=1, how='all')
    lb = lb.dropna(axis=0)

    corr = lb.T.corr()
    corr[corr == 1] = np.nan
    df = corr.loc[user]
    df = df.to_frame('correlation')

    d = lb - lb.loc[user]
    d = d * d
    d = d.mean(axis=1)
    d = np.sqrt(d)
    d = d.to_frame('distance')

    df = pd.concat([df, d], axis=1)
    df = df.drop(user, axis=0)
    df = df.sort_values(by='correlation', ascending=False)

    return df


def val_v_live_consistency(lb):
    "Live consistency versus validation consistency"
    cols = ['7/12', '8/12', '9/12', '10/12', '11/12', '12/12']
    df = pd.DataFrame(columns=cols)
    rounds = np.sort(lb['round'].unique())
    nan = np.nan
    for r in rounds:
        d = lb[lb['round'] == r]
        if not d['resolved'].any():
            consis = [nan, nan, nan, nan, nan, nan]
        else:
            d.insert(0, 'pass', d['live'] < LOGLOSS_BENCHMARK)
            d = d[['consis', 'pass']]
            d = d.groupby('consis').mean()
            d = d[d.index > 55]
            consis = d.T.values.tolist()[0]
            if len(consis) != len(cols):
                # TODO handle missing data
                consis = [np.nan] * len(cols)
        df.loc[r] = consis
    df.loc['mean'] = df.mean()
    return df
