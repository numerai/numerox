import numpy as np
import pandas as pd

import numerox as nx
from numerox.metrics import LOGLOSS_BENCHMARK
from numerox.numerai import calc_cutoff


class Report(object):

    def __init__(self):
        self.lb = nx.Leaderboard()

    def payout(self, round1, round2):
        "NMR and USD payouts per round"
        cols = ['staked_nmr', 'staked_above_cutoff', 'burned_nmr',
                'nmr_payout', 'usd_payout']
        df = pd.DataFrame(columns=cols)
        lb = self.lb[round1:round2]
        rounds = lb['round'].unique()
        for r in rounds:
            d = lb[lb['round'] == r]
            if r > 112:
                nmr_cut = 0
                for t in nx.tournament_all(as_str=False):
                    dt = d[d.tournament == t]
                    cutoff, ignore = calc_cutoff(dt)
                    nmr_cut += dt[dt.c >= cutoff].sum()['s']
            else:
                nmr_cut = np.nan
            ds = d.sum()
            pay = [ds['s'], nmr_cut, ds['nmr_burn'], ds['nmr_stake'],
                   ds['usd_stake']]
            df.loc[r] = pay
        fraction = df['burned_nmr'] / df['staked_above_cutoff']
        df.insert(3, 'fraction_burned', fraction)
        df.loc['mean'] = df.mean()
        df = df.round(2)
        return df

    def cutoff(self, round1, round2, cache_current_round=True):
        "Independent calculation of confidence cutoff"
        cols = nx.tournament_all(as_str=True)
        df = pd.DataFrame(columns=cols)
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
        rounds = lb['round'].unique()
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

    def pass_rate(self, round1, round2):
        "Fraction of users who beat benchmark in each round"
        cols = ['all', 'stakers', 'nonstakers', 'above_cutoff', 'below_cutoff']
        df = pd.DataFrame(columns=cols)
        lb = self.lb[round1:round2]
        rounds = lb['round'].unique()
        for r in rounds:
            d = lb[lb['round'] == r]
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
                pr_above = 1.0 * pabove / nabove
                pr_below = 1.0 * pbelow / nbelow
            else:
                pr_above = np.nan
                pr_below = np.nan
            df.loc[r] = [pr_all, pr_stakers, pr_nonstakers, pr_above, pr_below]
        df.loc['mean'] = df.mean()
        return df

    def out_of_five(self, round1, round2):
        "Fraction of users that get, e.g., 3/5 in a round"
        cols = ['N', '0/5', '1/5', '2/5', '3/5', '4/5', '5/5', 'mean/5']
        df = pd.DataFrame(columns=cols)
        lb = self.lb[round1:round2]
        rounds = lb['round'].unique()
        nan = np.nan
        for r in rounds:
            d = lb[lb['round'] == r]
            if not d['resolved'].any():
                fraction = [0, nan, nan, nan, nan, nan, nan, nan]
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
                mean = np.dot(fraction, np.array([0, 1, 2, 3, 4, 5]))
                fraction = fraction.tolist()
                fraction.insert(0, count)
                fraction.insert(7, mean)
            df.loc[r] = fraction
        df.loc['mean'] = df.mean()
        df['N'] = df['N'].astype(int)
        return df

    def five_star_club(self, round1):
        "Users who beat benchmark in all 5 tournaments sorted by mean logloss"
        lb = self.lb[round1]
        lb['pass'] = lb['live'] < LOGLOSS_BENCHMARK
        s = lb.groupby('user').sum()
        df = s[s['pass'] == 5]
        df = df[['live']] / 5
        df.columns = ['mean_logloss']
        df = df.sort_values('mean_logloss')
        return df

    def logloss_correlation(self, round1, round2):
        """
        Mean correlation of a users live logloss to all other users.

        Only those that have submitted in every tournament are considered.

        """
        lb = self.lb[round1:round2]
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

    def friends(self, user, round1, round2):
        """
        Correlation of live logloss of each user to a given `user` and
        Euclidean distance.

        Only those that have submitted in every tournament are considered. So
        given `user` must have submitted in every tournament.

        """

        lb = self.lb[round1:round2]
        lb.insert(0, 'rt', lb['round'] * 10 + lb['tournament'])
        lb = lb[['user', 'rt', 'live']]
        lb = lb.set_index('user')
        lb = lb.pivot(columns='rt', values='live')
        lb = lb.dropna()

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

    def val_v_live_consistency(self, round1, round2):
        "Live consistency versus validation consistency"
        cols = ['7/12', '8/12', '9/12', '10/12', '11/12', '12/12']
        df = pd.DataFrame(columns=cols)
        lb = self.lb[round1:round2]
        rounds = lb['round'].unique()
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
            df.loc[r] = consis
        df.loc['mean'] = df.mean()
        return df
