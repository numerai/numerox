import numpy as np
import pandas as pd

import numerox as nx
from numerox.metrics import LOGLOSS_BENCHMARK


class Report(object):

    def __init__(self):
        self.lb = nx.Leaderboard()

    def out_of_five(self, round1, round2):
        "Fraction of users that get, e.g., 3/5 in a round"
        cols = ['N', '0/5', '1/5', '2/5', '3/5', '4/5', '5/5', 'mean/5']
        df = pd.DataFrame(columns=cols)
        lb = self.lb[round1:round2]
        rounds = lb['round'].unique()
        for r in rounds:
            d = lb[lb['round'] == r]
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

    def val_v_live_consistency(self, round1, round2):
        "Live consistency versus validation consistency"
        cols = ['7/12', '8/12', '9/12', '10/12', '11/12', '12/12']
        df = pd.DataFrame(columns=cols)
        lb = self.lb[round1:round2]
        rounds = lb['round'].unique()
        for r in rounds:
            d = lb[lb['round'] == r]
            d.insert(0, 'pass', d['live'] < LOGLOSS_BENCHMARK)
            d = d[['consis', 'pass']]
            d = d.groupby('consis').mean()
            d = d[d.index > 55]
            consis = d.T.values.tolist()[0]
            df.loc[r] = consis
        df.loc['mean'] = df.mean()
        return df
