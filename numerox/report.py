import numpy as np
import pandas as pd

import numerox as nx
from numerox.metrics import LOGLOSS_BENCHMARK


class Report(object):

    def __init__(self):
        self.lb = nx.Leaderboard()

    def out_of_five(self, round1, round2):
        lb = self.lb[round1:round2]
        rounds = lb['round'].unique()
        cols = ['N', '0/5', '1/5', '2/5', '3/5', '4/5', '5/5', 'mean/5']
        df = pd.DataFrame(columns=cols)
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
        df['N'] = df['N'].astype(int)
        return df
