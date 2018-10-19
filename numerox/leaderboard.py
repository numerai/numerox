import decimal

import numpy as np
import pandas as pd
from numerapi import NumerAPI

import numerox as nx

DEFAULT_FIRST_ROUND = 51


class Leaderboard(object):

    def __init__(self, df=None, verbose=False):
        self.verbose = verbose
        self.df = df

    def __getitem__(self, index):
        "Index by round number, list (or tuple), or slice"
        if isinstance(index, slice):
            if index.step is not None:
                raise ValueError("slice step size must be 1")
            r1, r2 = self.rounds_to_ints(index.start, index.stop)
            rs = list(range(r1, r2 + 1))
            ts = nx.tournament_all(as_str=False)
        elif nx.isint(index):
            rs = [index]
            ts = nx.tournament_all(as_str=False)
        elif isinstance(index, list):
            rs, ts = zip(*index)
            ts = [nx.tournament_int(i) for i in ts]
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("tuple index must have length 2")
            r, t = index
            if not nx.isint(r):
                raise IndexError("first element of tuple index must be int")
            if not (nx.isint(t) or nx.isstring(t)):
                msg = "second element of tuple index must be int or str"
                raise IndexError(msg)
            rs = [r]
            ts = [nx.tournament_int(t)]
        else:
            raise IndexError("indexing method not supported")
        self.gets(rs, ts)
        ridx = self.df['round'].isin(rs)
        tidx = self.df['tournament'].isin(ts)
        idx = ridx & tidx
        df = self.df[idx]
        return df

    def gets(self, rounds, tournaments):
        "Download, if missing, round/tournament pairs"
        for r in rounds:
            for t in tournaments:
                if r < 111 and t != 1 and t != 'bernie':
                    # 5 tourney format started in round 111
                    continue
                self.get(r, t)

    def get(self, round_number, tournament):
        "Download, if missing, a single round/tournament pair"
        r = round_number
        t = nx.tournament_int(tournament)
        if (r, t) not in self:
            if self.verbose:
                print("downloading ({:d}, {})".format(r, t))
            b = download_leaderboard(r, t)
            if self.verbose:
                if b['resolved'].all():
                    print("({:d}, {}) is resolved".format(r, t))
                else:
                    print("({:d}, {}) is not resolved".format(r, t))
            self.df = pd.concat([self.df, b])
        else:
            if self.verbose:
                print("({:d}, {}) already downloaded".format(r, t))

    def __contains__(self, round_tournament_tuple):
        "Has (round, torunament) tuple already been downloaded? True or False"
        if len(round_tournament_tuple) != 2:
            raise ValueError("`round_tournament_tuple` must have length 2")
        if self.df is None:
            return False
        r, t = round_tournament_tuple
        t = nx.tournament_int(t)
        idx = (self.df['round'] == r) & (self.df['tournament'] == t)
        if idx.sum() > 0:
            return True
        return False

    def rounds_to_ints(self, round1, round2):
        "Convert `round1` and `round2`, which might be None, to integers"
        if round1 is None:
            round1 = DEFAULT_FIRST_ROUND
        if round2 is None:
            if self.current_round is None:
                self.current_round = get_current_round_number()
            round2 = self.current_round
        return round1, round2


def download_leaderboard(round_number=None, tournament=1):
    """
    Download leaderboard for specified tournament and round.

    Default is to download current round.
    """
    tournament = nx.tournament_int(tournament)
    if round_number is None:
        napi = NumerAPI(verbosity='warn')
        num = napi.get_current_round()
    else:
        num = round_number
    df = download_raw_leaderboard(round_number=num, tournament=tournament)
    df = raw_leaderboard_to_df(df, num)
    df.insert(1, 'tournament', tournament)
    cols = ['usd_main', 'usd_stake', 'nmr_main', 'nmr_stake', 'nmr_burn']
    d = df[cols]
    total = d.abs().sum().sum()
    if total == 0:
        resolved = False
    else:
        resolved = True
    df.insert(2, 'resolved', resolved)
    return df


def download_raw_leaderboard(round_number=None, tournament=1):
    "Download leaderboard for given round number"
    tournament = nx.tournament_int(tournament)
    query = '''
            query($number: Int!
                  $tournament: Int!) {
                rounds(number: $number
                       tournament: $tournament) {
                    leaderboard {
                        username
                        LiveLogloss
                        ValidationLogloss
                        Consistency
                        paymentGeneral {
                          nmrAmount
                          usdAmount
                        }
                        paymentStaking {
                          nmrAmount
                          usdAmount
                        }
                        stake {
                          value
                          confidence
                          soc
                        }
                        stakeResolution {
                          destroyed
                        }
                    }
                }
            }
    '''
    napi = NumerAPI(verbosity='warn')
    if round_number is None:
        round_number = get_current_round_number(tournament)
    arguments = {'number': round_number, 'tournament': tournament}
    leaderboard = napi.raw_query(query, arguments)
    leaderboard = leaderboard['data']['rounds'][0]['leaderboard']
    return leaderboard


def raw_leaderboard_to_df(raw_leaderboard, round_number):
    "Keep non-zero leaderboard and convert to dataframe"
    leaderboard = []
    for user in raw_leaderboard:
        main = user['paymentGeneral']
        stake = user['paymentStaking']
        burn = user['stakeResolution']
        burned = burn is not None and burn['destroyed']
        x = [round_number, user['username'],
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, 0.0, np.nan, np.nan]
        if main is not None:
            x[2] = float(main['usdAmount'])
            if 'nmrAmount' in main:
                x[4] = float(main['nmrAmount'])
        if stake is not None:
            x[3] = float(stake['usdAmount'])
            x[5] = float(stake['nmrAmount'])
        if burned:
            x[6] = float(user['stake']['value'])
        live = user['LiveLogloss']
        if live is None:
            if round_number < 51:
                x[7] = np.nan
            elif round_number < 90:
                x[7] = 1
            else:
                x[7] = np.nan
        else:
            x[7] = float(user['LiveLogloss'])
        val = user['ValidationLogloss']
        if val is not None:
            x[8] = float(val)
        consis = user['Consistency']
        if consis is not None:
            x[9] = float(consis)
        if user['stake']['value'] is not None:
            x[10] = float(user['stake']['value'])
            x[11] = decimal.Decimal(user['stake']['confidence'])
            x[12] = float(user['stake']['soc'])
        leaderboard.append(x)
    columns = ['round', 'user', 'usd_main', 'usd_stake', 'nmr_main',
               'nmr_stake', 'nmr_burn', 'live', 'val', 'consis', 's', 'c',
               'soc']
    df = pd.DataFrame(data=leaderboard, columns=columns)
    return df


def get_current_round_number():
    "Current round number as an integer."
    napi = NumerAPI(verbosity='warn')
    cr = napi.get_current_round(tournament=1)
    return cr
