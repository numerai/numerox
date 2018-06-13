import decimal

import numpy as np
import pandas as pd
from numerapi import NumerAPI

import numerox as nx

DEFAULT_FIRST_ROUND = 51


class Leaderboard(object):

    def __init__(self, tournament=1, verbose=False):
        self.tournament = nx.tournament_int(tournament)
        self.verbose = verbose
        self.df = None
        self.unresolved_rounds = []
        self.current_round = None

    def __getitem__(self, index):
        "Index by round number, list (or tuple), or slice"
        if isinstance(index, slice):
            if index.step is not None:
                raise ValueError("slice step size must be 1")
            r1, r2 = self.rounds_to_ints(index.start, index.stop)
            self.get_range(r1, r2)
            idx1 = self.df['round'] >= r1
            idx2 = self.df['round'] <= r2
            idx = idx1 & idx2
            df = self.df[idx]
        elif nx.isint(index):
            self.get_round(index)
            df = self.df[self.df['round'] == index]
        elif isinstance(index, list) or isinstance(index, tuple):
            self.get_list(index)
            idx = self.df['round'].isin(index)
            df = self.df[idx]
        else:
            raise IndexError("indexing method not supported")
        return df

    def rounds_to_ints(self, round1, round2):
        "Convert `round1` and `round2`, which might be None, to integers"
        if round1 is None:
            round1 = DEFAULT_FIRST_ROUND
        if round2 is None:
            if self.current_round is None:
                self.current_round = get_current_round_number(self.tournament)
            round2 = self.current_round
        return round1, round2

    def get_range(self, round1, round2):
        "Download leaderboards if not yet downloaded for round range provided"
        r1, r2 = self.rounds_to_ints(round1, round2)
        rounds = list(range(r1, r2 + 1))
        self.get_list(rounds)

    def get_list(self, round_list):
        "Download leaderboards, if missing, for list of rounds provided"
        for n in round_list:
            self.get_round(n)

    def get_round(self, round_number):
        "Doenload, if missing, a single round"
        r = round_number
        if r not in self:
            if r in self.unresolved_rounds:
                if self.verbose:
                    print("R{:d} already download and is unresolved".format(r))
            else:
                if self.verbose:
                    print("downloading R{:d}".format(r))
                b = download_leaderboard(r, self.tournament)
                if self.is_resolved(b):
                    if self.verbose:
                        print("R{:d} is resolved".format(r))
                    self.df = pd.concat([self.df, b])
                else:
                    if self.verbose:
                        print("R{:d} is unresolved".format(r))
                    self.unresolved_rounds.append(r)
        else:
            if self.verbose:
                print("R{:d} already downloaded and is resolved".format(r))

    @property
    def resolved_rounds(self):
        "List of resolved rounds"
        if self.df is None:
            return []
        return sorted(self.df['round'].unique().tolist())

    def is_resolved(self, df_single_round):
        "Has the round been resolved? True or False. Pass in a single round"
        df = df_single_round
        total = df.iloc[:, 2:-3].abs().sum().sum()
        if total == 0:
            return False
        return True

    def __contains__(self, round_number):
        "Has `round_number` already been downloaded? True or False"
        if self.df is None:
            return False
        return round_number in self.df['round'].values


def download_leaderboard(round_number=None, tournament=1):
    """
    Download leaderboard for specified tournament and round.

    Default is to download current round.
    """
    tournament = nx.tournament_int(tournament)
    if round_number is None:
        napi = NumerAPI(verbosity='warn')
        num = napi.get_current_round(tournament=tournament)
    else:
        num = round_number
    df = download_raw_leaderboard(round_number=num, tournament=tournament)
    df = raw_leaderboard_to_df(df, num)
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
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan]
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
        if user['stake']['value'] is not None:
            x[8] = float(user['stake']['value'])
            x[9] = decimal.Decimal(user['stake']['confidence'])
            x[10] = float(user['stake']['soc'])
        leaderboard.append(x)
    columns = ['round', 'user', 'usd_main', 'usd_stake', 'nmr_main',
               'nmr_stake', 'nmr_burn', 'live', 's', 'c', 'soc']
    df = pd.DataFrame(data=leaderboard, columns=columns)
    return df


def get_current_round_number(tournament):
    "Current round number as an integer."
    tournament = nx.tournament_int(tournament)
    napi = NumerAPI(verbosity='warn')
    cr = napi.get_current_round(tournament=tournament)
    return cr
