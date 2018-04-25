import os
import time
import tempfile
import datetime
import decimal

import pandas as pd
from numerapi import NumerAPI
from numerapi.utils import download_file

import numerox as nx
from numerox.prediction import CONSISTENCY_GTE


# ---------------------------------------------------------------------------
# download dataset

def download(filename, tournament=1, verbose=False):
    "Download the current Numerai dataset; overwrites if file exists"
    if verbose:
        print("Download dataset {}".format(filename))
    napi = NumerAPI()
    url = napi.get_dataset_url(tournament=tournament)
    filename = os.path.expanduser(filename)  # expand ~/tmp to /home/...
    download_file(url, filename)


def download_data_object(tournament=1, verbose=False):
    "Used by numerox to avoid hard coding paths; probably not useful to users"
    with tempfile.NamedTemporaryFile() as temp:
        download(temp.name, tournament=tournament, verbose=verbose)
        data = nx.load_zip(temp.name)
    return data


# ---------------------------------------------------------------------------
# upload submission

def upload(filename, public_id, secret_key, tournament=1, block=True):
    """
    Upload tournament submission (csv file) to Numerai.

    If block is True (default) then the scope of your token must be both
    upload_submission and read_submission_info. If block is False then only
    upload_submission is needed.
    """
    napi = NumerAPI(public_id=public_id, secret_key=secret_key,
                    verbosity='warning')
    upload_id = napi.upload_predictions(filename, tournament=tournament)
    if block:
        status = status_block(upload_id, public_id, secret_key)
    else:
        status = upload_status(upload_id, public_id, secret_key)
    return upload_id, status


def upload_status(upload_id, public_id, secret_key):
    "Dictionary containing the status of upload"
    napi = NumerAPI(public_id=public_id, secret_key=secret_key,
                    verbosity='warning')
    status_raw = napi.submission_status(upload_id)
    status = {}
    for key, value in status_raw.items():
        if isinstance(value, dict):
            value = value['value']
        status[key] = value
    return status


def status_block(upload_id, public_id, secret_key, verbose=True):
    """
    Block until status completes; then return status dictionary.

    The scope of your token must must include read_submission_info.
    """
    t0 = time.time()
    if verbose:
        print("metric                  value   minutes")
    seen = []
    fmt_f = "{:<19} {:>9.4f}   {:<.4f}"
    fmt_b = "{:<19} {:>9}   {:<.4f}"
    while True:
        status = upload_status(upload_id, public_id, secret_key)
        t = time.time()
        for key, value in status.items():
            if value is not None and key not in seen:
                seen.append(key)
                minutes = (t - t0) / 60
                if verbose:
                    if key in ('originality', 'concordance'):
                        print(fmt_b.format(key,  str(value), minutes))
                    else:
                        print(fmt_f.format(key,  value, minutes))
        if len(status) == len(seen):
            break
        seconds = min(5 + int((t - t0) / 100.0), 30)
        time.sleep(seconds)
    if verbose:
        t = time.time()
        minutes = (t - t0) / 60
        iscc = is_controlling_capital(status)
        print(fmt_b.format('controlling capital', str(iscc), minutes))
    return status


def is_controlling_capital(status):
    "Did you get controlling capital? Pending status returns False."
    if None in status.values():
        return False
    iscc = status['consistency'] >= 100 * CONSISTENCY_GTE
    iscc = iscc and status['originality'] and status['concordance']
    return iscc


# ---------------------------------------------------------------------------
# stakes

def show_stakes(round_number=None, tournament=1, sort_by='prize pool',
                mark_user=None, use_integers=True):
    "Display info on staking; cumsum is dollars above you"
    df, c_zero_users = get_stakes(round_number, tournament, sort_by, mark_user,
                                  use_integers)
    with pd.option_context('display.colheader_justify', 'left'):
        print(df.to_string(index=True))
    if len(c_zero_users) > 0:
        c_zero_users = ','.join(c_zero_users)
        print('C=0: {}'.format(c_zero_users))


def get_stakes(round_number=None, tournament=1, sort_by='prize pool',
               mark_user=None, use_integers=True):
    """
    Download stakes, modify it to make it more useful, return as dataframe.

    cumsum is dollars ABOVE you.
    """

    # get raw stakes
    napi = NumerAPI()
    query = '''
        query stakes($number: Int!
                     $tournament: Int!){
          rounds(number: $number
                 tournament: $tournament){
            leaderboard {
              username
              stake {
                insertedAt
                soc
                confidence
                value
              }
            }
          }
        }
    '''
    if round_number is None:
        round_number = 0
    elif round_number < 61:
        raise ValueError('First staking was in round 61')
    arguments = {'number': round_number, 'tournament': tournament}
    stakes = napi.raw_query(query, arguments)

    # massage raw stakes
    stakes = stakes['data']['rounds'][0]['leaderboard']
    stakes2 = []
    strptime = datetime.datetime.strptime
    now = datetime.datetime.utcnow()
    secperday = 24 * 60 * 60
    micperday = 1000000 * secperday
    for s in stakes:
        user = s['username']
        s = s['stake']
        if s['value'] is not None:
            s2 = {}
            s2['user'] = user
            s2['s'] = float(s['value'])
            s2['c'] = decimal.Decimal(s['confidence'])
            s2['soc'] = float(s['soc'])
            t = now - strptime(s['insertedAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
            d = t.days
            d += 1.0 * t.seconds / secperday
            d += 1.0 * t.microseconds / micperday
            s2['days'] = d
            stakes2.append(s2)
    stakes = stakes2

    # jam stakes into a dataframe
    stakes = pd.DataFrame(stakes)
    stakes = stakes[['days', 's', 'soc', 'c', 'user']]

    # remove C=0 stakers
    c_zero_users = stakes.user[stakes.c == 0].tolist()
    stakes = stakes[stakes.c != 0]

    # index by user
    stakes = stakes.set_index('user')

    # sort in prize pool order; add s/c cumsum
    stakes = stakes.sort_values(['c', 'days'], axis=0,
                                ascending=[False, False])
    cumsum = stakes.soc.cumsum(axis=0) - stakes.soc  # dollars above you
    stakes.insert(3, 'cumsum', cumsum)

    # other sorting
    if sort_by == 'prize pool':
        pass
    elif sort_by == 'c':
        stakes = stakes.sort_values(['c'], ascending=[False])
    elif sort_by == 's':
        stakes = stakes.sort_values(['s'], ascending=[False])
    elif sort_by == 'soc':
        stakes = stakes.sort_values(['soc'], ascending=[False])
    elif sort_by == 'days':
        stakes = stakes.sort_values(['days'], ascending=[True])
    elif sort_by == 'user':
        stakes = stakes.sort_values(['user'], ascending=[True])
    else:
        raise ValueError("`sort_by` key not recognized")

    # round stakes
    if use_integers:
        stakes['days'] = stakes['days'].round(4)
        stakes['s'] = stakes['s'].astype(int)
        stakes['soc'] = stakes['soc'].astype(int)
        stakes['cumsum'] = stakes['cumsum'].astype(int)

    # mark user
    if mark_user is not None and mark_user in stakes.index:
        stakes['mark'] = ''
        me = stakes.loc[mark_user]['days']
        idx = stakes.days < me
        stakes.loc[idx, 'mark'] = 'new'
        stakes.loc[mark_user, 'mark'] = '<<<<'

    return stakes, c_zero_users


# ---------------------------------------------------------------------------
# utilities


def round_resolution_date(tournament=1):
    "The date each round was resolved as a Dataframe."
    napi = NumerAPI(verbosity='warn')
    dates = napi.get_competitions(tournament=tournament)
    dates = pd.DataFrame(dates)[['number', 'resolveTime']]
    rename_map = {'number': 'round', 'resolveTime': 'date'}
    dates = dates.rename(rename_map, axis=1)
    date = dates['date'].tolist()
    date = [d.date() for d in date]
    dates['date'] = date
    dates = dates.set_index('round')
    dates = dates.sort_index()
    return dates


def year_to_round_range(year, tournament=1):
    "First and last (or latest) round number resolved in given year."
    if year < 2016:
        raise ValueError("`year` must be at least 2016")
    year_now = datetime.datetime.now().year
    if year > year_now:
        raise ValueError("`year` cannot be greater than {}".format(year_now))
    # numerai api incorrectly gives R32 as the first in 2017, so skip api
    # for 2016 and 2017; faster too
    if year == 2016:
        round1 = 1
        round2 = 31
    elif year == 2017:
        round1 = 32
        round2 = 83
    else:
        date = round_resolution_date(tournament=tournament)
        dates = date['date'].tolist()
        years = [d.year for d in dates]
        date['year'] = years
        date = date[date['year'] == year]
        round1 = date.index.min()
        round2 = date.index.max()
    return round1, round2
