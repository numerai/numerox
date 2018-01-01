import os
import requests
import tempfile
import datetime

import pandas as pd
from numerapi.numerapi import NumerAPI

import numerox as nx


def download_dataset(saved_filename, overwrite=False, verbose=False):
    "Download the current Numerai dataset; by default raise if file exists"
    if not overwrite and os.path.exists(saved_filename):
        msg = "`saved_filename` already exists: {}"
        raise ValueError(msg.format(saved_filename))
    if verbose:
        print("Download dataset {}".format(saved_filename))
    url = dataset_url()
    r = requests.get(url)
    if r.status_code != 200:
        msg = 'failed to download dataset (staus code {}))'
        raise IOError(msg.format(r.status_code))
    with open(saved_filename, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)


def dataset_url():
    "URL of current Numerai dataset"
    napi = NumerAPI()
    query = "query {dataset}"
    url = napi.raw_query(query)['data']['dataset']
    return url


def download_data_object(verbose=False):
    "Used by numerox to avoid hard coding paths; probably not useful to users"
    with tempfile.NamedTemporaryFile() as temp:
        download_dataset(temp.name, verbose=verbose)
        data = nx.load_zip(temp.name)
    return data


def show_stakes(round_number=None, sort_by='prize pool'):
    "Display info on staking; cumsum is dollars above you"
    df, c_zero_users = get_stakes(round_number=round_number)
    if sort_by == 'prize pool':
        pass
    elif sort_by == 'c':
        df = df.sort_values(['c'], ascending=[False])
    elif sort_by == 's':
        df = df.sort_values(['s'], ascending=[False])
    elif sort_by == 'soc':
        df = df.sort_values(['soc'], ascending=[False])
    elif sort_by == 'days':
        df = df.sort_values(['days'], ascending=[True])
    elif sort_by == 'user':
        df = df.sort_values(['user'], ascending=[True])
    else:
        raise ValueError("`sort_by` key not recognized")
    df['days'] = df['days'].round(4)
    df['s'] = df['s'].astype(int)
    df['soc'] = df['soc'].astype(int)
    df['cumsum'] = df['cumsum'].astype(int)
    with pd.option_context('display.colheader_justify', 'left'):
        print(df.to_string(index=False))
    if len(c_zero_users) > 0:
        c_zero_users = ','.join(c_zero_users)
        print('C=0: {}'.format(c_zero_users))


def get_stakes(round_number=None):
    """
    Download stakes, modify it to make it more useful, return as dataframe.

    cumsum is dollars ABOVE you.
    """

    # get raw stakes; eventually use numerapi for this block
    napi = NumerAPI()
    query = '''
        query stakes($number: Int!){
          rounds(number: $number){
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
    arguments = {'number': round_number}
    # ~92% of time spent on the following line
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
            s2['c'] = float(s['confidence'])
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

    # sort in prize pool order; add s/c cumsum
    stakes = stakes.sort_values(['c', 'days'], axis=0,
                                ascending=[False, False])
    cumsum = stakes.soc.cumsum(axis=0) - stakes.soc  # dollars above you
    stakes.insert(3, 'cumsum', cumsum)

    return stakes, c_zero_users
