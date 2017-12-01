import requests
import tempfile
import datetime

import pandas as pd

import numerox as nx

API_TOURNAMENT_URL = 'https://api-tournament.numer.ai'


def download_dataset(saved_filename, verbose=False):
    "Download the current Numerai dataset"
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
    api = Numerai()
    query = "query {dataset}"
    url = api.call(query)['data']['dataset']
    return url


def download_data_object():
    "Used by numerox to avoid hard coding paths; probably not useful to users"
    with tempfile.NamedTemporaryFile() as temp:
        download_dataset(temp.name)
        data = nx.load_zip(temp.name)
    return data


def show_stakes(round_number=None):
    "Display info on staking"
    df, c_zero_users = get_stakes(round_number=round_number)
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
    "Download stakes, modify it to make it more useful, return as dataframe"

    # get raw stakes; eventually use numerapi for this block
    api = Numerai()
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
    stakes = api.call(query, arguments)  # ~92% of time spent on this line

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
    stakes.insert(3, 'cumsum', stakes.soc.cumsum(axis=0))

    return stakes, c_zero_users


class Numerai(object):

    def __init__(self, public_id=None, secret_key=None):
        if public_id and secret_key:
            self.token = (public_id, secret_key)
        elif not public_id and not secret_key:
            self.token = None
        else:
            print("You supply both a public id and a secret key.")
            self.token = None

    def has_token(self):
        if self.token is not None:
            return True
        return False

    def call(self, query, variables=None):
        body = {'query': query,
                'variables': variables}
        headers = {'Content-type': 'application/json',
                   'Accept': 'application/json'}
        if self.token:
            public_id, secret_key = self.token
            headers['Authorization'] = \
                'Token {}${}'.format(public_id, secret_key)
        r = requests.post(API_TOURNAMENT_URL, json=body, headers=headers)
        return r.json()
