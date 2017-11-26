import requests
import tempfile

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
