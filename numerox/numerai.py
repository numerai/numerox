import os
import time
import requests

API_TOURNAMENT_URL = 'https://api-tournament.numer.ai'


# ---------------------------------------------------------------------------
# high-level user functions

def download_dataset(saved_filename):
    "Download the current Numerai dataset"
    url = 'https://api.numer.ai/competitions/current/dataset'
    r = requests.get(url)
    if r.status_code != 200:
        msg = 'failed to download dataset (staus code {}))'
        raise IOError(msg.format(r.status_code))
    with open(saved_filename, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)


def upload_submission(filename, public_id, secret_key, verbose=True):
    """
    Upload tournament submission (csv file) to Numerai.

    If verbose is True (default) then the scope of your token must be both
    upload_submission and read_submission_info. If verbose is False then only
    upload_submission is needed.
    """
    sub = Submission(public_id, secret_key)
    sub.upload(filename)
    sub.status_block(verbose)


def submission_status(submission_id, public_id, secret_key):
    "display submission status"
    if submission_id is None:
        raise ValueError('`submission_id` cannot be None')
    api = Numerai(public_id, secret_key)
    if not api.has_token():
        raise ValueError("Must supply public_id, secret_key")
    query = \
        '''
        query submissions($submission_id: String!) {
          submissions(id: $submission_id) {
            originality {
              pending
              value
            }
            concordance {
              pending
              value
            }
            consistency
            validation_logloss
          }
        }
        '''
    variable = {'submission_id': submission_id}
    status_raw = api.call(query, variable)
    status_raw = status_raw['data']['submissions'][0]
    status = {}
    for key, value in status_raw.items():
        if value is None:
            value = None
        elif isinstance(value, dict):
            value = value['value']
        status[key] = value
    return status


class Submission(object):

    def __init__(self, public_id, secret_key):
        self.api = Numerai(public_id, secret_key)
        if not self.api.has_token():
            raise ValueError("Must supply public_id, secret_key")
        self.t0 = None
        self.submission_id = None
        self.status = None

    def upload(self, filename):
        self.t0 = time.time()
        auth_query = \
            '''
            query($filename: String!) {
                submission_upload_auth(filename: $filename) {
                    filename
                    url
                }
            }
            '''
        variable = {'filename': os.path.basename(filename)}
        submission_resp = self.api.call(auth_query, variable)
        submission_auth = submission_resp['data']['submission_upload_auth']
        file_object = open(filename, 'rb').read()
        requests.put(submission_auth['url'], data=file_object)
        create_query = \
            '''
            mutation($filename: String!) {
                create_submission(filename: $filename) {
                    id
                }
            }
            '''
        variables = {'filename': submission_auth['filename']}
        create = self.api.call(create_query, variables)
        self.submission_id = create['data']['create_submission']['id']
        return self.submission_id

    def status(self):
        "status dictionary"
        self.status = submission_status(self.submission_id, *self.api.token)
        return self.status

    def status_block(self, verbose=True):
        "block until until status completes; then return status_dict"
        seen = []
        fmt = "{:>10.6f}  {:<.4f}  {:<}"
        while True:
            status = submission_status(self.submission_id, *self.api.token)
            t = time.time()
            for key, value in status.items():
                if value is not None and key not in seen:
                    seen.append(key)
                    minutes = (t - self.t0) / 60
                    if verbose:
                        print(fmt.format(value, minutes, key))
            if len(status) == len(seen):
                break
            seconds = min(5 + int((t - self.t0) / 100.0), 30)
            time.sleep(seconds)
        self.status = status
        return self.status

    @property
    def upload_completed(self):
        if self.submission_id is None:
            return False
        return True

    @property
    def status_completed(self):
        return self._is_status_complete(self.status)

    def _is_status_complete(self, status):
        n_none = status.values().count(None)
        if n_none > 0:
            return False
        return True


# ---------------------------------------------------------------------------
# low-level numerai api functions

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
