import requests
import os

API_TOURNAMENT_URL = 'https://api-tournament.numer.ai'


class ApiTournament:

    def __init__(self, public_id=None, secret_key=None):
        if public_id and secret_key:
            self.token = (public_id, secret_key)
        elif not public_id and not secret_key:
            self.token = None
        else:
            print("You supply both a public id and a secret key.")
            self.token = None

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

    def upload_submission(self, full_filename):
        filename = os.path.basename(full_filename)
        if not self.token:
            print("You supply an API token to upload.")
        auth_query = \
            '''
            query($filename: String!) {
                submission_upload_auth(filename: $filename) {
                    filename
                    url
                }
            }
            '''
        submission_resp = self.call(auth_query, {'filename': filename})
        submission_auth = submission_resp['data']['submission_upload_auth']
        file_object = open(full_filename, 'rb').read()
        requests.put(submission_auth['url'], data=file_object)
        create_query = \
            '''
            mutation($filename: String!) {
                create_submission(filename: $filename) {
                    id
                }
            }
            '''
        create = self.call(create_query,
                           {'filename': submission_auth['filename']})
        return create['data']['create_submission']['id']
