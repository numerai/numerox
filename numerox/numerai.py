import requests


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
