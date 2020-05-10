import os
import time
import tempfile
import datetime

import pandas as pd
from numerapi import NumerAPI
from numerapi.utils import download_file

import numerox as nx
from numerox.util import flatten_dict
from numerox.prediction import CONSISTENCY_GTE

NMR_PRIZE_POOL = 2000

# ---------------------------------------------------------------------------
# download dataset


def download(filename,
             load=True,
             n_tries=100,
             sleep_seconds=300,
             verbose=False,
             include_train=True,
             single_precision=True):
    """
    Download current Numerai dataset; overwrites if file exists.

    If `load` is True (default) then return data object; otherwise return
    None. It will include train data if `include_train` is True as well.

    Set `single_precision` to True in order to load data in float32 precision.

    If download fails then retry download `n_tries` times, pausing
    `sleep_seconds` between each try.

    Unlike nx.download() this function loads and returns the data object.
    """
    # line below expands e.g. ~/tmp to /home/me/tmp...
    filename = os.path.expanduser(filename)
    count = 0
    while count < n_tries:
        try:
            if verbose:
                print("Download dataset {}".format(filename))
            napi = NumerAPI()
            url = napi.get_dataset_url(tournament=8)
            download_file(url, filename)
            break
        except:  # noqa
            print('download failed')
            time.sleep(sleep_seconds)
        count += 1
    if load:
        data = nx.load_zip(filename,
                           verbose=verbose,
                           include_train=include_train,
                           single_precision=single_precision)
    else:
        data = None
    return data


def download_data_object(verbose=False):
    "Used by numerox to avoid hard coding paths; probably not useful to users"
    with tempfile.NamedTemporaryFile() as temp:
        download(temp.name, verbose=verbose)
        data = nx.load_zip(temp.name)
    return data


# ---------------------------------------------------------------------------
# upload submission


def upload(filename,
           tournament,
           public_id,
           secret_key,
           block=True,
           n_tries=100,
           sleep_seconds=60,
           verbose=False,
           model_id=None):
    """
    Upload tournament submission (csv file) to Numerai.

    Accounts with multiple models must specify model_id

    If upload fails then retry upload `n_tries` times, pausing `sleep_seconds`
    between each try.

    If block is True (default) then the scope of your token must be both
    upload_submission and read_submission_info. If block is False then only
    upload_submission is needed.

    """
    tournament = nx.tournament_int(tournament)
    count = 0
    napi = NumerAPI(public_id=public_id,
                    secret_key=secret_key,
                    verbosity='warning')
    models = napi.get_models()
    if len(models) > 1 and model_id is None:
        raise Exception(
            f"Account has multiple models - you must specify model_id from {models}"
        )
    elif model_id and model_id not in models.values():
        raise Exception(
            f"Specified model_id {model_id} not found in account models {models}"
        )

    while count < n_tries:
        try:
            upload_id = napi.upload_predictions(filename,
                                                tournament=tournament,
                                                model_id=model_id)
            if block:
                status = status_block(upload_id, public_id, secret_key)
            else:
                status = upload_status(upload_id, public_id, secret_key)
            break

        except:  # noqa
            print('upload failed')
            time.sleep(sleep_seconds)
        count += 1

    else:
        raise Exception('Upload failed after reaching max retries')

    return upload_id, status


def upload_status(upload_id, public_id, secret_key):
    "Dictionary containing the status of upload"
    napi = NumerAPI(public_id=public_id,
                    secret_key=secret_key,
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
                        print(fmt_b.format(key, str(value), minutes))
                    else:
                        print(fmt_f.format(key, value, minutes))
        if len(status) == len(seen):
            break
        seconds = min(5 + int((t - t0) / 100.0), 30)
        time.sleep(seconds)
    if verbose:
        t = time.time()
        minutes = (t - t0) / 60
        iss = is_stakeable(status)
        print(fmt_b.format('stakeable', str(iss), minutes))
    return status


def is_stakeable(status):
    "Is sumission stakeable? Pending status returns False."
    if None in status.values():
        return False
    iss = status['consistency'] >= 100 * CONSISTENCY_GTE
    iss = iss and status['concordance']
    return iss


# ---------------------------------------------------------------------------
# utilities


def round_dates():
    "The dates each round was opened and resolved as a Dataframe."
    napi = NumerAPI(verbosity='warn')
    dates = napi.get_competitions(tournament=1)
    dates = pd.DataFrame(dates)[['number', 'openTime', 'resolveTime']]
    rename_map = {
        'number': 'round',
        'openTime': 'open',
        'resolveTime': 'resolve'
    }
    dates = dates.rename(rename_map, axis=1)
    for item in ('open', 'resolve'):
        date = dates[item].tolist()
        date = [d.date() for d in date]
        dates[item] = date
    dates = dates.set_index('round')
    dates = dates.sort_index()
    return dates


def year_to_round_range(year):
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
        date = round_dates()
        date = date.drop('open', axis=1)
        dates = date['resolve'].tolist()
        years = [d.year for d in dates]
        date['year'] = years
        date = date[date['year'] == year]
        round1 = date.index.min()
        round2 = date.index.max()
    return round1, round2


def get_user_names():
    "A list containing all Numerai users, past and present."
    q = '''
        query {
            rankings(limit:100000, offset:0)
                {
                    username
                }
        }
    '''
    napi = NumerAPI()
    users = napi.raw_query(q)
    users = [x['username'] for x in users['data']['rankings']]
    return users


def get_user_activities(user):
    "Activity of `user` across all rounds and tournaments as dataframe"
    napi = NumerAPI()
    data = []
    for number, name in nx.tournament_iter():
        data += napi.get_user_activities(user, number)
    flat = [flatten_dict(d) for d in data]
    df = pd.DataFrame.from_dict(flat)
    return df
