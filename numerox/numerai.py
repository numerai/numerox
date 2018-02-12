import os
import time
import tempfile
import datetime

import pandas as pd
from numerapi import NumerAPI
from numerapi.utils import download_file

import numerox as nx


# ---------------------------------------------------------------------------
# download dataset

def download(filename, verbose=False):
    "Download the current Numerai dataset; overwrites if file exists"
    if verbose:
        print("Download dataset {}".format(filename))
    napi = NumerAPI()
    url = napi.get_dataset_url()
    filename = os.path.expanduser(filename)  # expand ~/tmp to /home/...
    download_file(url, filename)


def download_data_object(verbose=False):
    "Used by numerox to avoid hard coding paths; probably not useful to users"
    with tempfile.NamedTemporaryFile() as temp:
        download(temp.name, verbose=verbose)
        data = nx.load_zip(temp.name)
    return data


# ---------------------------------------------------------------------------
# upload submission

def upload(filename, public_id, secret_key, block=True):
    """
    Upload tournament submission (csv file) to Numerai.

    If block is True (default) then the scope of your token must be both
    upload_submission and read_submission_info. If block is False then only
    upload_submission is needed.
    """
    napi = NumerAPI(public_id=public_id, secret_key=secret_key,
                    verbosity='warning')
    upload_id = napi.upload_predictions(filename)
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
    iscc = status['consistency'] >= 75 and status['originality']
    iscc = iscc and status['concordance']
    return iscc


# ---------------------------------------------------------------------------
# stakes

def show_stakes(tournament_number=None, sort_by='prize pool'):
    "Display info on staking; cumsum is dollars above you"
    df, c_zero_users = get_stakes(tournament_number=tournament_number)
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


def get_stakes(tournament_number=None):
    """
    Download stakes, modify it to make it more useful, return as dataframe.

    cumsum is dollars ABOVE you.
    """

    # get raw stakes
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
    if tournament_number is None:
        tournament_number = 0
    elif tournament_number < 61:
        raise ValueError('First staking was in tournament 61')
    arguments = {'number': tournament_number}
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


# ---------------------------------------------------------------------------
# earnings


def ten99(user, year=2017):
    "Generate unoffical 1099-MISC report"
    r0, r1 = year_to_tournament_range(year)
    df = download_earnings(r0, r1)
    df = df[df.user == user]
    df = df[['tournament', 'usd_main', 'usd_stake', 'nmr_main']]
    df = df.set_index('tournament')
    nmrprice = nx.nmr_resolution_price()
    price = []
    for n in df.index:
        if n < 58:
            # nmr not yet traded on bittrex
            p = 0
        else:
            p = nmrprice.loc[n]['usd']
        price.append(p)
    df['nmr_usd'] = price
    total = df['usd_main'].values + df['usd_stake'].values
    total = total + df['nmr_main'].values * df['nmr_usd'].values
    df['total'] = total
    earn = df['usd_main'] + df['nmr_main'] + df['usd_stake']
    df = df[earn != 0]  # remove burn only rounds
    date = tournament_resolution_date()
    date = date.loc[df.index]
    df.insert(0, 'date', date)
    return df


def top_stakers(tournament1=61, tournament2=None, ntop=20):
    "Earnings report of top stakers"
    price = nx.token_price_data(ticker='nmr')['price']
    df = download_earnings(tournament1, tournament2)
    t1 = df['tournament'].min()
    t2 = df['tournament'].max()
    fmt = "Top stake earners (R{} - R{}) at {:.2f} usd/nmr"
    print(fmt.format(t1, t2, price))
    df = df[['user', 'usd_stake', 'nmr_burn']]
    df = df.groupby('user').sum()
    df = df.rename({'usd_stake': 'earn_usd', 'nmr_burn': 'burn_nmr'}, axis=1)
    ratio = df['earn_usd'] / df['burn_nmr']
    ratio = ratio.fillna(0)
    df['earn/burn'] = ratio
    df['profit_usd'] = df['earn_usd'] - price * df['burn_nmr']
    df = df.sort_values('profit_usd', ascending=False)
    if ntop < 0:
        df = df[ntop:]
    else:
        df = df[:ntop]
    df = df.round()
    cols = ['earn_usd', 'burn_nmr', 'profit_usd']
    df[cols] = df[cols].astype(int)
    print(df)


def top_earners(tournament1, tournament2=None, ntop=20):
    "Report on top earners"
    price = nx.token_price_data(ticker='nmr')['price']
    df = download_earnings(tournament1, tournament2)
    t1 = df['tournament'].min()
    t2 = df['tournament'].max()
    fmt = "Top earners (R{} - R{}) at {:.2f} usd/nmr"
    print(fmt.format(t1, t2, price))
    df = df.drop('tournament', axis=1)
    df = df.groupby('user').sum()
    profit = df['usd_main'] + df['usd_stake']
    profit += price * (df['nmr_main'] - df['nmr_burn'])
    df['profit_usd'] = profit
    df = df.sort_values('profit_usd', ascending=False)
    if ntop < 0:
        df = df[ntop:]
    else:
        df = df[:ntop]
    df = df.round()
    cols = ['usd_main', 'usd_stake', 'nmr_main', 'nmr_burn', 'profit_usd']
    df[cols] = df[cols].astype(int)
    print(df)


def download_earnings(tournament1=None, tournament2=None):
    "Download earnings for specified tournament range."
    napi = NumerAPI(verbosity='warn')
    if tournament1 is None and tournament2 is None:
        r0 = napi.get_current_round()
        r1 = r0
    elif tournament1 is None:
        r0 = napi.get_current_round()
        r1 = tournament2
    elif tournament2 is None:
        r0 = tournament1
        r1 = napi.get_current_round()
    else:
        r0 = tournament1
        r1 = tournament2
    for num in range(r0, r1 + 1):
        e = download_raw_earnings(tournament_number=num)
        e = raw_earnings_to_df(e, num)
        if num == r0:
            df = e
        else:
            df = pd.concat([df, e])
    return df


def download_raw_earnings(tournament_number=None):
    "Download earnings for given tournament number"
    query = '''
            query($number: Int!) {
                rounds(number: $number) {
                    leaderboard {
                        username
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
                        }
                        stakeResolution {
                          destroyed
                        }
                    }
                }
            }
    '''
    napi = NumerAPI(verbosity='warn')
    if tournament_number is None:
        tournament_number = napi.get_current_round()
    arguments = {'number': tournament_number}
    earnings = napi.raw_query(query, arguments)
    earnings = earnings['data']['rounds'][0]['leaderboard']
    return earnings


def raw_earnings_to_df(raw_earnings, tournament_number):
    "Keep non-zero earnings and convert to dataframe"
    earnings = []
    for user in raw_earnings:
        main = user['paymentGeneral']
        stake = user['paymentStaking']
        burn = user['stakeResolution']
        earned = main is not None or stake is not None
        burned = burn is not None and burn['destroyed']
        if earned or burned:
            x = [tournament_number, user['username'], 0.0, 0.0, 0.0, 0.0]
            if main is not None:
                x[2] = float(main['usdAmount'])
                if 'nmrAmount' in main:
                    x[4] = float(main['nmrAmount'])
            if stake is not None:
                x[3] = float(stake['usdAmount'])
            if burned:
                x[5] = float(user['stake']['value'])
            earnings.append(x)
    columns = ['tournament', 'user', 'usd_main', 'usd_stake', 'nmr_main',
               'nmr_burn']
    df = pd.DataFrame(data=earnings, columns=columns)
    return df


# ---------------------------------------------------------------------------
# utilities


def tournament_resolution_date():
    "The date each tournament was resolved as a Dataframe."
    napi = NumerAPI(verbosity='warn')
    dates = napi.get_competitions()
    dates = pd.DataFrame(dates)[['number', 'resolveTime']]
    rename_map = {'number': 'tournament', 'resolveTime': 'date'}
    dates = dates.rename(rename_map, axis=1)
    date = dates['date'].tolist()
    date = [d.date() for d in date]
    dates['date'] = date
    dates = dates.set_index('tournament')
    dates = dates.sort_index()
    return dates


def year_to_tournament_range(year):
    "First and last (or latest) tournament number resolved in given year."
    if year < 2016:
        raise ValueError("`year` must be at least 2016")
    year_now = datetime.datetime.now().year
    if year > year_now:
        raise ValueError("`year` cannot be greater than {}".format(year_now))
    # numerai api incorrectly gives R32 as the first in 2017, so skip api
    # for 2016 and 2017; faster too
    if year == 2016:
        tournament1 = 1
        tournament2 = 31
    elif year == 2017:
        tournament1 = 32
        tournament2 = 83
    else:
        date = tournament_resolution_date()
        dates = date['date'].tolist()
        years = [d.year for d in dates]
        date['year'] = years
        date = date[date['year'] == year]
        tournament1 = date.index.min()
        tournament2 = date.index.max()
    return tournament1, tournament2
