import datetime
import requests

import pandas as pd

import numerox as nx


def nmr_at_addr(addr_str):
    "Number of NMR (float) at given address."
    url = 'https://api.etherscan.io/api?module=account&action=tokenbalance&'
    url += 'contractaddress=0x1776e1F26f98b1A5dF9cD347953a26dd3Cb46671&'
    url += 'address=%s'
    r = requests.get(url % addr_str)
    data = r.json()
    nmr = int(data['result']) / 1e18
    return nmr


def nmr_transactions(addr_str, map_known_exchanges=False):
    """
    NMR transactions (dataframe) to/from given address.

    The sign of the 'nmr' column gives the direction of the transaction. Plus
    (minus) means the balance of NMR increased (decreased).

    The 'address' column is the from address for incoming transactions and
    the to address for outgoing transactions.

    The sum of the 'nmr' column is not necessarily the NMR balance at the
    address because burns are not included. To find the balance use
    the function 'nmr_at_addr'.

    If there are more than 1000 transactions then only the first 1000 are
    returned.

    Non-Numeraire tokens are skipped.

    When `map_known_exchanges` is True an attempt will be made to map the
    addresses to known exchange names.
    """
    url = 'http://api.etherscan.io/api?module=account&action=tokentx&'
    url += 'address=%s'
    r = requests.get(url % addr_str)
    data = r.json()
    if data['status'] != '1':
        if data['message'] != 'No transactions found':
            raise IOError('Could not get nmr transactions')
    txs = data['result']
    d = []
    for tx in txs:
        if tx['tokenName'] != 'Numeraire':
            continue
        date = datetime.datetime.fromtimestamp(int(tx['timeStamp']))
        to = tx['to']
        if to == addr_str:
            mult = 1
            addr = tx['from']
        else:
            mult = -1
            addr = tx['to']
        nmr = mult * int(tx['value']) / 1e18
        d.append([date, nmr, addr])
    df = pd.DataFrame(data=d, columns=['date', 'nmr', 'address'])
    df = df.set_index('date')
    if map_known_exchanges:
        exmap = {'0xfbb1b73c4f0bda4f67dca266ce6ef42f520fbb98': 'bittrex',
                 '0x7e5f1d2f3176b0fbcf551efa59367419e2c31812': 'bittrex',
                 '0xe94b04a0fed112f3664e45adb2b8915693dd5ff3': 'bittrex_2',
                 '0x8d12a197cb00d4747a1fe03395095ce2a5cc6819': 'etherdelta',
                 '0xb68704549cab20de1fadad34d27a86ffabb04a6e': 'shapeshift',
                 '0x0e1e175a0c57ebec5c17a72517dd3236efbe282e': 'changelly',
                 '0x32a263fcd370ea00e92d694e10080f97d7ef52a2': 'yobit',
                 '0xe269e891a2ec8585a378882ffa531141205e92e9': '0x'}
        df = df.replace({'address': exmap})
    return df


def token_price_data(ticker='nmr'):
    "Most recent price (and return) data for given ticker; returns dictionary."
    tickers = {'nmr': 'numeraire',
               'btc': 'bitcoin',
               'eth': 'ethereum',
               'ltc': 'litecoin'}
    if ticker in tickers:
        ticker = tickers[ticker]
    url = 'https://api.coinmarketcap.com/v1/ticker/%s/' % ticker
    r = requests.get(url)
    data = r.json()[0]
    price = {}
    price['name'] = ticker
    price['price'] = float(data['price_usd'])
    price['ret1h'] = float(data['percent_change_1h']) / 100.0
    price['ret1d'] = float(data['percent_change_24h']) / 100.0
    price['ret7d'] = float(data['percent_change_7d']) / 100.0
    price['date'] = datetime.datetime.fromtimestamp(int(data['last_updated']))
    return price


def historical_price(ticker, one_per_day=False):
    "Historical prices as a dataframe with date as index"
    tickers = {'nmr': 'currencies/numeraire',
               'btc': 'currencies/bitcoin',
               'eth': 'currencies/ethereum',
               'ltc': 'currencies/litecoin',
               'mkt': 'global/marketcap-total'}
    url = 'https://graphs2.coinmarketcap.com/%s'
    r = requests.get(url % tickers[ticker])
    data = r.json()
    if ticker == 'mkt':
        data = data['market_cap_by_available_supply']
    else:
        data = data['price_usd']
    dates = []
    prices = []
    for date, price in data:
        d = datetime.datetime.fromtimestamp(date / 1e3)
        if one_per_day:
            d = d.date()
        dates.append(d)
        prices.append(price)
    if one_per_day:
        p = []
        d = []
        for i in range(len(prices) - 1):
            d1 = dates[i]
            d2 = dates[i+1]
            if d1 != d2:
                p.append(prices[i])
                d.append(d1)
        if dates[-1] != d[-1]:
            p.append(prices[-1])
            d.append(dates[-1])
        prices = p
        dates = d
    prices = pd.DataFrame(data=prices, columns=['usd'], index=dates)
    return prices


def nmr_resolution_price(tournament=1):
    "Price of NMR in USD and date versus round number as a dataframe."
    price = nx.historical_price('nmr', one_per_day=True)
    dates = nx.round_resolution_date(tournament=tournament)
    price = pd.merge(dates, price, how='inner', left_on='date',
                     right_index=True)
    return price
