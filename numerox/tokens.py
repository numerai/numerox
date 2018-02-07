import datetime
import requests


def nmr_at_addr(addr_str):
    url = 'https://api.etherscan.io/api?module=account&action=tokenbalance&'
    url += 'contractaddress=0x1776e1F26f98b1A5dF9cD347953a26dd3Cb46671&'
    url += 'address=%s'
    r = requests.get(url % addr_str)
    data = r.json()
    nmr = int(data['result']) / 1e18
    return nmr


def token_price_data(ticker='nmr'):
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
    price['time'] = datetime.datetime.fromtimestamp(int(data['last_updated']))
    return price
