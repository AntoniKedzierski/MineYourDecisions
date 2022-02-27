import json

import pandas as pd
import requests
import io

from core.data_types.time_series import TimeSeries
from core.data_types.time_series import MultiTimeSeries


class Connector:
    @staticmethod
    def get_series() -> TimeSeries:
        pass


class AlphaVantageConnector(Connector):
    api_key = '9PFUHO5D9B0B6S4Q'

    @staticmethod
    def get_stock(function, symbol, outputsize='compact') -> MultiTimeSeries:
        url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={AlphaVantageConnector.api_key}&outputsize={outputsize}&datatype=csv'
        response = requests.get(url)

        if response.status_code == 400:
            raise ValueError(f'{symbol} for {function} does not exists at the endpoint.')
        if response.status_code == 401:
            raise ConnectionRefusedError(f'An user with api key {AlphaVantageConnector.api_key} has no permission to perform such query.')
        if response.status_code == 404:
            raise ConnectionError(f'Server not found. Path: {url}')
        if response.status_code == 408:
            raise ConnectionResetError('Timeout!')
        if response.status_code == 500:
            raise ConnectionAbortedError('Internal server error!')

        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d', errors='coerce')
        ts = MultiTimeSeries(df)
        return ts

    @staticmethod
    def search_stock(keyword) -> pd.DataFrame:
        url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={keyword}&apikey={AlphaVantageConnector.api_key}&datatype=csv'
        response = requests.get(url)

        if response.status_code == 400:
            raise ValueError(f'Entity not found.')
        if response.status_code == 401:
            raise ConnectionRefusedError(f'An user with api key {AlphaVantageConnector.api_key} has no permission to perform such query.')
        if response.status_code == 404:
            raise ConnectionError(f'Server not found. Path: {url}')
        if response.status_code == 408:
            raise ConnectionResetError('Timeout!')
        if response.status_code == 500:
            raise ConnectionAbortedError('Internal server error!')

        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df)


def scrap_marketwatch(instance, ticker, year):
    url = f'https://www.marketwatch.com/investing/{instance}/{ticker}/downloaddatapartial?startdate=01/01/{year}%2000:00:00&enddate=12/31/{year}%2000:00:00&daterange=d30&frequency=p1d&csvdownload=true&downloadpartial=false&newdates=false'
    response = requests.get(url)

    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), parse_dates=['Date'], decimal=',')
    df.to_csv(f'data/Marketwatch/{ticker}_{year}.csv')


def dump_alphavantage(function, symbol, outputsize='compact'):
    AlphaVantageConnector.get_stock(function, symbol, outputsize).to_csv(f'data/{symbol}.csv')


