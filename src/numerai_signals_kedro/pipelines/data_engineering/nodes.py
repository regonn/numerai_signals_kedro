from typing import Any, Dict, List

import pandas as pd
import numpy as np
import yfinance
import simplejson
import numerapi
from sklearn import preprocessing


def RSI(prices, interval=10):
    '''Computes Relative Strength Index given a price series and lookback interval
  Modified from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
  See more here https://www.investopedia.com/terms/r/rsi.asp'''
    delta = prices.diff()

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(interval).mean()
    RolDown = dDown.rolling(interval).mean().abs()

    RS = RolUp / RolDown
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI


def download_yfinance(ticker_map: pd.DataFrame):
    napi = numerapi.SignalsAPI()
    eligible_tickers = pd.Series(
        napi.ticker_universe(), name='bloomberg_ticker')
    print(f"Number of eligible tickers: {len(eligible_tickers)}")
    print(f"Number of tickers in map: {len(ticker_map)}")
    yfinance_tickers = eligible_tickers.map(
        dict(zip(ticker_map['bloomberg_ticker'], ticker_map['yahoo']))).dropna()
    bloomberg_tickers = ticker_map['bloomberg_ticker']
    print(f'Number of eligible, mapped tickers: {len(yfinance_tickers)}')

    n = 1000  # chunk row size
    chunk_df = [
        yfinance_tickers.iloc[i:i + n]
        for i in range(0, len(yfinance_tickers), n)
    ]

    concat_dfs = []
    print("Downloading data...")
    for df in chunk_df:
        try:
            # set threads = True for faster performance, but tickers will fail, scipt may hang
            # set threads = False for slower performance, but more tickers will succeed
            temp_df = yfinance.download(df.str.cat(sep=' '),
                                        start='2005-12-01',
                                        threads=True)
            temp_df = temp_df['Adj Close'].stack().reset_index()
            concat_dfs.append(temp_df)
        except:
            pass

    full_data = pd.concat(concat_dfs)

    full_data.columns = ['date', 'ticker', 'price']
    full_data['bloomberg_ticker'] = full_data.ticker.map(
        dict(zip(ticker_map['yahoo'], bloomberg_tickers)))

    return full_data


def make_features(full_data: pd.DataFrame):
    full_data.set_index('date', inplace=True)

    ticker_groups = full_data.groupby('ticker')
    full_data['RSI'] = ticker_groups['price'].transform(lambda x: RSI(x))

    date_groups = full_data.groupby(full_data.index)
    full_data['RSI_quintile'] = date_groups['RSI'].transform(
        lambda group: pd.qcut(group, 5, labels=False, duplicates='drop'))
    full_data.dropna(inplace=True)

    ticker_groups = full_data.groupby('ticker')
    num_days = 5

    for day in range(num_days + 1):
        full_data[f'RSI_quintile_lag_{day}'] = ticker_groups[
            'RSI_quintile'].transform(lambda group: group.shift(day))

    for day in range(num_days):
        full_data[f'RSI_diff_{day}'] = full_data[
            f'RSI_quintile_lag_{day}'] - full_data[
            f'RSI_quintile_lag_{day + 1}']
        full_data[f'RSI_abs_diff_{day}'] = np.abs(
            full_data[f'RSI_quintile_lag_{day}'] -
            full_data[f'RSI_quintile_lag_{day + 1}'])

    # 国カテゴリーを追加してラベルエンコーディング処理
    full_data['country'] = full_data['bloomberg_ticker'].str.split(' ', expand=True)[
        1]

    category_columns = ['country']
    for column in category_columns:
        le = preprocessing.LabelEncoder()
        le.fit(full_data[column])
        full_data[column] = le.transform(full_data[column])

    feature_names = category_columns + ['RSI_quintile_lag_{}'.format(num) for num in range(num_days)] + [
        'RSI_diff_{}'.format(num) for num in range(num_days)
    ] + ['RSI_abs_diff_{}'.format(num) for num in range(num_days)]

    return dict(
        full_data_with_features=full_data,
        feature_names=feature_names
    )


def make_input(full_data: pd.DataFrame, feature_names: List[str], targets: pd.DataFrame):
    targets['date'] = pd.to_datetime(targets['friday_date'], format='%Y%m%d')
    ML_data = pd.merge(full_data.reset_index(), targets,
                       on=['date', 'bloomberg_ticker']).set_index('date')

    ML_data.dropna(inplace=True)
    ML_data = ML_data[ML_data.index.weekday == 4]
    ML_data = ML_data[ML_data.index.value_counts() > 50]

    train_data = ML_data[ML_data['data_type'] == 'train']
    test_data = ML_data[ML_data['data_type'] == 'validation']

    return dict(
        train_data=train_data,
        test_data=test_data,
    )
