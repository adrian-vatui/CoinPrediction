import os
import re
from datetime import date

import pandas as pd
import pandas_ta as pdt
import requests

from models import LSTM

PERIOD = 14
FAST_PERIOD = 12
SLOW_PERIOD = 26


def preprocess(csv_path):
    output_path = os.path.join(
        "training", "processed", os.path.split(csv_path)[1])

    df = pd.read_csv(csv_path, header=1)
    df = df.iloc[::-1]

    calculate_rsi(df)
    calculate_MA(df)
    calculate_MACD(df)

    df.to_csv(output_path, na_rep='N/A', index=False)
    print(df)


def calculate_rsi(dataframe):
    dataframe.ta.rsi(close='close', length=PERIOD, append=True)


def calculate_MA(dataframe):
    dataframe['MA'] = dataframe['close'].rolling(window=PERIOD).mean()


def calculate_MACD(dataframe):
    dataframe.ta.macd(close='close', fast=FAST_PERIOD, slow=SLOW_PERIOD, append=True)


def calculate_prediction_rsi(dataframe, close):
    dataframe.loc[dataframe.shape[0]] = [None, None, None, None, None, None, close, None, None, None, None, None, None,
                                         None, None]
    calculate_rsi(dataframe)


def get_last_rsi(dataframe):
    return dataframe[f'RSI_{PERIOD}'].iloc[-1]


def evaluate_rsi(rsi):
    lower_bound = 30
    upper_bound = 70

    if lower_bound <= rsi <= upper_bound:
        return "The RSI is within bounds - it is safe to buy"
    elif rsi < lower_bound:
        return "The coin is oversold and undervalued"
    elif rsi > upper_bound:
        return "The coin is overbought and overvalued"


def download_csv(csv_path, url):
    r = requests.get(
        url=url, verify=False).content.decode('utf-8')
    with open(csv_path, "w") as f:
        f.write(r)


def train_lstm(csv_name, min_bound=1, max_bound=50, epochs=1):
    for i in range(min_bound, max_bound):
        csv_name = csv_name
        lstm = LSTM(csv_name, i)
        lstm.train(epochs=epochs)
        coin_name = csv_name.split('_')[1]
        file_name = os.path.join("models", 'LSTM', f'lstm_{coin_name}_{i}.h5')
        lstm.save(file_name)
        print(f'Trained {coin_name} with lag {i}')


def get_delta_days(date_from_user, df1):
    df = df1[['date']]
    last_date = str(df.iloc[-1].date[:10])
    # print(last_date)
    last_date = last_date.split('-')
    # print(last_date)
    processed_date = []
    for value in last_date:
        if re.match("^[0][1-9]$", value):
            value = value.strip('0')
        processed_date.append(value)
    # print(processed_date)
    start_date = date(int(processed_date[0]), int(processed_date[1]), int(processed_date[2]))
    end_date = date_from_user
    delta = end_date - start_date
    return delta.days
