import datetime
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from models import LSTM, Sentiment
from models import SVM
from utils import get_delta_days, evaluate_rsi, get_last_rsi, calculate_prediction_rsi, download_csv, preprocess
from datetime import date
import warnings

warnings.filterwarnings("ignore")

df = None


def dropdown(text, options):
    option = st.selectbox(text, options)
    return option


def draw_price_graph(prices):
    # TODO: use date as x axis (pandas function)
    chart_data = pd.DataFrame({
        'Days': list(range(len(prices))),
        'Predicted Price': prices.tolist()
    })

    chart = alt.Chart(chart_data).mark_line().encode(
        x="Days",
        y=alt.Y('Predicted Price', scale=alt.Scale(zero=False))
    ).interactive()

    st.altair_chart(chart)


def draw_sentiment_graph(graph_type):
    sentiment_df = Sentiment.tweets_sentiments("light.csv")

    if graph_type == "Scatter Plot":
        chart = alt.Chart(sentiment_df).mark_circle(size=60).encode(
            x=alt.X('polarity', axis=alt.Axis(title='Negativity -> Positivity')),
            y=alt.Y('subjectivity', axis=alt.Axis(title="Objectivity -> Subjectivity"))
        ).interactive()

    else:
        chart = alt.Chart(sentiment_df).mark_bar().encode(
            x=alt.X('sentiment:N'),
            y=alt.Y('count(sentiment):Q')
        ).properties(width=400).interactive()

    st.altair_chart(chart)


def select_date():
    date = st.date_input("Select a date", value=None, min_value=None, max_value=None, key=None)
    return date.year, date.month, date.day


def select_coin():
    coin = dropdown("Select a coin", ("Bitcoin", "Ethereum"))
    if coin == "Bitcoin":
        return "BTCUSDT"
    else:
        return "ETHUSDT"


def run_model(model, coin_name, year, month, day):
    global df
    csv_name = f"Binance_{coin_name}_d.csv"
    df = pd.read_csv(os.path.join("training", "processed", csv_name))
    lag = get_delta_days(datetime.date(year, month, day), df)

    if lag <= 0:
        lag = 1

    if model == "LSTM":
        lstm = LSTM.load(f"models/LSTM/lstm_{coin_name}_{lag}.h5")
        prices = lstm.get_prediction()
    else:
        svm = SVM(f'Binance_{coin_name}_d.csv', lag)
        prices = svm.get_prediction()

    prices = prices.reshape(len(prices))  # reshape into 1-dimensional array
    return prices


def get_updated_csv():
    today = date.today().strftime("%Y-%m-%d")

    btc_raw_path = os.path.join(".", "training", "raw", "Binance_BTCUSDT_d.csv")
    eth_raw_path = os.path.join(".", "training", "raw", "Binance_ETHUSDT_d.csv")
    btc_processed_path = os.path.join(".", "training", "processed", "Binance_BTCUSDT_d.csv")
    eth_processed_path = os.path.join(".", "training", "processed", "Binance_ETHUSDT_d.csv")

    dataset = pd.DataFrame(pd.read_csv(btc_processed_path))

    if today != dataset.iloc[-1]['date'].split(" ")[0]:
        os.remove(btc_raw_path)
        os.remove(eth_raw_path)
        download_csv(btc_raw_path, "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv")
        download_csv(eth_raw_path, "https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_d.csv")
        os.remove(btc_processed_path)
        os.remove(eth_processed_path)
        preprocess(btc_raw_path)
        preprocess(eth_raw_path)


def main():
    global df

    get_updated_csv()
    title = "Coin prediction"
    st.set_page_config(page_title=title)
    st.title(title)

    col1, col2, col3 = st.columns([1, 0.2, 1.5])

    with col1:
        year, month, day = select_date()
        coin_name = select_coin()
        model = dropdown("Select a prediction model", ("LSTM", "SVM"))
        graph_type = dropdown("Select a graph type for the sentiment analysis", ("Bar", "Scatter Plot"))

    with col3:
        prices = run_model(model, coin_name, year, month, day)

        for price in prices:
            calculate_prediction_rsi(df, price)

        st.markdown(f'<center>{evaluate_rsi(get_last_rsi(df))}</center>', unsafe_allow_html=True)
        draw_price_graph(prices)
        draw_sentiment_graph(graph_type)


if __name__ == '__main__':
    main()
