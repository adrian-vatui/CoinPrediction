import os
import os.path
import pickle

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential, layers
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

plt.style.use('fivethirtyeight')


class LSTM:
    def __init__(self, csv_name, lag, model=None):
        self.csv_name = csv_name
        self.lag = lag if lag > 0 else 1
        self.model = model

    def get_prediction(self):
        if self.model is None:
            self.train()

        # get last 60 days from csv, scale and feed them to model
        df = pd.read_csv(os.path.join('training', 'processed', self.csv_name), date_parser=True)
        df = np.array(df['close']).reshape(-1, 1)

        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
        past_data = df[-(60 + self.lag):]

        x = []
        for i in range(60, past_data.shape[0]):
            x.append(past_data[i - 60:i])

        x = np.array(x)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        y_pred = self.model.predict(x)

        y_pred = scaler.inverse_transform(y_pred)
        return y_pred

    @staticmethod
    def prepare_data(df, window, lag):
        x = []
        y = []
        for i in range(window, df.shape[0] - lag):
            x.append(df[i - window:i])
            y.append(df[i + lag])

        return np.array(x), np.array(y)

    def build_model(self, input_shape):
        self.model = Sequential()

        # input layer
        self.model.add(layers.LSTM(32, return_sequences=True, input_shape=input_shape))
        self.model.add(layers.Dropout(0.2))

        # pre-output layer
        self.model.add(layers.LSTM(32))
        self.model.add(layers.Dropout(0.2))

        # output layer
        self.model.add(layers.Dense(1))

        # self.model.summary()
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, epochs=10, show_plots=False):
        # loading the data
        df = pd.read_csv(os.path.join('training', 'processed', self.csv_name), date_parser=True)

        train_data = df[df['date'] < '2021-09-01'].copy()
        test_data = df[df['date'] > '2021-09-01'].copy()

        train_data = np.array(train_data['close']).reshape(-1, 1)

        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)

        x_train, y_train = self.prepare_data(train_data, 60, self.lag - 1)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        self.build_model(input_shape=(x_train.shape[1], 1))

        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_split=0.1, verbose=0)

        if show_plots:
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(len(loss))
            plt.figure()
            plt.plot(epochs, loss, 'b', label='Training loss')
            plt.plot(epochs, val_loss, 'r', label='Validation loss')
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()

            past_60_days = df[df['date'] < '2021-09-01'].copy().tail(60)
            df2 = past_60_days.append(test_data, ignore_index=True)

            df2 = np.array(df2['close']).reshape(-1, 1)
            inputs = scaler.fit_transform(df2)

            x_test, y_test = self.prepare_data(inputs, 60, self.lag - 1)

            y_pred = self.model.predict(x_test)

            y_pred = scaler.inverse_transform(y_pred)
            y_test = scaler.inverse_transform(y_test)

            plt.figure(figsize=(14, 5))
            plt.plot(y_test, color='red', label='Real Bitcoin Price')
            plt.plot(y_pred, color='green', label='Predicted Bitcoin Price')
            plt.title('Bitcoin Price Prediction using RNN-LSTM')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

    def save(self, path='models/LSTM.h5'):
        self.model.save(path)

    @staticmethod
    def load(path='models/LSTM.h5'):
        filename = path.split('/')[-1]
        filename = filename[:-3]  # to remove .h5 extension
        coin_name = filename.split('_')[1]
        lag = int(filename.split('_')[2])

        if os.path.exists(path):
            return LSTM(csv_name=f'Binance_{coin_name}_d.csv', lag=lag, model=keras.models.load_model(path))

        lstm = LSTM(f'Binance_{coin_name}_d.csv', lag)
        lstm.train(epochs=10)
        lstm.save(os.path.join("models", 'LSTM', f'lstm_{coin_name}_{lag}.h5'))
        return lstm


class SVM:
    def __init__(self, csv_name, lag, model=None):
        self.csv_name = csv_name
        self.prediction_days = lag if lag > 0 else 1
        self.model = model
        self.prediction_days_array = []

    def get_prediction(self):
        if self.model is None:
            self.train()

        future_values = self.model.predict(self.prediction_days_array)
        return future_values

    def prepare_data(self):
        df = pd.read_csv(os.path.join("training", "processed", self.csv_name))[['close']]

        # creating one more column (prediction) which "has" stock values prediction_days in advance
        # by shifting prediction_days rows up
        df['prediction'] = (df[['close']].shift(-self.prediction_days))

        # creating a dataset x and converting it into a numpy array , which will contain the actual values
        x = np.array(df.drop(columns='prediction'))
        # Removing the last prediction_days rows
        x = x[:-self.prediction_days]

        # creating a dataset y which will contain the predicted values and converting it into numpy array
        y = np.array(df['prediction'])

        # Removing the last prediction_days rows
        y = y[:-self.prediction_days]

        # scaler = StandardScaler()
        # x = scaler.fit_transform(x)

        self.prediction_days_array = np.array(df.drop(columns='prediction'))[-self.prediction_days:]

        return train_test_split(x, y, test_size=0.2)

    def train(self):
        x_train, x_test, y_train, y_test = self.prepare_data()

        self.model = SVR(kernel='rbf', C=1000, gamma=0.0000001)
        self.model.fit(x_train, y_train)

        svm_rbf_confidence = self.model.score(x_test, y_test)
        # print("svr_rbf confidence: ", self.svm_rbf_confidence)

    def save(self, path='models/svm_model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load(path='models/svm_model.pkl'):
        with open(path, 'rb') as f:
            svm_model = pickle.load(f)
        return svm_model


class Sentiment:
    @staticmethod
    def getSubjectivity(twt):
        return TextBlob(twt).sentiment.subjectivity

    @staticmethod
    def getPolarity(twt):
        return TextBlob(twt).sentiment.polarity

    @staticmethod
    def getSentiment(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    @staticmethod
    def baseGraph(type):
        theP = plt.figure(figsize=(6, 8))
        if type == "Points":
            plt.title('Sentiment Analysis Scatter Plot')
            plt.xlabel('Negativity -> Positivity')
            plt.ylabel('Objectivity -> Subjectivity')
        if type == "Bar":
            plt.title('Sentiment Analysis Bar Plot')
            plt.xlabel('Sentiment')
            plt.ylabel('Number of Tweets')
        return theP

    @staticmethod
    def tweets_sentiments(csv_name):
        plt.rcParams.update({'font.size': 7})
        data = pd.read_csv(os.path.join("training", "processed", csv_name))
        df = pd.DataFrame(data, columns=['text'])
        df['subjectivity'] = df['text'].apply(Sentiment.getSubjectivity)
        df['polarity'] = df['text'].apply(Sentiment.getPolarity)
        df['sentiment'] = df['polarity'].apply(Sentiment.getSentiment)

        return df
        # theP = plt.figure(figsize=(6, 8))
        #
        # if type == "Points":
        #     for i in range(0, df.shape[0]):
        #         plt.scatter(df['polarity'][i], df['subjectivity']
        #         [i], color='Purple')
        #
        #     plt.title('Sentiment Analysis Scatter Plot')
        #     plt.xlabel('Negativity -> Positivity')
        #     plt.ylabel('Objectivity -> Subjectivity')
        #     return theP
        #
        # if type == "Bar":
        #     df['sentiment'].value_counts().plot(kind='bar')
        #     plt.title('Sentiment Analysis Bar Plot')
        #     plt.xlabel('Sentiment')
        #     plt.ylabel('Number of Tweets')
        #     return theP
