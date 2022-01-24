import os

from models import LSTM
from utils import preprocess, train_lstm

preprocess(os.path.join("training", "raw", "Binance_BTCUSDT_d.csv"))
preprocess(os.path.join("training", "raw", "Binance_ETHUSDT_d.csv"))
# download_csv(os.path.join("training", "raw", "Binance_BTCUSDT_d.csv"))


train_lstm("Binance_BTCUSDT_d.csv", epochs=100, min_bound=1, max_bound=51)
train_lstm("Binance_ETHUSDT_d.csv", epochs=100, min_bound=1, max_bound=51)

# lstm = LSTM("Binance_ETHUSDT_d.csv", 1)
# lstm.train(show_plots=True, epochs=100)
# plot = lstm.get_prediction()
# matplotlib.pyplot.show()
