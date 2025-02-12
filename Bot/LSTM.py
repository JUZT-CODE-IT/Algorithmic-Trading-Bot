from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from config import ALPACA_CONFIG
from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
class LSTM(Strategy):
    def MinMaxScaling(self):
        data = self.get_historical_prices("GLD", 51, "day").df
        print(data)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['close', 'other_features']])
        print(scaled_data)
        scaled_data.plot(x="Date", y="Price",kind="line")
if __name__ == "__main__":
    broker = Alpaca(ALPACA_CONFIG)
    strategy = LSTM(broker=broker)
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()