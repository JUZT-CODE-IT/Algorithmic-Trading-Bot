import sys
import time
import os
import numpy as np
import pandas as pd
import joblib
import alpaca_trade_api as tradeapi
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from dotenv import load_dotenv
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader

load_dotenv("setup.env")

ALPACA_CONFIG = {
    'API_KEY': os.getenv("APCA_API_KEY_ID"),
    'API_SECRET': os.getenv("APCA_API_SECRET_KEY"),
    "PAPER": True
}

restAPI = tradeapi.REST(
    key_id=ALPACA_CONFIG['API_KEY'],
    secret_key=ALPACA_CONFIG['API_SECRET'],
    base_url=os.getenv("APCA_API_BASE_URL")
)

class ML_Trend(Strategy):
    def __init__(self, broker, selected_symbol, stock_name):
        super().__init__(broker)
        self.broker = broker
        self.selected_symbol = selected_symbol
        self.stock_name = stock_name
        self.model = self.load_model()

    def load_model(self):
        try:
            model = joblib.load("model.pkl")
            return model
        except Exception as e:
            print("Error loading model:", e)
            return None

    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data, short_period=7, long_period=21, signal_period=9):
        short_ema = data.ewm(span=short_period, adjust=False).mean()
        long_ema = data.ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal_line, macd - signal_line

    def create_sequences(self, features, labels, sequence_length):
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features.iloc[i:i+sequence_length].values)
            y.append(labels.iloc[i+sequence_length])
        return np.array(X), np.array(y)

    def initialize(self):
        self.sleeptime = "1m"
        self.force_start_immediately = True
        self.ignore_market_hours = True

        historical_data = self.get_historical_prices(self.selected_symbol, 730, "day")
        if historical_data is None:
            print(f"Error: No historical data found for {self.selected_symbol}")
            return

        self.data = historical_data.df
        self.data['RSI'] = self.calculate_rsi(self.data['close'])
        self.data['MACD'], _, _ = self.calculate_macd(self.data['close'])
        self.data.dropna(inplace=True)

        self.features = self.data[['RSI', 'MACD']]
        self.labels = (self.data['close'].shift(-1) > self.data['close']).astype(int)

        sequence_length = 30
        X_train, y_train = self.create_sequences(self.features, self.labels, sequence_length)

        self.model = Sequential([
            LSTM(128, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=50, batch_size=64)

        joblib.dump(self.model, "model.pkl")

        # Accuracy & F1 Score Calculation
        y_pred = (self.model.predict(X_train) > 0.5).astype(int)
        accuracy = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)

        print(f"Training Accuracy: {accuracy:.4f}")
        print(f"Training F1 Score: {f1:.4f}")

    def on_trading_iteration(self):
        bars = self.get_historical_prices(self.selected_symbol, 500, "day")
        gld = bars.df

        if gld.empty:
            print("No data available.")
            return

        gld['RSI'] = self.calculate_rsi(gld['close'])
        gld['MACD'], _, _ = self.calculate_macd(gld['close'])
        gld.dropna(inplace=True)

        last_close_price = gld["close"].iloc[-1]
        last_known_data = gld.iloc[-1][['RSI', 'MACD']].values.reshape(1, 1, -1)
        
        predicted_price = self.model.predict(last_known_data)[0][0]
        predicted_prices = [last_close_price * np.exp(predicted_price - 0.5)] * 28

        future_dates = pd.date_range(gld.index[-1] + pd.Timedelta(days=1), periods=28)
        plt.plot(gld.index, gld["close"], label="Historical Price", color="blue")
        plt.plot(future_dates, predicted_prices, label="Predicted Price", color="orange", linestyle="dotted")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"{self.stock_name} ({self.selected_symbol}) - Price vs Prediction")
        plt.legend()
        plt.grid()
        plt.savefig("trade_decision.png")
        plt.show()

    def execute_trade(self):
        print(f"Trade executed for {self.selected_symbol}")

def search_assets(query):
    tradable_assets = [asset for asset in restAPI.list_assets() if asset.tradable]
    query = query.lower()
    results = {asset.symbol: asset.name for asset in tradable_assets if query in asset.name.lower()}
    return results

if __name__ == "__main__":
    while True:
        query = input("Enter the stock/crypto name (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            sys.exit(0)

        results = search_assets(query)
        if not results:
            print("No matches found. Try again.")
            continue

        print("\nMatching results:")
        for idx, (symbol, name) in enumerate(results.items(), 1):
            print(f"{idx}. {name} ({symbol})")

        try:
            choice = int(input("\nEnter selection number: "))
            selected_symbol = list(results.keys())[choice - 1]
            stock_name = results[selected_symbol]
            print(f"Selected: {stock_name} ({selected_symbol})")
            break
        except (ValueError, IndexError):
            print("Invalid selection. Try again.")

    broker = Alpaca(ALPACA_CONFIG)
    strategy = ML_Trend(broker=broker, selected_symbol=selected_symbol, stock_name=stock_name)
    trader = Trader(logfile="")
    trader.add_strategy(strategy=strategy)
    time.sleep(5)
    trader.run_all()
