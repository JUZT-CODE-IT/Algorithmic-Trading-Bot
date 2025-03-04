import sys
import time
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi 
import os
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime, timedelta
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")

load_dotenv("setup.env")

class ML_Trend(Strategy):

    force_start_immediately = True

    def before_market_opens(self):
        print("[ML_Trend] Skipping before_market_opens() to start trading immediately.")
    
    @staticmethod
    def calculate_historical_volatility(data, period=20):
        log_returns = np.log(data['close'] / data['close'].shift(1))
        hv = log_returns.rolling(window=period).std() * np.sqrt(252)  # Annualize
        return hv

    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data, short_period=7, long_period=21, signal_period=9):
        short_ema = data.ewm(span=short_period, min_periods=1, adjust=False).mean()
        long_ema = data.ewm(span=long_period, min_periods=1, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_period, min_periods=1, adjust=False).mean()
        macd_histogram = macd - signal_line
        return macd, signal_line, macd_histogram
    
    @staticmethod
    def create_sequences(data, labels, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i : i + sequence_length].values)
            y.append(labels.iloc[i + sequence_length])
        return np.array(X), np.array(y)
    
    def initialize(self):
        self.sleeptime = "1m"  # ✅ Run every 1 minute
        self.force_start_immediately = True  # ✅ Start immediately
        self.ignore_market_hours = True  # ✅ Run even when the market is closed

        historical_data = self.get_historical_prices(selected_symbol, 730, "day")
        
        if historical_data is None:
            print(f"Error: No historical data found for {selected_symbol}")
            return

        self.data = historical_data.df
        self.data['7_day_ma'] = self.data['close'].rolling(7).mean()
        self.data['21_day_ma'] = self.data['close'].rolling(21).mean()
        self.data['RSI'] = self.calculate_rsi(self.data['close'])
        self.data['MACD'], _, _ = self.calculate_macd(self.data['close'])
        self.data['Price_Movement'] = (
            self.data['close'].rolling(2)
            .apply(lambda x: x.iloc[-1] > x.iloc[0] if len(x) == 2 else np.nan)
        )

        self.data['Price_Movement'] = self.data['Price_Movement'].fillna(0).astype(int)
        self.features = self.data[['7_day_ma', '21_day_ma', 'RSI', 'MACD']]
        self.labels = self.data['Price_Movement']
        self.features = self.features.fillna(self.features.mean())

        sequence_length = 30
        split = int(0.8 * len(self.features))

        X_train, X_test = self.features.iloc[:split], self.features.iloc[split:]
        y_train, y_test = self.labels.iloc[:split], self.labels.iloc[split:]

        X_train, y_train = self.create_sequences(X_train, y_train, sequence_length)
        X_test, y_test = self.create_sequences(X_test, y_test, sequence_length)

        self.model = Sequential()
        self.model.add(LSTM(128, activation='relu', return_sequences=True))
        self.model.add(LSTM(64, activation='relu', return_sequences=True))
        self.model.add(LSTM(32, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        self.model.fit(X_train, y_train, epochs=50, batch_size=64, class_weight=class_weight_dict)

        predictions = (self.model.predict(X_test) > 0.5).astype(int).flatten()
        y_test = np.array(y_test).flatten()

        print("Accuracy: ", accuracy_score(y_test, predictions))

    def on_trading_iteration(self):

        self.ignore_market_hours = True

        print("Fetching historical data...")
        
        bars = self.get_historical_prices(selected_symbol, 500, "day")  # Get two years of data
        gld = bars.df

        if gld.empty:
            print("No valid data available.")
            return

        # Ensure Date Index is in Correct Format
        gld.index = pd.to_datetime(gld.index)

        print("Computing indicators...")
        gld['7_day_ma'] = gld['close'].rolling(7).mean()
        gld['21_day_ma'] = gld['close'].rolling(21).mean()
        gld['RSI'] = self.calculate_rsi(gld['close'])
        gld['MACD'], _, _ = self.calculate_macd(gld['close'])
        gld.dropna(inplace=True)  # Drop NaNs

        if gld.empty:
            print("No data left after dropping NaNs.")
            return

        last_close_price = gld["close"].iloc[-1]
        last_known_data = gld.iloc[-1][['7_day_ma', '21_day_ma', 'RSI', 'MACD']].values.reshape(1, 1, -1)
        
        predicted_prices = []
        future_dates = pd.date_range(gld.index[-1] + pd.Timedelta(days=1), periods=14)
        
        for i in range(14):
            prediction = self.model.predict(last_known_data)[0][0]
            predicted_price = last_close_price * (1 + (prediction - 0.5) / 10)  # Normalize prediction
            predicted_prices.append(predicted_price)
            
            # Update moving averages dynamically
            ma_7 = np.mean(predicted_prices[-7:]) if len(predicted_prices) >= 7 else np.mean(predicted_prices)
            ma_21 = np.mean(predicted_prices[-21:]) if len(predicted_prices) >= 21 else np.mean(predicted_prices)
            
            new_features = np.array([[ma_7, ma_21, last_known_data[0, 0, 2], last_known_data[0, 0, 3]]]).reshape(1, 1, -1)
            last_known_data = new_features

        predicted_prices = np.array(predicted_prices)
        
        # Adjust Buy/Sell Thresholds Based on Historical Volatility
        vol = np.std(gld['close'].pct_change().dropna())
        buy_threshold = 1 + (vol * 2)
        sell_threshold = 1 - (vol * 2)
        
        buy_signals = predicted_prices > last_close_price * buy_threshold
        sell_signals = predicted_prices < last_close_price * sell_threshold
        
        plt.figure(figsize=(12, 6))
        plt.plot(gld.index, gld["close"], label="Actual Price", color="blue", linewidth=2)
        plt.plot(future_dates, predicted_prices, label="Predicted Price", color="orange", linestyle="dotted", linewidth=2)
        
        plt.scatter(np.array(future_dates)[buy_signals], predicted_prices[buy_signals], color="green", marker="^", label="BUY", s=100)
        plt.scatter(np.array(future_dates)[sell_signals], predicted_prices[sell_signals], color="red", marker="v", label="SELL", s=100)

        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"{stock_name} ({selected_symbol}) - Price vs Prediction")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)

        plt.savefig("trade_decision.png")
        plt.show(block=True)

        # Execute trade based on decision
        # quantity = 200
        # if decision == "BUY":
        #     pos = self.get_position(selected_symbol)
        #     if pos is None:
        #         order = self.create_order(selected_symbol, quantity, "buy")
        #         self.submit_order(order)
        # else:  # SELL
        #     pos = self.get_position(selected_symbol)
        #     if pos is not None:
        #         self.sell_all()

def search_assets(query):
    tradable_assets = [asset for asset in restAPI.list_assets() if asset.tradable]
    query = query.lower()
    results = {asset.symbol: asset.name for asset in tradable_assets if query in asset.name.lower()}
    return results

if __name__ == "__main__":

    load_dotenv("setup.env") #Loading .env file for confidential info.

    ALPACA_CONFIG = {
    'API_KEY': os.getenv("APCA_API_KEY_ID"),
    'API_SECRET': os.getenv("APCA_API_SECRET_KEY"),
    "PAPER": True  # Use this instead of ENDPOINT
    }

    restAPI=tradeapi.REST(key_id=ALPACA_CONFIG['API_KEY'],secret_key=ALPACA_CONFIG['API_SECRET'],base_url=os.getenv("APCA_API_BASE_URL"))

    while True:
        query = input("Enter the name of the stock/crypto (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        results = search_assets(query)

        if not results:
            print("No matches found. Try again.")
            continue

        print("\nMatching results:")
        for idx, (symbol, name) in enumerate(results.items(), 1):
            print(f"{idx}. {name} ({symbol})")

        try:
            choice = int(input("\nEnter the number corresponding to your selection: "))
            selected_symbol = list(results.keys())[choice - 1]
            stock_name=results[selected_symbol]
            print(f"\nYou have selected: {results[selected_symbol]} ({selected_symbol})")

            action = input(f"\nDo you want to trade {results[selected_symbol]}? (Y/N): ").strip().lower()
            if action == 'y':
                trade = True
            else:
                action = input(f"\nDo you want to perform backtesting? (Y/N): ").strip().lower()
                backtest = action == 'y'

            break  # Exit after successful selection

        except (ValueError, IndexError):
            print("Invalid selection. Try again.")

    if trade:
        broker = Alpaca(ALPACA_CONFIG)  # Paper trading is already set in ALPACA_CONFIG
        try:
            strategy = ML_Trend(broker=broker)
        except tradeapi.rest.APIError as e:
            print("Error:", e)

        trader = Trader(logfile="")  # Create Trader without extra arguments
        trader.add_strategy(strategy=strategy)

        time.sleep(5)  # Wait 5 seconds before starting

        trader.run_all()  
    elif backtest:
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        ML_Trend.backtest(
            YahooDataBacktesting,
            start,
            end,
        )
    else:
        print("\nExiting... No trade or backtest selected. Have a great day!")
        sys.exit(0)