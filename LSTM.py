import sys
import time
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import alpaca_trade_api as tradeapi
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import matplotlib.animation as animation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from lumibot.entities import Order

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

load_dotenv("setup.env")

ALPACA_CONFIG = {
    'API_KEY': os.getenv("APCA_API_KEY_ID"),
    'API_SECRET': os.getenv("APCA_API_SECRET_KEY"),
    "PAPER": True,
    "BASE_URL": os.getenv("APCA_API_BASE_URL")
}

restAPI = tradeapi.REST(
    key_id=ALPACA_CONFIG['API_KEY'],
    secret_key=ALPACA_CONFIG['API_SECRET'],
    base_url=os.getenv("APCA_API_BASE_URL")
)

class ML_Trend(Strategy):
    def __init__(self, selected_symbol, stock_name, broker=None,data=None,lstm_model=None,xgb_model=None):
        if broker:
            super().__init__(broker)
        self.broker = broker
        self.selected_symbol = selected_symbol
        self.stock_name = stock_name
        if data is not None:
            self.data = data
        else:
            self.data = self.get_historical_prices(self.selected_symbol, 1000, "day")
        
        # Initialize models if not passed
        self.lstm_model = lstm_model if lstm_model is not None else self.load_lstm_model()
        self.xgb_model = xgb_model if xgb_model is not None else self.load_xgb_model()


    def load_lstm_model(self):
        try:
            model = load_model("lstm_model.h5")
            return model
        except Exception as e:
            print("Error loading LSTM model:", e)
            return None

    def load_xgb_model(self):
        try:
            model = joblib.load("xgboost_model.pkl")
            return model
        except Exception as e:
            print("Error loading XGBoost model:", e)
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

    @staticmethod
    def calculate_bollinger_bands(data, window=20):
        data['SMA'] = data['close'].rolling(window=window).mean()
        data['STD']= data['close'].rolling(window=window).std()
        data['Upper_Band']= data['SMA'] + (data['STD'] * 2)
        data['Lower_Band']= data['SMA'] - (data['STD'] * 2)
        data['BB_Width']= (4*data['STD'])/data['SMA']
        data['%B']=(data['close']-data['Lower_Band'])/4*data['STD']
        return data

    @staticmethod
    def calculate_volatility(data, window=20):
        data['Volatility'] = data['close'].pct_change().rolling(window=window).std()
        return data

    @staticmethod
    def calculate_atr(data, period=14):
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()

        true_range = high_low.combine(high_close, max).combine(low_close, max)
        atr = true_range.rolling(window=period, min_periods=1).mean()

        data['ATR'] = atr
        return data
    
    @staticmethod
    def plot_live(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        def update(frame):
            ax.clear()
            ax.plot(self.balance_history, label="Portfolio Balance", color="blue", linewidth=2)

            # Plot buy/sell points
            for trade in self.trades:
                index, price, trade_type = trade
                if trade_type == "BUY":
                    ax.scatter(index, self.balance_history[index - 50], color="green", marker="^", s=100, label="BUY" if index == self.trades[0][0] else "")
                elif trade_type == "SELL":
                    ax.scatter(index, self.balance_history[index - 50], color="red", marker="v", s=100, label="SELL" if index == self.trades[0][0] else "")

            ax.set_xlabel("Time")
            ax.set_ylabel("Portfolio Value ($)")
            ax.set_title("Live Trading Performance")
            ax.legend()
            ax.grid(True)

        ani = animation.FuncAnimation(fig, update, interval=1000)  # Update every second
        plt.show()

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

        if self.data is None:
            print(f"Error: No historical data found for {self.selected_symbol}")
            return

        self.data = self.data.df
        print(self.data.columns)
        self.data['RSI'] = self.calculate_rsi(self.data['close'])
        self.data['MACD'], _, _ = self.calculate_macd(self.data['close'])
        self.data = self.calculate_bollinger_bands(self.data)
        self.data = self.calculate_volatility(self.data)
        self.data = self.calculate_atr(self.data)
        self.data['Momentum'] = (self.data['close'] - self.data['close'].shift(5)) / self.data['close'].shift(5)
        self.data['Price_Change'] = (self.data['close'] - self.data['open']) / self.data['open']
        correlation = self.data[['ATR', 'Volatility']].corr()
        self.features = self.data[['RSI', 'MACD', 'SMA','BB_Width','%B', 'Volatility','ATR', 'Momentum', 'Price_Change']]
        self.data.dropna(inplace=True)

        scaler = MinMaxScaler()
        self.features = pd.DataFrame(scaler.fit_transform(self.features.copy()),columns=self.features.columns,index=self.features.index)
        joblib.dump(scaler,"scaler.pkl")
        self.labels = (self.data['close'].shift(-1) > self.data['close']).astype(int)
        self.labels.dropna(inplace=True)
        # unique, counts = np.unique(self.labels, return_counts=True)
        # print(dict(zip(unique, counts)))
        self.features = self.features[:len(self.labels)]

        split = int(len(self.features) * 0.70)
        X_train, X_test = self.features.iloc[:split], self.features.iloc[split:]
        y_train, y_test = self.labels.iloc[:split], self.labels.iloc[split:]

        sequence_length = 50
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, sequence_length)


        # Train LSTM
        self.lstm_model = Sequential([
            LSTM(256, activation='tanh', return_sequences=True),
            Dropout(0.3),
            LSTM(128, activation='tanh', return_sequences=True),
            Dropout(0.3),
            LSTM(64, activation='tanh'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.lstm_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=64)
        self.lstm_model.save("lstm_model.h5")

        y_pred_lstm = self.lstm_model.predict(X_test_seq)
        y_pred_lstm = (y_pred_lstm > 0.5).astype(int)
        LSTM_Accuracy = accuracy_score(y_test_seq, y_pred_lstm)
        LSTM_F1 = f1_score(y_test_seq, y_pred_lstm)

        print(f"LSTM_Accuracy: {LSTM_Accuracy:.4f}")
        print(f"LSTM_F1: {LSTM_F1:.4f}")

        self.features.dropna(inplace=True)
        self.labels = self.labels[self.features.index]  # Keep labels aligned with features
        # Train XGBoost
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(self.features, self.labels)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=500,  
            learning_rate=0.02,  
            max_depth=4,  
            subsample=0.8,  
            colsample_bytree=0.8,  
            random_state=42
        )


        self.xgb_model.fit(X_resampled, y_resampled)
        xgb.plot_importance(self.xgb_model)
        plt.savefig("Output.png")  
        joblib.dump(self.xgb_model, "xgboost_model.pkl")

        y_pred_xgb = self.xgb_model.predict(X_test)
        y_pred_xgb = (y_pred_xgb > 0.5).astype(int)
        XGBoost_Accuracy = accuracy_score(y_test, y_pred_xgb)
        XGBoost_F1 = f1_score(y_test, y_pred_xgb)

        print(f"XGBoost_Accuracy: {XGBoost_Accuracy:.4f}")
        print(f"XGBoost_F1: {XGBoost_F1:.4f}")

    def on_trading_iteration(self):
        bars = restAPI.get_barset([self.selected_symbol], timeframe="minute", limit=120).df

        if bars is None:
            print("No data available.")
            return

        data = bars.df
        data['RSI'] = self.calculate_rsi(data['close'])
        data['MACD'], _, _ = self.calculate_macd(data['close'])
        data = self.calculate_bollinger_bands(data)
        data = self.calculate_volatility(data)
        data = self.calculate_atr(data)
        data['Momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
        data['Price_Change'] = (data['close'] - data['open']) / data['open']
        data.dropna(inplace=True)

        # LSTM Prediction (Price)
        last_sequence = data[['RSI', 'MACD', 'SMA', 'BB_Width', '%B', 'Volatility', 'ATR', 'Momentum', 'Price_Change']].iloc[-50:, :]
        num_rows = last_sequence.shape[0]
        if num_rows < 50:
            print(f"Warning: Only {num_rows} rows available, expected 50.")
        last_sequence = last_sequence.values.reshape(1, num_rows, -1)
        predicted_price = self.lstm_model.predict(last_sequence)[0][0]

        # XGBoost Prediction (Buy/Sell)
        last_features = data[['RSI', 'MACD', 'SMA','BB_Width','%B', 'Volatility','ATR', 'Momentum', 'Price_Change']].iloc[-1].values.reshape(1, -1)
        signal = self.xgb_model.predict(last_features)[0]  
        
        atr = self.calculate_atr(self.data)  # Ensure ATR is calculated in your dataset
        cash = self.get_cash()
        entry_price = self.get_last_price(self.selected_symbol)

        # ATR-Based Position Sizing (Risk 2% of capital)
        risk_per_trade = cash * 0.02 
        atr_value = atr.iloc[-1, 0] if isinstance(atr, pd.DataFrame) else atr
        position_size = risk_per_trade / (atr_value * 2)
        quantity = max(1, int(position_size))

        if signal == 1:
            stop_price = entry_price - (atr * 1.5)  # Adjust multiplier based on recent volatility
            limit_price = entry_price + (atr * 3)  # Aim for at least 2:1 risk-reward

            order = Order(
                strategy=self,
                asset=self.selected_symbol,
                quantity=quantity,
                side="buy",
                order_type="market",
                time_in_force="gtc",
                stop_price=stop_price,
                limit_price=limit_price
            )
            print(f"Limit Price Type: {type(self.limit_price)} | Value: {limit_price}")
            self.submit_order(order)
                # Execute buy order
        else:
            position = self.get_position(symbol)
            if position:
                print(position)
                stop_price = entry_price + (atr * 1.5)  # 2x ATR above for short
                limit_price = entry_price - (atr * 3)  # 2x ATR below for short

                order = Order(
                    strategy=self,
                    asset=self.selected_symbol,
                    quantity=position.quantity,
                    side="sell",
                    order_type="market",
                    time_in_force="gtc",
                    stop_price=stop_price,
                    limit_price=limit_price
                )
                self.submit_order(order)
        portfolio_value = self.get_portfolio_value()
        self.balance_history.append(portfolio_value)
        self.plot_live()

class Backtest(ML_Trend):
    def __init__(self, data, lstm_model, xgb_model, initial_balance=100000):
        self.data = data.df if hasattr(data, "df") else pd.DataFrame(data)
        self.data['RSI'] = self.calculate_rsi(self.data['close'])
        self.data['MACD'], _, _ = self.calculate_macd(self.data['close'])
        self.data = self.calculate_bollinger_bands(self.data)
        self.data = self.calculate_volatility(self.data)
        self.data = self.calculate_atr(self.data)
        self.data['Momentum'] = (self.data['close'] - self.data['close'].shift(5)) / self.data['close'].shift(5)
        self.data['Price_Change'] = (self.data['close'] - self.data['open']) / self.data['open']
        self.features = self.data[['RSI', 'MACD', 'SMA','BB_Width','%B', 'Volatility','ATR', 'Momentum', 'Price_Change']]
        self.data.dropna(inplace=True)
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        self._cash_balance = initial_balance
        self.initial_balance = initial_balance
        self.trades = []
        self.balance_history = []
        self._positions = {}  

    def get_cash(self):
        return self._cash_balance  

    def get_positions(self):
        return self._positions

def run(self):
    """Runs the backtest on historical data with a live-updating graph."""

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Time")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Backtest Performance (Live)")
    
    balance_line, = ax.plot([], [], color="blue", label="Portfolio Value")
    ax.legend()
    
    self.balance_history = []
    self.trades = []
    
    def update(frame):
        i = frame + 50  # Start after 50 data points for LSTM
        if i >= len(self.data):
            ani.event_source.stop()  # Stop animation when data ends
            return
        
        current_price = self.data.iloc[i]["close"]
        
        # Prepare input features
        last_sequence = self.data.iloc[i-50:i][['RSI', 'MACD', 'SMA', 'BB_Width', '%B', 'Volatility', 'ATR', 'Momentum', 'Price_Change']].values.reshape(1, 50, -1)
        last_features = self.data.iloc[i][['RSI', 'MACD', 'SMA', 'BB_Width', '%B', 'Volatility', 'ATR', 'Momentum', 'Price_Change']].values.reshape(1, -1)

        # Make predictions
        predicted_price = self.lstm_model.predict(last_sequence)[0][0]
        signal = self.xgb_model.predict(last_features)[0]  # 1 = Buy, 0 = Sell

        print(f"Predicted price= ${predicted_price:.2f} | Signal: {'BUY' if signal == 1 else 'SELL'}")

        # Trading Logic
        if signal == 1 and self._cash_balance >= current_price:  # Buy
            self.position = self._cash_balance / current_price
            self._cash_balance = 0
            self.trades.append((i, current_price, "BUY"))

        elif signal == 0 and self.position > 0:  # Sell
            self._cash_balance = self.position * current_price
            self.position = 0
            self.trades.append((i, current_price, "SELL"))

        # Track balance
        portfolio_value = self._cash_balance + (self.position * current_price)
        self.balance_history.append(portfolio_value)

        # Update the graph
        balance_line.set_data(range(len(self.balance_history)), self.balance_history)
        ax.set_xlim(0, len(self.balance_history))  # Adjust X-axis dynamically
        ax.set_ylim(min(self.balance_history), max(self.balance_history))  # Adjust Y-axis

        return balance_line,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(self.data) - 50, interval=100, blit=False)
    
    plt.show()

    final_balance = self._cash_balance + (self.position * self.data.iloc[-1]["close"])
    print(f"Final Balance: ${final_balance:.2f} (Initial: $100000)")
    print(f"Cash: ${self._cash_balance:.2f}, Holdings: {self.position:.2f} shares at ${self.data.iloc[-1]['close']:.2f} per share")

def is_market_open():
    url = f"{os.getenv('APCA_API_BASE_URL')}/v2/clock"
    headers = {
        "APCA-API-KEY-ID": ALPACA_CONFIG['API_KEY'],
        "APCA-API-SECRET-KEY": ALPACA_CONFIG['API_SECRET']
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data["is_open"]  # Returns True if market is open, False otherwise
    else:
        print("Error:", response.json())
        return None

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
    # if is_market_open():
    broker = Alpaca(ALPACA_CONFIG)
    strategy = ML_Trend(broker=broker, selected_symbol=selected_symbol, stock_name=stock_name)
    trader = Trader(logfile="")
    trader.add_strategy(strategy=strategy)
    time.sleep(5)
    trader.run_all()
    # else:
    #     broker = Alpaca(ALPACA_CONFIG)
    #     strategy = ML_Trend(broker=broker, selected_symbol=selected_symbol, stock_name=stock_name)
    #     print("Running backtest...")
    #     if strategy.data is None :
    #         print("No data available for backtesting")
    #     else:
    #         backtester = Backtest(strategy.data, strategy.lstm_model, strategy.xgb_model)
    #         backtester.run()

