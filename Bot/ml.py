from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime, timedelta
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader
import alpaca_trade_api as tradeapi 
from alpaca.trading.client import TradingClient
import numpy as np
from dotenv import load_dotenv
import os



class ML_Trend(Strategy):
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

    def initialize(self):
        self.sleeptime = "1D"
        self.model = RandomForestClassifier(n_estimators=100)

        # Load historical data
        self.data = self.get_historical_prices("GOOG", 51, "day").df
        self.data['7_day_ma'] = self.data['close'].rolling(7).mean()
        self.data['21_day_ma'] = self.data['close'].rolling(21).mean()
        self.data['RSI'] = self.calculate_rsi(self.data['close'])
        self.data['MACD'], _, _ = self.calculate_macd(self.data['close'])

        # Define target labels (Price Movement)
        self.data['Price_Movement'] = (self.data['close'].shift(-1) > self.data['close']).astype(int)

        # Drop rows with NaN values (from rolling calculations)
        self.data = self.data.dropna()

        # Prepare features and labels
        self.features = self.data[['7_day_ma', '21_day_ma', 'RSI', 'MACD']]
        self.labels = self.data['Price_Movement']

        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        print("Accuracy: ", accuracy_score(y_test, predictions))

    def on_trading_iteration(self):
        # Use the model to predict price movement
        bars = self.get_historical_prices("GOOG", 21, "day")
        gld = bars.df
        gld['7_day_ma'] = gld['close'].rolling(7).mean()
        gld['21_day_ma'] = gld['close'].rolling(21).mean()
        gld['RSI'] = self.calculate_rsi(gld['close'])
        gld['MACD'], _, _ = self.calculate_macd(gld['close'])
        gld = gld.dropna()

        # Prepare the latest features
        features = gld[['7_day_ma', '21_day_ma', 'RSI', 'MACD']].iloc[-1].values.reshape(1, -1)
        prediction = self.model.predict(features)

        symbol = "GOOG"
        quantity = 200

        if prediction == 1:  # Buy
            pos = self.get_position(symbol)
            if pos is None:
                order = self.create_order(symbol, quantity, "buy")
                self.submit_order(order)
        elif prediction == 0:  # Sell
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()

if __name__ == "__main__":

    load_dotenv("setup.env") #Loading .env file for confidential info.

    ALPACA_CONFIG = {
    'API_KEY' : os.getenv("APCA_API_KEY_ID"),
    'API_SECRET' : os.getenv("APCA_API_SECRET_KEY"),
    "ENDPOINT": os.getenv("APCA_API_BASE_URL")
    } #saving everything in one variable
     
    trade = True #starting trade
    if trade:
        broker = Alpaca(ALPACA_CONFIG) #Setting up Alpaca.
        try:
            strategy = ML_Trend(broker=broker)
        except tradeapi.rest.APIError as e:
            print("Error:",e)
        trader=Trader(logfile="")
        # trader = TradingClient(api_key=os.getenv("APCA_API_KEY_ID"),secret_key=os.getenv("APCA_API_SECRET_KEY"),oauth_token=None)
        trader.add_strategy(strategy=strategy)
        trader.run_all()
    else:
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        ML_Trend.backtest(
            YahooDataBacktesting,
            start,
            end,
        )