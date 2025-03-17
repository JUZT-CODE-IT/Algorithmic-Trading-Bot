import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from LSTM import search_assets, restAPI, ALPACA_CONFIG,ML_Trend, Alpaca
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder="build", static_url_path="")
CORS(app, origins="http://localhost:3000")

def calculate_rsi(series, period=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(series, short_period=12, long_period=26, signal_period=9):
    """Calculate the MACD and Signal Line."""
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    return macd, signal, macd - signal

# Serve React app (index.html) when visiting the root URL
@app.route('/')
def serve_react():
    return send_from_directory(os.path.join(app.root_path, 'build'), 'index.html')

# API route for stock predictions
@app.route('/predict', methods=['GET'])
def predict():
    predictions = {
        "AAPL": {"price": 150.5, "signal": "BUY"},
        "TSLA": {"price": 210.8, "signal": "SELL"},
        "GOOGL": {"price": 2800.3, "signal": "BUY"},
    }
    return jsonify(predictions)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '').strip()
    if query:
        results = search_assets(query)  # Get results from lstm.py
        # Return only the symbols
        return jsonify(results)
    else:
        return jsonify({"error": "Query parameter is required."}), 400

broker = Alpaca(ALPACA_CONFIG)
strategy = ML_Trend(broker=broker)

# @app.route('/trade', methods=['POST'])
# def trade():
#     data = request.json
#     symbol = data.get("symbol")
    
#     if not symbol:
#         return jsonify({"error": "Stock symbol is required"}), 400

#     try:
#         # Fetch historical stock data
#         bars = strategy.get_historical_prices(symbol, 730, "day")
#         gld = bars.df
#         gld['7_day_ma'] = gld['close'].rolling(7).mean()
#         gld['21_day_ma'] = gld['close'].rolling(21).mean()
#         gld['RSI'] = strategy.calculate_rsi(gld['close'])
#         gld['MACD'], _, _ = strategy.calculate_macd(gld['close'])
#         gld = gld.dropna()

#         if gld.empty:
#             return jsonify({"error": "Insufficient data for trading"}), 400

#         # Prepare model features
#         features = gld[['7_day_ma', '21_day_ma', 'RSI', 'MACD']].iloc[-1].values.reshape(1, -1)
#         prediction = strategy.model.predict(features)

#         quantity = 200
#         trade_action = None

#         if prediction == 1:  # Buy signal
#             pos = strategy.get_position(symbol)
#             if pos is None:
#                 order = strategy.create_order(symbol, quantity, "buy")
#                 strategy.submit_order(order)
#                 trade_action = "BUY"

#         elif prediction == 0:  # Sell signal
#             pos = strategy.get_position(symbol)
#             if pos is not None:
#                 strategy.sell_all()
#                 trade_action = "SELL"

#         if trade_action:
#             return jsonify({"status": f"Trade executed: {trade_action} {symbol}"})
#         else:
#             return jsonify({"status": "No trade executed"})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)
