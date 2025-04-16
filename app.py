import os
import sys
import asyncio
import joblib
import json
import threading
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import alpaca_trade_api as api
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from flask_cors import CORS,cross_origin
from concurrent.futures import ThreadPoolExecutor
from alpaca_trade_api.rest import REST
from flask import Flask, request, jsonify
from LSTM import restAPI,ALPACA_CONFIG,ML_Trend
from alpaca.trading.stream import TradingStream
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from lumibot.entities import Order
from alpaca.trading.client import TradingClient
from flask import session
from alpaca.data.historical import StockHistoricalDataClient


selected_symbol=None
stock_name=None
try:
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("Model loading failed. Ensure the models exist.", e)
    sys.exit(1)


app = Flask(__name__, static_folder="build", static_url_path="")
CORS(app, supports_credentials=True, origins=["*"])
app.secret_key = 'finalyearproject@batch35'
client = StockHistoricalDataClient(ALPACA_CONFIG['API_KEY'], ALPACA_CONFIG['API_SECRET'])
restAPI = REST(ALPACA_CONFIG['API_KEY'], ALPACA_CONFIG['API_SECRET'], ALPACA_CONFIG['BASE_URL'])

try:
    account = restAPI.get_account()
except Exception as e:
    print(f"Authentication failed: {e}")

broker = Alpaca(ALPACA_CONFIG)
trading_ws = None
trading_client = TradingClient(
    api_key=ALPACA_CONFIG['API_KEY'],
    secret_key= ALPACA_CONFIG['API_SECRET'],
    paper=True
)
 # Converts to dictionary (v2 replacement for .dict())

def get_historical_prices(symbol, days=1000, timeframe='day'):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    timeframe_map = {
        'minute': TimeFrame.Minute,
        'hour': TimeFrame.Hour,
        'day': TimeFrame.Day,
        'week': TimeFrame.Week,
        'month': TimeFrame.Month
    }

    if timeframe not in timeframe_map:
        raise ValueError(f"Invalid timeframe: {timeframe}. Use one of {list(timeframe_map.keys())}")

    alpaca_timeframe = timeframe_map[timeframe]

    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=alpaca_timeframe,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        feed="iex"  # or remove this line to use the default
    )

    barset = client.get_stock_bars(request_params)

    # Safely extract symbol data
    symbol_data = barset.data.get(symbol)
    if not symbol_data:
        print(f"No data returned for {symbol}")
        return {}

    data_dict = {
        'time': [bar.timestamp for bar in symbol_data],
        'open': [bar.open for bar in symbol_data],
        'high': [bar.high for bar in symbol_data],
        'low': [bar.low for bar in symbol_data],
        'close': [bar.close for bar in symbol_data],
        'volume': [bar.volume for bar in symbol_data]
    }

    return {symbol: data_dict}

@app.route('/search', methods=['GET'])
@cross_origin(origins=['http://localhost:3000','http://localhost:5000'], supports_credentials=True)
def search_assets():
    query = request.args.get('query', '').lower()
    try:
        tradable_assets = [asset for asset in restAPI.list_assets() if asset.tradable]
        results = {asset.symbol: asset.name for asset in tradable_assets if query in asset.name.lower()}

        if not results:
            return jsonify({"message": "No matching stocks found", "results": {}})
        
        return jsonify(results)
    except Exception as e:
        print(f"Error fetching assets: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

async def connect_with_retry():
    global trading_ws
    max_retries = 5
    attempt = 0

    while attempt < max_retries:
        try:
            if trading_ws is None:
                trading_ws = TradingStream(
                    api_key=ALPACA_CONFIG['API_KEY'],
                    secret_key=ALPACA_CONFIG['API_SECRET'],
                    paper=True
                )
            await trading_ws.subscribe_trade_updates(my_trade_handler)
            await trading_ws.run()
            break  # Exit loop if connection is successful
        except Exception as e:
            print(f"⚠️ WebSocket Error: {e}, retrying in {2**attempt * 5} seconds...")
            await asyncio.sleep(min((2 ** attempt) * 5, 60))
            attempt += 1
    print("Max retries reached. Could not connect to WebSocket.")

async def my_trade_handler(data):
    print(f"Trade update received: {data}")
    if data.get("event") == "fill":
        print(f"Order filled! Symbol: {data['order']['symbol']}, Price: {data['order']['filled_avg_price']}")
    elif data.get("event") == "partial_fill":
        print(f"Partial fill: {data['order']['filled_qty']} shares of {data['order']['symbol']}")

async def websocket_task():
    global trading_ws
    while True:
        try:
            if trading_ws is None:
                trading_ws = TradingStream(
                    api_key=ALPACA_CONFIG['API_KEY'],
                    secret_key=ALPACA_CONFIG['API_SECRET'],
                    paper=True
                )
            await trading_ws.subscribe_trade_updates(my_trade_handler)
            await trading_ws.run()
        except Exception as e:
            print(f"⚠️ WebSocket Error: {e}, retrying in 5 seconds...")
            await asyncio.sleep(5)

def start_websocket():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(websocket_task())  
@app.route('/trade', methods=['POST'])
@cross_origin(origins=['http://localhost:3000','http://localhost:5000'], supports_credentials=True)
def trade():
    data = request.json
    selected_symbol = data.get("symbol")
    stock_name = data.get("name")
    session["symbol"] = selected_symbol
    selected_symbol = session.get("symbol")
    session["name"] = stock_name
    stock_name = session.get("name")
    if not selected_symbol:
        return jsonify({"error": "Symbol is required"}), 400

    alpaca_data = get_historical_prices(selected_symbol)
    if isinstance(alpaca_data, dict) and alpaca_data:
        symbol_data = next(iter(alpaca_data.values()))  # first list inside the dict
        df = pd.DataFrame(symbol_data)
        df['time'] = df['time'].astype(str)
    else:
        return jsonify({"error": "No valid data returned from Alpaca"}), 500
    print("Dataframe Conversion:",selected_symbol,df.head(),sep='\n') 
     # This will still show the DataFrame view
    return jsonify(df.to_dict(orient="records"))

@app.route("/predict", methods=['POST'])
@cross_origin(origins=['http://localhost:3000','http://localhost:5000'], supports_credentials=True)
def predict():
    raw_json = request.get_json() # Wrap data in a list to form DataFrame
    selected_symbol = session.get("symbol")
    stock_name = session.get("name")
    if not raw_json:
        return jsonify({"error": "No data received"}), 400
    if isinstance(raw_json, str):  
        try:
            raw_json = json.loads(raw_json)  # Convert JSON string to Python object
        except json.JSONDecodeError as e:
            return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400
    parsed_data=raw_json
    df = pd.DataFrame(parsed_data)
    strategy = ML_Trend(broker=broker, selected_symbol=selected_symbol, stock_name=stock_name,data=df)
    strategy.data=df
    trader = Trader(logfile="")
    trader.add_strategy(strategy=strategy)

    # Run the trader in a separate thread to avoid signal issues
    def run_trader():
        trader.run_all()

    trader_thread = threading.Thread(target=run_trader)
    trader_thread.start()
    print("raw data converted to df",df.head(),sep="\n")
    
    return jsonify(df.head().to_dict(orient="records"))

def run_startegy(selected_symbol, stock_name):
    try:
        strategy = ML_Trend(broker=broker, selected_symbol=selected_symbol, stock_name=stock_name)
        strategy.run()
    except Exception as e:
        print(f"Error running strategy: {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5000)