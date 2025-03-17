from flask import Flask, request, jsonify
from flask_cors import CORS
from LSTM import ML_Trend 
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load Alpaca API credentials
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_API_SECRET = os.getenv("APCA_API_SECRET_KEY")
ALPACA_API_BASE_URL = os.getenv("APCA_API_BASE_URL")

# Instantiate Alpaca broker (set it up in ML_Trend)
broker = None  # Your Alpaca broker initialization

# Endpoint to perform stock trading strategy
@app.route('/trade', methods=['POST'])
def trade_stock():
    data = request.json
    selected_symbol = data.get('symbol', None)
    
    if not selected_symbol:
        return jsonify({"status": "Symbol is required!"}), 400
    
    # Initialize the trading strategy
    strategy = ML_Trend(broker=broker)
    strategy.selected_symbol = selected_symbol
    
    try:
        # Initialize strategy and start trading
        strategy.initialize()
        strategy.on_trading_iteration()
        return jsonify({"status": "Trade executed successfully!"})
    except Exception as e:
        return jsonify({"status": str(e)}), 500

# Endpoint to backtest the model
@app.route('/backtest', methods=['POST'])
def backtest_stock():
    data = request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    if not start_date or not end_date:
        return jsonify({"status": "Start and End date are required!"}), 400

    # Perform backtesting using YahooDataBacktesting
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Initialize the strategy and run backtest
        strategy = ML_Trend(broker=broker)
        strategy.backtest(start, end)
        return jsonify({"status": "Backtest completed successfully!"})
    except Exception as e:
        return jsonify({"status": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
