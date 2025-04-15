import os
from dotenv import load_dotenv
import requests

# Load API keys from .env
load_dotenv("setup.env")

ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

BASE_URL = "https://paper-api.alpaca.markets"

# headers = {
#     "APCA-API-KEY-ID": ALPACA_API_KEY,
#     "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
# }

# response = requests.get(f"{BASE_URL}/v2/account", headers=headers)
# print(response.json())  # Check if you get a valid response


import alpaca_trade_api as tradeapi

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")
barset = api.get_barset('AAPL', 'day', limit=5)  # Example fetch
print(barset)  # Debugging: Ensure this isn't empty
