import alpaca_trade_api as tradeapi
API_KEY = 'PKYA2RUW3CFL01GRDM6G'
API_SECRET = 'Ts8JQ6KKdn4S34WT01ovPiIhtnGAssokMfzVvY42'
ALPACA_CONFIG = {
    'API_KEY' : 'PKYA2RUW3CFL01GRDM6G',
    'API_SECRET' : 'Ts8JQ6KKdn4S34WT01ovPiIhtnGAssokMfzVvY42',
    "ENDPOINT": "https://paper-api.alpaca.markets/"
}

api = tradeapi.REST(
    ALPACA_CONFIG["API_KEY"], 
    ALPACA_CONFIG["API_SECRET"], 
    "https://paper-api.alpaca.markets/"
)
if __name__=="__main__":
    try:
        account = api.get_account()
        print(account)
    except tradeapi.rest.APIError as e:
        print("Error:", e)