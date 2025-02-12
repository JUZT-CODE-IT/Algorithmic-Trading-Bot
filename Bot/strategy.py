import datetime as dt
from dateutil.relativedelta import relativedelta
import os
import pandas as pd
import numpy as np
import quantstats as qs
import webbrowser as web
import yfinance as yf

def ma_cross_strategy(ticker, slow=200, fast=50, end=None, period=3):
    if not end:
        end = dt.date.today()
    start = end - relativedelta(years=period)

    # Download the data
    data = yf.download(ticker, start=start, end=end)[["Close"]]

    # Check if data was downloaded
    if data.empty:
        raise ValueError(f"No data retrieved for ticker {ticker}. Check the ticker symbol and date range.")

    # Calculate moving averages
    data[f'{fast}-day'] = data['Close'].rolling(fast).mean()
    data[f'{slow}-day'] = data['Close'].rolling(slow).mean()

    # Calculate returns and strategy signals
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['position'] = np.where(data[f'{fast}-day'] > data[f'{slow}-day'], 1, -1)
    data['strategy'] = data['position'].shift(1) * data['returns']  # Apply the strategy with a shift

    # Return only the strategy returns as a Series, removing any NaN values
    strategy_returns = data['strategy'].dropna()
    return strategy_returns

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Run strategy
gld_cross = ma_cross_strategy("GLD", slow=21, fast=9, period=3)
gld_cross.index = gld_cross.index.tz_localize(None)  # Remove timezone

# Download benchmark returns and remove timezone information
gld = qs.utils.download_returns("GLD", period='3y')
gld.index = gld.index.tz_localize(None)  # Remove timezone
gld = gld[gld_cross.index[0]:]  # Align start date with strategy

# Generate the QuantStats report
qs.extend_pandas()
qs.reports.html(gld_cross, benchmark=gld, output="output/gld_cross.html", download_filename="output/gld_cross.html")

# Open the generated HTML file
web.open_new(f"file:///{os.getcwd()}/output/gld_cross.html")
