import yfinance as yf
import pandas as pd
import numpy as np


TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
           "WIPRO.NS", "MARUTI.NS", "BAJFINANCE.NS", "SUNPHARMA.NS", "ITC.NS"]

def fetch_prices(tickers=TICKERS, start="2015-01-01", end=None):
    """
    Fetch adjusted closing prices for given tickers.
    Returns a DataFrame: rows = dates, columns = tickers
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices = raw["Close"]

    prices = prices.dropna(how="all")
    prices = prices.dropna(axis=1, how="all")

    return prices


def compute_returns(prices: pd.DataFrame, method="log"):
    """
    Compute daily returns.
    method: 'log' for log returns (preferred for quant work), 'simple' for pct_change
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()
    return returns


def compute_rolling_volatility(returns: pd.DataFrame, window=21):
    """
    Rolling annualised volatility. Window=21 ~ 1 trading month.
    """
    return returns.rolling(window).std() * np.sqrt(252)


if __name__ == "__main__":
    prices = fetch_prices()
    prices = prices.dropna(axis=1, how="all")  
    print(f"Fetched prices: {prices.shape}")
    print(prices.tail())

    returns = compute_returns(prices)
    print(f"\nReturns shape: {returns.shape}")
    print(returns.describe())