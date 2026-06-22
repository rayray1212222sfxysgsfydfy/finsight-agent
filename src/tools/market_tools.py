"""Market data tools using yfinance.

This module exposes pure tool functions that fetch market data for a ticker and
normalize it into the expected contract.
"""

from typing import Any, Dict

import yfinance as yf


def fetch_market_data(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """Fetch ticker market data and return normalized market metadata.

    Args:
        ticker: the stock ticker symbol to query.
        period: yfinance history window (e.g. "1y", "10y", "max"). Use a wide
            window to recover year-end prices for historical market-cap
            computation (price x per-year shares outstanding).

    Returns:
        A dictionary containing `data` and `error`.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")

    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        price_history = ticker_obj.history(period=period)
    except Exception as exc:
        return {"data": None, "error": f"Market data fetch failed: {exc}"}

    try:
        history_dict = price_history.to_dict() if hasattr(price_history, "to_dict") else dict(price_history)
    except Exception:
        history_dict = {}

    return {
        "data": {
            "market_cap": info.get("marketCap", None),
            "shares_outstanding": info.get("sharesOutstanding", None),
            "price_history": history_dict,
        },
        "error": None,
    }


__all__ = ["fetch_market_data"]
