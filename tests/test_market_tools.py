import pytest

from src.tools.market_tools import fetch_market_data


class DummyHistory:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


class DummyTicker:
    def __init__(self, info, history_data):
        self.info = info
        self._history_data = history_data
        self.requested_period = None

    def history(self, period="1y"):
        self.requested_period = period
        return self._history_data


def test_fetch_market_data_success(mocker):
    dummy_info = {"marketCap": 1234567890, "sharesOutstanding": 1000000}
    dummy_history = DummyHistory({"Close": [100.0, 102.0]})
    mocker.patch("src.tools.market_tools.yf.Ticker", return_value=DummyTicker(dummy_info, dummy_history))

    result = fetch_market_data("PNC")

    assert result["error"] is None
    assert isinstance(result["data"], dict)
    assert result["data"]["market_cap"] == 1234567890
    assert result["data"]["shares_outstanding"] == 1000000
    assert result["data"]["price_history"] == {"Close": [100.0, 102.0]}


def test_fetch_market_data_passes_period_to_history(mocker):
    # Historical market cap needs multi-year history; the period must pass through.
    dummy = DummyTicker({"marketCap": 1}, DummyHistory({"Close": [1.0]}))
    mocker.patch("src.tools.market_tools.yf.Ticker", return_value=dummy)

    fetch_market_data("PNC", period="max")

    assert dummy.requested_period == "max"


def test_fetch_market_data_invalid_ticker_raises_value_error():
    with pytest.raises(ValueError, match="ticker"):
        fetch_market_data("")


def test_fetch_market_data_handles_missing_market_cap_gracefully(mocker):
    dummy_info = {}
    dummy_history = DummyHistory({"Close": [100.0]})
    mocker.patch("src.tools.market_tools.yf.Ticker", return_value=DummyTicker(dummy_info, dummy_history))

    result = fetch_market_data("PNC")

    assert result["error"] is None
    assert result["data"]["market_cap"] is None
    assert result["data"]["shares_outstanding"] == None or result["data"].get("shares_outstanding") is None


def test_fetch_market_data_network_exception_returns_error(mocker):
    mocker.patch("src.tools.market_tools.yf.Ticker", side_effect=Exception("yfinance error"))

    result = fetch_market_data("PNC")

    assert result["data"] is None
    assert isinstance(result["error"], str)
    assert "yfinance" in result["error"].lower() or "error" in result["error"].lower()
