import datetime

import pytest

from src.agents.context_schema import FRED_SNAPSHOT_KEYS
from src.tools.fred_tools import fetch_fred_series, get_macro_snapshot


class DummyResponse:
    def __init__(self, status_code: int, json_data: dict = None, text: str = ""):
        self.status_code = status_code
        self.json_data = json_data or {}
        self.text = text

    def json(self):
        return self.json_data


def test_fetch_fred_series_success(mocker):
    fake_response = DummyResponse(
        200,
        {
            "observations": [
                {"date": "2023-12-31", "value": "5.33"},
                {"date": "2024-01-31", "value": "5.25"},
            ]
        },
    )
    mocker.patch("src.tools.fred_tools.requests.get", return_value=fake_response)

    result = fetch_fred_series("FEDFUNDS", "2023-01-01", "2023-12-31")

    assert result["error"] is None
    assert isinstance(result["data"], list)
    assert len(result["data"]) > 0


def test_fetch_fred_series_requests_json_file_type(mocker):
    # FRED defaults to XML; without file_type=json, response.json() fails.
    captured = {}

    def fake_get(url: str, params: dict = None, **kwargs):
        captured["params"] = params
        return DummyResponse(200, {"observations": [{"date": "2023-12-31", "value": "5.33"}]})

    mocker.patch("src.tools.fred_tools.requests.get", side_effect=fake_get)

    fetch_fred_series("FEDFUNDS", "2023-01-01", "2023-12-31")

    assert captured["params"].get("file_type") == "json"


def test_fetch_fred_series_invalid_series_returns_error(mocker):
    fake_response = DummyResponse(400, {"error_code": 400})
    mocker.patch("src.tools.fred_tools.requests.get", return_value=fake_response)

    result = fetch_fred_series("INVALID_SERIES", "2023-01-01", "2023-12-31")

    assert result["data"] is None
    assert isinstance(result["error"], str)


def test_fetch_fred_series_network_error_returns_error(mocker):
    mocker.patch(
        "src.tools.fred_tools.requests.get",
        side_effect=Exception("Network error"),
    )

    result = fetch_fred_series("FEDFUNDS", "2023-01-01", "2023-12-31")

    assert result["data"] is None
    assert isinstance(result["error"], str)
    assert "Network" in result["error"] or "error" in result["error"].lower()


def test_get_macro_snapshot_returns_all_fred_keys(mocker):
    # Mock responses for all 5 FRED series
    def mock_get(url: str, params: dict = None, **kwargs):
        series_id = params.get("series_id")
        value = {"FEDFUNDS": "5.33", "CPIAUCSL": "306.7", "UNRATE": "3.8", "GS10": "4.2", "T10Y2Y": "1.1"}.get(
            series_id, "0"
        )
        return DummyResponse(
            200,
            {
                "observations": [
                    {"date": "2023-12-31", "value": value},
                ]
            },
        )

    mocker.patch("src.tools.fred_tools.requests.get", side_effect=mock_get)

    result = get_macro_snapshot("2023-12-31")

    assert result["error"] is None
    assert result["data"] is not None
    for key in FRED_SNAPSHOT_KEYS:
        assert key in result["data"]


def test_get_macro_snapshot_future_date_returns_error(mocker):
    future_date = (datetime.date.today() + datetime.timedelta(days=365)).isoformat()

    result = get_macro_snapshot(future_date)

    assert result["data"] is None
    assert isinstance(result["error"], str)
    assert "future" in result["error"].lower() or "date" in result["error"].lower()
