"""Federal Reserve Economic Data (FRED) tools.

Fetches time series data from the FRED API and computes macroeconomic snapshots
for a given fiscal year end date.
"""

import datetime
from typing import Any, Dict, List, Optional

import requests

from src.agents.context_schema import FRED_SNAPSHOT_KEYS
from src.utils import config


def fetch_fred_series(
    series_id: str, start_date: str, end_date: str
) -> Dict[str, Any]:
    """Fetch observations for a FRED series within a date range.

    Args:
        series_id: FRED series identifier (e.g. 'FEDFUNDS')
        start_date: start date as YYYY-MM-DD string
        end_date: end date as YYYY-MM-DD string

    Returns:
        {'data': observations list, 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    try:
        if not series_id or not isinstance(series_id, str):
            raise ValueError("series_id must be a non-empty string")
    except (TypeError, ValueError) as exc:
        return {"data": None, "error": str(exc)}

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": config.FRED_API_KEY,
        "observation_start": start_date,
        "observation_end": end_date,
        # FRED defaults to XML; without this, response.json() fails to parse.
        "file_type": "json",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
    except Exception as exc:
        return {"data": None, "error": f"Network error: {exc}"}

    if response.status_code != 200:
        return {"data": None, "error": f"FRED API error: HTTP {response.status_code}"}

    try:
        json_data = response.json()
        if "error_code" in json_data:
            return {"data": None, "error": f"FRED API error: {json_data}"}
        return {"data": json_data.get("observations", []), "error": None}
    except Exception as exc:
        return {"data": None, "error": f"JSON parse error: {exc}"}


def get_macro_snapshot(fiscal_year_end_date: str) -> Dict[str, Any]:
    """Fetch macro snapshot for a given fiscal year end date.

    Returns a dict keyed by FRED series ID with the last observation on or
    before the given fiscal_year_end_date. Respects FRED_SNAPSHOT_KEYS from
    context_schema.py.

    Args:
        fiscal_year_end_date: YYYY-MM-DD string, typically Dec 31 of fiscal year

    Returns:
        {'data': {series_id: value, ...}, 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    try:
        year, month, day = map(int, fiscal_year_end_date.split("-"))
        target_date = datetime.date(year, month, day)
    except (ValueError, IndexError) as exc:
        return {"data": None, "error": f"Invalid date format: {exc}"}

    today = datetime.date.today()
    if target_date > today:
        return {"data": None, "error": f"Date {fiscal_year_end_date} is in the future"}

    snapshot: Dict[str, Any] = {}

    for series_id in FRED_SNAPSHOT_KEYS:
        result = fetch_fred_series(series_id, "1990-01-01", fiscal_year_end_date)

        if result["error"] is not None:
            return {
                "data": None,
                "error": f"Failed to fetch {series_id}: {result['error']}",
            }

        observations: List[Dict[str, str]] = result["data"]
        if not observations:
            snapshot[series_id] = None
            continue

        last_obs = observations[-1]
        try:
            snapshot[series_id] = float(last_obs.get("value", "0"))
        except (ValueError, TypeError):
            snapshot[series_id] = None

    return {"data": snapshot, "error": None}


__all__ = ["fetch_fred_series", "get_macro_snapshot"]
