import datetime

import pytest

from src.tools import sec_tools
from src.tools.sec_tools import fetch_10k_filing

FAKE_TICKER_MAP = {
    "0": {"cik_str": 713676, "ticker": "PNC", "title": "PNC FINANCIAL SERVICES GROUP, INC."},
    "1": {"cik_str": 19617, "ticker": "JPM", "title": "JPMORGAN CHASE & CO"},
}

PRETAX_TAG = (
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
)


def _annual(val: float) -> dict:
    """Durational (income statement) fact: full-year period plus a prior year."""
    return {
        "units": {
            "USD": [
                {"start": "2023-01-01", "end": "2023-12-31", "val": val, "form": "10-K", "fy": 2023, "fp": "FY", "filed": "2024-02-28"},
                {"start": "2022-01-01", "end": "2022-12-31", "val": val * 0.9, "form": "10-K", "fy": 2022, "fp": "FY", "filed": "2023-02-28"},
                # a quarterly fact that must be ignored for the annual figure
                {"start": "2023-10-01", "end": "2023-12-31", "val": val / 4, "form": "10-Q", "fy": 2023, "fp": "Q4", "filed": "2024-01-15"},
            ]
        }
    }


def _instant(val: float) -> dict:
    """Instantaneous (balance sheet) fact: point-in-time at fiscal year end."""
    return {
        "units": {
            "USD": [
                {"end": "2023-12-31", "val": val, "form": "10-K", "fy": 2023, "fp": "FY", "filed": "2024-02-28"},
                {"end": "2022-12-31", "val": val * 0.9, "form": "10-K", "fy": 2022, "fp": "FY", "filed": "2023-02-28"},
            ]
        }
    }


def _shares(val: float, fy: int) -> dict:
    """dei share-count fact: cover-page count whose `end` falls in fy+1."""
    return {
        "units": {
            "shares": [
                {"end": f"{fy + 1}-02-05", "val": val, "form": "10-K", "fy": fy, "fp": "FY", "filed": f"{fy + 1}-02-28"},
                {"end": f"{fy}-02-05", "val": val * 1.01, "form": "10-K", "fy": fy - 1, "fp": "FY", "filed": f"{fy}-02-28"},
            ]
        }
    }


def _fake_companyfacts(include_current: bool = True) -> dict:
    gaap = {
        "Revenues": _annual(22_000_000_000),
        "NetIncomeLoss": _annual(5_400_000_000),
        "Assets": _instant(557_000_000_000),
        "Liabilities": _instant(504_000_000_000),
        "StockholdersEquity": _instant(53_000_000_000),
        PRETAX_TAG: _annual(6_100_000_000),
        "InterestExpense": _annual(4_200_000_000),
        "RetainedEarningsAccumulatedDeficit": _instant(29_000_000_000),
        "CashAndDueFromBanks": _instant(31_000_000_000),
    }
    if include_current:
        gaap["AssetsCurrent"] = _instant(120_000_000_000)
        gaap["LiabilitiesCurrent"] = _instant(30_000_000_000)
    # dei share count is selected by fiscal year (fy), since its cover-page
    # `end` date falls in the following calendar year.
    dei = {"EntityCommonStockSharesOutstanding": _shares(397_808_112, 2023)}
    return {"facts": {"us-gaap": gaap, "dei": dei}}


class DummyResponse:
    def __init__(self, status_code: int, json_data: dict = None, text: str = ""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


def _make_fake_get(companyfacts: dict, facts_status: int = 200, capture: dict = None):
    def fake_get(url: str, headers: dict = None, timeout: int = None, **kwargs):
        assert headers and headers.get("User-Agent"), "SEC requests require a User-Agent"
        if "company_tickers.json" in url:
            return DummyResponse(200, FAKE_TICKER_MAP)
        if "companyfacts" in url:
            if capture is not None:
                capture["companyfacts_url"] = url
            return DummyResponse(facts_status, companyfacts, text="error body")
        raise AssertionError(f"unexpected URL: {url}")

    return fake_get


@pytest.fixture(autouse=True)
def _clear_sec_caches():
    """Reset module-level caches so each test is isolated."""
    sec_tools._clear_caches()
    yield
    sec_tools._clear_caches()


def test_fetch_10k_filing_success(monkeypatch):
    monkeypatch.setattr(
        "src.tools.sec_tools.requests.get", _make_fake_get(_fake_companyfacts())
    )

    result = fetch_10k_filing("PNC", 2023)

    assert result["error"] is None
    data = result["data"]
    assert data["revenue"] == 22_000_000_000.0
    assert data["net_income"] == 5_400_000_000.0
    assert data["total_assets"] == 557_000_000_000.0
    assert data["total_liabilities"] == 504_000_000_000.0
    assert data["shareholders_equity"] == 53_000_000_000.0
    assert data["ebit"] == 6_100_000_000.0  # pretax-income proxy
    assert data["interest_expense"] == 4_200_000_000.0
    assert data["retained_earnings"] == 29_000_000_000.0
    assert data["cash"] == 31_000_000_000.0
    assert data["current_assets"] == 120_000_000_000.0
    assert data["current_liabilities"] == 30_000_000_000.0
    assert data["working_capital"] == 90_000_000_000.0  # current_assets - current_liabilities
    assert data["shares_outstanding"] == 397_808_112.0  # selected by fiscal year, not end date
    assert data["raw_text_id"] == "PNC-2023"


def test_fetch_10k_filing_selects_shares_by_fiscal_year(monkeypatch):
    # dei share count's `end` (cover date) falls in 2024, but it is the FY2023
    # figure — selection must key on fy, not the end-date year.
    monkeypatch.setattr(
        "src.tools.sec_tools.requests.get", _make_fake_get(_fake_companyfacts())
    )

    result = fetch_10k_filing("PNC", 2023)

    assert result["error"] is None
    assert result["data"]["shares_outstanding"] == 397_808_112.0


def test_fetch_10k_filing_missing_shares_returns_none(monkeypatch):
    facts = _fake_companyfacts()
    del facts["facts"]["dei"]  # no dei taxonomy at all
    monkeypatch.setattr(
        "src.tools.sec_tools.requests.get", _make_fake_get(facts)
    )

    result = fetch_10k_filing("PNC", 2023)

    assert result["error"] is None  # missing shares is not fatal
    assert result["data"]["shares_outstanding"] is None


def test_fetch_10k_filing_invalid_ticker_raises_value_error():
    with pytest.raises(ValueError, match="ticker"):
        fetch_10k_filing("", 2023)


def test_fetch_10k_filing_future_year_raises_value_error():
    future_year = datetime.datetime.now().year + 1
    with pytest.raises(ValueError, match="year"):
        fetch_10k_filing("PNC", future_year)


def test_fetch_10k_filing_unknown_ticker_returns_error(monkeypatch):
    monkeypatch.setattr(
        "src.tools.sec_tools.requests.get", _make_fake_get(_fake_companyfacts())
    )

    result = fetch_10k_filing("ZZZZ", 2023)

    assert result["data"] is None
    assert isinstance(result["error"], str)
    assert "CIK" in result["error"] or "ZZZZ" in result["error"]


def test_fetch_10k_filing_bank_without_current_items_returns_none(monkeypatch):
    # Banks use unclassified balance sheets — no AssetsCurrent/LiabilitiesCurrent.
    monkeypatch.setattr(
        "src.tools.sec_tools.requests.get",
        _make_fake_get(_fake_companyfacts(include_current=False)),
    )

    result = fetch_10k_filing("PNC", 2023)

    assert result["error"] is None  # missing current items is NOT fatal
    assert result["data"]["current_assets"] is None
    assert result["data"]["current_liabilities"] is None
    assert result["data"]["working_capital"] is None
    # core fields still present
    assert result["data"]["total_assets"] == 557_000_000_000.0


def test_fetch_10k_filing_companyfacts_http_error_returns_error(monkeypatch):
    monkeypatch.setattr(
        "src.tools.sec_tools.requests.get",
        _make_fake_get(_fake_companyfacts(), facts_status=500),
    )

    result = fetch_10k_filing("PNC", 2023)

    assert result["data"] is None
    assert isinstance(result["error"], str)
    assert "500" in result["error"] or "EDGAR" in result["error"]


def test_fetch_10k_filing_uses_cik_override_for_bk(monkeypatch):
    # BK (BNY Mellon) is listed under ticker "BNY" in EDGAR's map; an override
    # maps the requested "BK" to CIK 1390777.
    capture = {}
    monkeypatch.setattr(
        "src.tools.sec_tools.requests.get",
        _make_fake_get(_fake_companyfacts(), capture=capture),
    )
    assert sec_tools.TICKER_CIK_OVERRIDES.get("BK") == 1390777

    result = fetch_10k_filing("BK", 2023)

    assert result["error"] is None
    assert "CIK0001390777" in capture["companyfacts_url"]
