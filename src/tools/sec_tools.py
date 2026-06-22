"""SEC EDGAR ingestion tools.

Fetches annual financials from EDGAR's structured XBRL "company facts" API
(https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json) rather than
scraping raw submission text. Resolves a ticker to its numeric CIK via EDGAR's
public ticker map, with overrides for tickers that changed.

Bank caveat: banks file unclassified balance sheets, so AssetsCurrent /
LiabilitiesCurrent are typically absent — those fields (and working_capital)
come back as None rather than failing the whole fetch. EBIT is unavailable for
banks (no OperatingIncomeLoss); we fall back to pre-tax income as a proxy.

Tool contract: never raise except ValueError for invalid inputs before I/O.
"""

import datetime
import time
from typing import Any, Dict, List, Optional

import requests

from src.utils import config

# us-gaap tags to try, in priority order, for each logical field.
FIELD_ALIASES: Dict[str, List[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "InterestAndDividendIncomeOperating",
        "SalesRevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "shareholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    # Banks file UNCLASSIFIED balance sheets, so these tags are absent from
    # their XBRL and resolve to None for every bank in our dataset. As a result
    # `current_ratio` was dropped from the ML feature set (feature_names.json)
    # and working_capital is Optional — the Altman WC/TA term uses WC=0 when None.
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "ebit": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseBorrowings",
        "InterestAndDebtExpense",
    ],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndDueFromBanks",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
}

# Fields without which the fetch is considered a failure.
REQUIRED_FIELDS = ("revenue", "net_income", "total_assets")

# Tickers whose EDGAR symbol differs from common usage.
TICKER_CIK_OVERRIDES: Dict[str, int] = {
    "BK": 1390777,  # Bank of New York Mellon — listed under "BNY" in EDGAR's map
}

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Module-level caches so a 30 ticker-year notebook loop downloads each ticker
# map / company-facts payload at most once.
_TICKER_MAP_CACHE: Dict[str, int] = {}
_FACTS_CACHE: Dict[int, dict] = {}


def _clear_caches() -> None:
    """Clear cached ticker map and company facts (used by tests)."""
    _TICKER_MAP_CACHE.clear()
    _FACTS_CACHE.clear()


def _validate_inputs(ticker: str, year: int) -> None:
    """Validate ticker/year, raising ValueError on invalid input."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    if not isinstance(year, int) or isinstance(year, bool):
        raise ValueError("year must be an integer")
    if year > datetime.date.today().year:
        raise ValueError("year must not be in the future")


def _http_get_json(url: str) -> Any:
    """GET a URL with the required SEC User-Agent, retrying 429s.

    Raises:
        RuntimeError: on network failure or non-200 status after retries.
    """
    headers = {"User-Agent": config.SEC_USER_AGENT}
    last_status: Optional[int] = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=15)
        except Exception as exc:
            raise RuntimeError(f"EDGAR request failed: {exc}")
        if response.status_code == 429:
            last_status = 429
            time.sleep(2 ** attempt)
            continue
        if response.status_code != 200:
            raise RuntimeError(
                f"EDGAR request failed: HTTP {response.status_code}"
            )
        return response.json()
    raise RuntimeError(f"EDGAR request failed: HTTP {last_status} after retries")


def _lookup_cik(ticker: str) -> Optional[int]:
    """Resolve a ticker to its numeric CIK, applying known overrides.

    Returns:
        the integer CIK, or None if the ticker cannot be resolved.
    """
    symbol = ticker.strip().upper()
    if symbol in TICKER_CIK_OVERRIDES:
        return TICKER_CIK_OVERRIDES[symbol]

    if not _TICKER_MAP_CACHE:
        payload = _http_get_json(_TICKER_MAP_URL)
        for row in payload.values():
            _TICKER_MAP_CACHE[str(row["ticker"]).upper()] = int(row["cik_str"])
    return _TICKER_MAP_CACHE.get(symbol)


def _load_company_facts(cik: int) -> dict:
    """Fetch (and cache) the full facts block (us-gaap + dei) for a CIK."""
    if cik not in _FACTS_CACHE:
        payload = _http_get_json(_COMPANYFACTS_URL.format(cik=cik))
        _FACTS_CACHE[cik] = payload.get("facts", {})
    return _FACTS_CACHE[cik]


def _select_annual_value(usd_facts: List[dict], year: int) -> Optional[float]:
    """Pick the annual value for `year` from a list of XBRL USD facts.

    Handles both instantaneous (balance sheet) facts and durational (income
    statement) facts, ignoring quarterly periods and preferring 10-K forms.
    """
    candidates: List[dict] = []
    for fact in usd_facts:
        end = str(fact.get("end", ""))
        if end[:4] != str(year):
            continue
        start = fact.get("start")
        if start:  # durational — require roughly a full year, not a quarter
            try:
                d0 = datetime.date.fromisoformat(str(start))
                d1 = datetime.date.fromisoformat(end)
            except ValueError:
                continue
            if (d1 - d0).days < 300:
                continue
        candidates.append(fact)

    if not candidates:
        return None

    # Prefer 10-K filings, then the latest period end, then the latest filing.
    candidates.sort(
        key=lambda f: (
            str(f.get("form", "")).startswith("10-K"),
            str(f.get("end", "")),
            str(f.get("filed", "")),
        )
    )
    try:
        return float(candidates[-1]["val"])
    except (KeyError, TypeError, ValueError):
        return None


def _extract_field(gaap: dict, aliases: List[str], year: int) -> Optional[float]:
    """Extract one logical field by trying each us-gaap alias in order."""
    for tag in aliases:
        block = gaap.get(tag)
        if not block:
            continue
        usd_facts = block.get("units", {}).get("USD")
        if not usd_facts:
            continue
        value = _select_annual_value(usd_facts, year)
        if value is not None:
            return value
    return None


def _select_fiscal_year_value(facts: List[dict], year: int) -> Optional[float]:
    """Pick a value for fiscal year `year`, keyed on `fy` not the end date.

    Used for dei share counts, whose cover-page `end` date falls in the year
    *after* the fiscal year (e.g. FY2023 is reported as of early 2024).
    """
    candidates = [
        f for f in facts
        if f.get("fy") == year and f.get("fp") == "FY"
        and str(f.get("form", "")).startswith("10-K")
    ]
    if not candidates:  # fall back to any fact tagged to this fiscal year
        candidates = [f for f in facts if f.get("fy") == year]
    if not candidates:
        return None
    candidates.sort(key=lambda f: str(f.get("filed", "")))
    try:
        return float(candidates[-1]["val"])
    except (KeyError, TypeError, ValueError):
        return None


def _extract_shares_outstanding(dei: dict, year: int) -> Optional[float]:
    """Extract dei:EntityCommonStockSharesOutstanding for a fiscal year."""
    block = dei.get("EntityCommonStockSharesOutstanding")
    if not block:
        return None
    share_facts = block.get("units", {}).get("shares")
    if not share_facts:
        return None
    return _select_fiscal_year_value(share_facts, year)


def fetch_10k_filing(ticker: str, year: int) -> dict:
    """Fetch annual financials for a ticker/year from EDGAR company facts.

    Args:
        ticker: stock ticker symbol (resolved to a CIK via EDGAR's map)
        year: 4-digit fiscal year, not in the future

    Returns:
        {'data': {<financial fields>, 'raw_text_id': str}, 'error': None} on
        success; {'data': None, 'error': 'message'} on failure. Fields that the
        filer does not report (e.g. current assets for banks) are None.
    """
    _validate_inputs(ticker, year)

    try:
        cik = _lookup_cik(ticker)
        if cik is None:
            return {"data": None, "error": f"No CIK found for ticker '{ticker}'"}
        facts = _load_company_facts(cik)
    except RuntimeError as exc:
        return {"data": None, "error": str(exc)}

    gaap = facts.get("us-gaap", {})
    dei = facts.get("dei", {})

    fields: Dict[str, Optional[float]] = {
        name: _extract_field(gaap, aliases, year)
        for name, aliases in FIELD_ALIASES.items()
    }

    missing = [name for name in REQUIRED_FIELDS if fields.get(name) is None]
    if missing:
        return {
            "data": None,
            "error": f"Missing required financials for {ticker} {year}: {', '.join(missing)}",
        }

    current_assets = fields.get("current_assets")
    current_liabilities = fields.get("current_liabilities")
    if current_assets is not None and current_liabilities is not None:
        working_capital: Optional[float] = current_assets - current_liabilities
    else:
        working_capital = None

    data = dict(fields)
    data["working_capital"] = working_capital
    data["shares_outstanding"] = _extract_shares_outstanding(dei, year)
    data["raw_text_id"] = f"{ticker.strip().upper()}-{year}"
    return {"data": data, "error": None}


__all__ = ["fetch_10k_filing", "FIELD_ALIASES", "TICKER_CIK_OVERRIDES"]
