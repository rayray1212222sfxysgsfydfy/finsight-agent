"""Shared pytest fixtures for FinSight.

Per CLAUDE.md, ALL shared fixtures live here and nowhere else.

Altman Z-Score ground truth (PNC 2023): Z = 0.253 from the CLAUDE.md fixture
values, Z = 0.284 from real EDGAR data (historical market cap = per-year shares ×
year-end close) — NOT 1.847 (that figure was an error) and NOT 0.328 (that used
yfinance's current market cap, since superseded). Both current values fall in the
"distress" zone, which is expected for leveraged banks (see the bank Z-Score
caveat in CLAUDE.md). Do not assert 1.847 anywhere.
"""

import pytest


@pytest.fixture
def pnc_2023_ingest() -> dict:
    """IngestAgent output shape for PNC 2023 (CLAUDE.md fixture values).

    `current_assets`, `current_liabilities`, and `working_capital` are None on
    purpose: banks file unclassified balance sheets, so these fields are
    structurally unavailable from XBRL. `current_ratio` was dropped from the ML
    feature set as a result; the Altman WC/TA term uses WC=0 when None.
    """
    return {
        "revenue": 22_000_000_000.0,
        "net_income": 5_400_000_000.0,
        "total_assets": 557_000_000_000.0,
        "total_liabilities": 504_000_000_000.0,
        "shareholders_equity": 53_000_000_000.0,
        "ebit": 6_100_000_000.0,
        "interest_expense": 4_200_000_000.0,
        "retained_earnings": 29_000_000_000.0,
        "cash": 31_000_000_000.0,
        "market_cap": 55_000_000_000.0,
        # Structurally unavailable for banks — see docstring above.
        "current_assets": None,
        "current_liabilities": None,
        "working_capital": None,
        "raw_text_id": "PNC-2023",
    }
