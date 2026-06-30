"""Build the processed feature CSV for model training.

Live ingestion (EDGAR / FRED / yfinance via src/tools) mirroring the EDA
notebook's feature-engineering section, persisted to data/processed/features.csv
so that src/models/train.py reads a static, reproducible frame instead of
re-hitting the network. No LLM calls, no agent logic. Run as a script:

    python src/utils/build_features.py

Columns written (feature order matches src/models/feature_names.json):
    ticker, year, debt_to_equity, return_on_assets, interest_coverage,
    revenue_growth, net_margin, altman_z, fedfunds, t10y2y

current_ratio and working_capital are intentionally excluded — banks file
unclassified balance sheets, so they are structurally NaN (see CLAUDE.md).
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.tools.fred_tools import get_macro_snapshot
from src.tools.market_tools import fetch_market_data
from src.tools.sec_tools import fetch_10k_filing

BANKS = [
    "PNC", "JPM", "BK", "WFC", "BAC",           # original 5
    "USB", "TFC", "CFG", "FITB", "KEY",           # added batch 1
    "RF", "HBAN", "MTB", "CMA", "ZION",           # added batch 2
]
YEARS = list(range(2018, 2024))  # 2018-2023 inclusive

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data" / "processed" / "features.csv"

# Feature order is the canonical order in src/models/feature_names.json.
FEATURE_COLUMNS = [
    "debt_to_equity",
    "return_on_assets",
    "interest_coverage",
    "revenue_growth",
    "net_margin",
    "altman_z",
    "fedfunds",
    "t10y2y",
]


def _year_end_closes(price_history: dict) -> Dict[int, Optional[float]]:
    """Map each fiscal year to its last close on/before Dec 31 (tz-naive)."""
    close = price_history.get("Close", {}) if price_history else {}
    if not close:
        return {}
    items = sorted(close.items(), key=lambda kv: kv[0])
    idx = pd.to_datetime([k for k, _ in items], utc=True).tz_convert(None)
    series = pd.Series([v for _, v in items], index=idx).sort_index()
    out: Dict[int, Optional[float]] = {}
    for year in YEARS:
        value = series.asof(pd.Timestamp(f"{year}-12-31"))
        out[year] = float(value) if pd.notna(value) else None
    return out


def build_feature_frame() -> pd.DataFrame:
    """Ingest all bank-years from live sources and engineer the model features.

    Returns:
        a DataFrame with the columns documented in this module's docstring.
        Rows with missing source data keep NaN; train.py drops them.
    """
    closes_by_ticker: Dict[str, Dict[int, Optional[float]]] = {}
    for ticker in BANKS:
        market = fetch_market_data(ticker, period="max")
        if market["error"] is not None:
            print(f"[yfinance ERROR] {ticker}: {market['error']}")
            closes_by_ticker[ticker] = {}
        else:
            closes_by_ticker[ticker] = _year_end_closes(
                market["data"]["price_history"]
            )

    macro_by_year: Dict[int, dict] = {}
    for year in YEARS:
        snapshot = get_macro_snapshot(f"{year}-12-31")
        if snapshot["error"] is not None:
            print(f"[FRED ERROR] {year}: {snapshot['error']}")
            macro_by_year[year] = {}
        else:
            macro_by_year[year] = snapshot["data"]

    rows = []
    for ticker in BANKS:
        for year in YEARS:
            filing = fetch_10k_filing(ticker, year)
            if filing["error"] is not None:
                print(f"[SEC ERROR] {ticker} {year}: {filing['error']}")
                continue
            row = dict(filing["data"])
            row["ticker"] = ticker
            row["year"] = year
            shares = row.get("shares_outstanding")
            close = closes_by_ticker.get(ticker, {}).get(year)
            row["market_cap"] = (
                close * shares if close is not None and shares is not None else None
            )
            macro = macro_by_year.get(year, {})
            row["fedfunds"] = macro.get("FEDFUNDS")
            row["t10y2y"] = macro.get("T10Y2Y")
            rows.append(row)

    df = pd.DataFrame(rows)
    print(
        f"Ingested {len(df)} bank-year rows out of "
        f"{len(BANKS) * len(YEARS)} attempted."
    )

    # Ratios (vectorized). interest_coverage floors the denominator at 0.001.
    df["debt_to_equity"] = df["total_liabilities"] / df["shareholders_equity"]
    df["return_on_assets"] = df["net_income"] / df["total_assets"]
    df["net_margin"] = df["net_income"] / df["revenue"]
    df["interest_coverage"] = df["ebit"] / df["interest_expense"].clip(lower=0.001)

    # Altman Z — WC/TA term uses WC=0 when working_capital is NaN (banks).
    working_capital = df["working_capital"].fillna(0.0)
    df["altman_z"] = (
        1.2 * (working_capital / df["total_assets"])
        + 1.4 * (df["retained_earnings"] / df["total_assets"])
        + 3.3 * (df["ebit"] / df["total_assets"])
        + 0.6 * (df["market_cap"] / df["total_liabilities"])
        + 1.0 * (df["revenue"] / df["total_assets"])
    )

    # Year-over-year revenue growth per ticker (NaN on each ticker's first year).
    df = df.sort_values(["ticker", "year"]).reset_index(drop=True)
    df["revenue_growth"] = df.groupby("ticker")["revenue"].pct_change()

    return df[["ticker", "year"] + FEATURE_COLUMNS]


def save_feature_frame(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    """Write the feature frame to CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    """Build and persist the processed feature CSV."""
    df = build_feature_frame()
    path = save_feature_frame(df)
    print(f"\nWrote {len(df)} rows x {df.shape[1]} cols to {path}")
    print("\nFeature non-null counts:")
    print(df[FEATURE_COLUMNS].notna().sum())
    print("\nHead:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
