# IngestAgent Skill

## Purpose
Fetch raw financial data for a given bank ticker and fiscal year from EDGAR (SEC), FRED, and yfinance. Produce a structured ingest context dict that downstream agents (AnalysisAgent, RiskAgent) depend on. Never call other agents. Never make LLM calls inside tool functions.

---

## Tools Available

```python
fetch_10k_filing(ticker: str, year: int) -> dict
    # Returns {"data": {...}, "error": str | None}
    # data keys: revenue, net_income, total_assets, total_liabilities,
    #   shareholders_equity, current_assets, current_liabilities, working_capital,
    #   ebit, interest_expense, retained_earnings, cash, shares_outstanding
    # current_assets / current_liabilities / working_capital are None for banks
    #   (unclassified balance sheets — see CLAUDE.md CAVEAT)

fetch_market_data(ticker: str, period: str = "max") -> dict
    # Returns {"data": {"market_cap": float | None, "price_history": dict}, "error": str | None}
    # market_cap is None intermittently — always handle None

get_macro_snapshot(date_str: str) -> dict
    # Returns {"data": {"FEDFUNDS": float, "CPIAUCSL": float, "UNRATE": float,
    #                   "GS10": float, "T10Y2Y": float}, "error": str | None}
    # date_str format: "YYYY-MM-DD" (fiscal year end, e.g. "2023-12-31")
```

---

## Input Schema

Receives `context` dict with at minimum:
```python
{
    "ticker": str,   # e.g. "PNC"
    "year": int,     # e.g. 2023
    "run_id": str,   # unique run identifier
}
```

---

## Output Contract

Writes results under `context["ingest"]`. Every key must be present — `None` is allowed, missing is not.

```python
context["ingest"] = {
    "revenue": float,
    "net_income": float,
    "total_assets": float,
    "total_liabilities": float,
    "shareholders_equity": float,
    "current_assets": float | None,       # None for banks (structural)
    "current_liabilities": float | None,  # None for banks (structural)
    "working_capital": float | None,      # None for banks (structural)
    "ebit": float,
    "interest_expense": float,
    "retained_earnings": float,
    "cash": float,
    "shares_outstanding": float | None,
    "market_cap": float | None,
    "fred_snapshot": {
        "FEDFUNDS": float, "CPIAUCSL": float, "UNRATE": float,
        "GS10": float, "T10Y2Y": float
    },
    "raw_text_id": str,   # cache key written to SQLite via db_tools
}
```

Required keys before handing off to AnalysisAgent (INGEST_TO_ANALYSIS_REQUIRED):
`ticker`, `year`, `revenue`, `net_income`, `total_assets`

---

## Step-by-Step Behavior

1. Check cache via `check_cache(f"{ticker}_{year}_ingest")`. If hit, return cached result and skip all API calls.
2. Call `fetch_10k_filing(ticker, year)`. Check `result["error"]` before accessing `result["data"]`. On error, raise `AgentExecutionError`.
3. Call `fetch_market_data(ticker, period="max")`. Extract year-end close from `price_history` and multiply by `shares_outstanding` to compute historical `market_cap`. If market data errors or `shares_outstanding` is None, set `market_cap = None` (non-fatal).
4. Call `get_macro_snapshot(f"{year}-12-31")`. On error, set `fred_snapshot = {}` and log warning — do not abort.
5. Assemble `context["ingest"]` with all fields. Set `raw_text_id` to `f"{ticker}_{year}_10k"`.
6. Write assembled dict to cache via `write_to_db(raw_text_id, context["ingest"])`.
7. Validate that INGEST_TO_ANALYSIS_REQUIRED keys are present and non-None. Raise `AgentExecutionError` if any are missing.

---

## Error Handling

- Tool errors: always check `result["error"]` before `result["data"]`.
- EDGAR failures: raise `AgentExecutionError(f"IngestAgent: SEC fetch failed for {ticker} {year}: {error}")`.
- Market data failures: log and continue with `market_cap = None`.
- FRED failures: log and continue with `fred_snapshot = {}`.
- Missing required fields after assembly: raise `AgentExecutionError`.

---

## Forbidden Behaviors

- Do not call other agents.
- Do not access `result["data"]` without checking `result["error"]` first.
- Do not hardcode API keys — all credentials come from `config.py`.
- Do not use `print()` — use the structured logger.
- Do not make real API calls in tests — mock all tool functions.

---

## Worked Example — PNC 2023

Input:
```python
context = {"ticker": "PNC", "year": 2023, "run_id": "abc123"}
```

Expected `context["ingest"]` after run:
```python
{
    "revenue": 22_000_000_000.0,         # ~22B
    "net_income": 5_400_000_000.0,       # ~5.4B
    "total_assets": 557_000_000_000.0,   # ~557B
    "total_liabilities": 504_000_000_000.0,
    "shareholders_equity": 53_000_000_000.0,
    "current_assets": None,              # banks: unclassified balance sheet
    "current_liabilities": None,
    "working_capital": None,
    "ebit": 6_100_000_000.0,
    "interest_expense": 4_200_000_000.0,
    "retained_earnings": 29_000_000_000.0,
    "cash": 31_000_000_000.0,
    "shares_outstanding": <per EDGAR dei facts>,
    "market_cap": <shares_outstanding × 2023-12-31 close>,
    "fred_snapshot": {
        "FEDFUNDS": 5.33, "CPIAUCSL": <value>, "UNRATE": <value>,
        "GS10": <value>, "T10Y2Y": -0.35
    },
    "raw_text_id": "PNC_2023_10k",
}
```

Altman Z from these fixture values = 0.253 (distress zone — expected for banks, contextualize vs peers).
