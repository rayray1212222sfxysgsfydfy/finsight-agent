"""IngestAgent — fetches raw financials from EDGAR, FRED, and yfinance.

Loads skills/ingest_agent/SKILL.md as its system prompt at runtime.
Writes all output under context["ingest"]. Never calls other agents.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.agents.context_schema import INGEST_TO_ANALYSIS_REQUIRED
from src.tools.db_tools import check_cache, write_to_db
from src.tools.fred_tools import get_macro_snapshot
from src.tools.market_tools import fetch_market_data
from src.tools.rag_tools import embed_text
from src.tools.sec_tools import fetch_10k_filing
from src.utils.exceptions import AgentExecutionError
from src.utils.logger import log_agent_trace

_SKILL_PATH = Path(__file__).resolve().parents[2] / "skills" / "ingest_agent" / "SKILL.md"


def _load_skill() -> str:
    """Load the ingest_agent SKILL.md system prompt."""
    return _SKILL_PATH.read_text()


def _year_end_market_cap(price_history: dict, year: int, shares: Optional[float]) -> Optional[float]:
    """Compute historical market cap from year-end close × shares outstanding."""
    if shares is None:
        return None
    close_data = price_history.get("Close", {})
    if not close_data:
        return None
    import pandas as pd
    items = sorted(close_data.items(), key=lambda kv: kv[0])
    idx = pd.to_datetime([k for k, _ in items], utc=True).tz_convert(None)
    series = pd.Series([v for _, v in items], index=idx).sort_index()
    value = series.asof(pd.Timestamp(f"{year}-12-31"))
    if pd.isna(value):
        return None
    return float(value) * shares


class IngestAgent:
    """Fetch and assemble raw financial data for a given bank-year."""

    def __init__(self) -> None:
        self.skill = _load_skill()

    def run(self, ticker: str, year: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch all data sources and return an IngestOutput-shaped dict.

        Raises ValueError for invalid inputs before any I/O.
        Raises AgentExecutionError for unrecoverable tool failures.
        Degradable failures (FRED, market data) add to context['warnings'].
        """
        if not ticker or not str(ticker).strip():
            raise ValueError("ticker must be a non-empty string")
        if not isinstance(year, int) or len(str(year)) != 4:
            raise ValueError(f"year must be a 4-digit integer, got {year!r}")

        context.setdefault("warnings", [])
        raw_text_id = f"{ticker}_{year}_10k"

        # 1. Cache check — return early on hit
        cache_result = check_cache(raw_text_id)
        if cache_result["error"] is None and cache_result["data"] is not None:
            log_agent_trace(
                run_id=context["run_id"],
                agent_name="IngestAgent",
                tool_name=None,
                input_summary=f"{ticker} {year}",
                output_summary="cache_hit",
                latency_ms=0.0,
            )
            return cache_result["data"]

        # 2. SEC / EDGAR — unrecoverable
        sec_result = fetch_10k_filing(ticker, year)
        if sec_result["error"] is not None:
            raise AgentExecutionError(
                f"IngestAgent: SEC fetch failed for {ticker} {year}: {sec_result['error']}"
            )
        filing = sec_result["data"]

        # 3. FRED macro snapshot — degradable
        fred_result = get_macro_snapshot(f"{year}-12-31")
        if fred_result["error"] is not None:
            context["warnings"].append(f"FRED error for {year}: {fred_result['error']}")
            fred_snapshot: dict = {}
        else:
            fred_snapshot = fred_result["data"]

        # 4. Market data — degradable
        market_result = fetch_market_data(ticker, period="max")
        if market_result["error"] is not None:
            context["warnings"].append(f"Market data error for {ticker}: {market_result['error']}")
            market_cap = None
        else:
            market_cap = _year_end_market_cap(
                market_result["data"].get("price_history", {}),
                year,
                filing.get("shares_outstanding"),
            )

        # 5. Embed structured financial summary for RAG — degradable
        def _fmt(v: Any, scale: float = 1e9, suffix: str = "B") -> str:
            return f"{v / scale:.2f}{suffix}" if v is not None else "N/A"

        filing_summary = (
            f"{ticker} {year} Annual Report (10-K) — Financial Highlights\n"
            f"Revenue: {_fmt(filing.get('revenue'))}\n"
            f"Net Income: {_fmt(filing.get('net_income'))}\n"
            f"Total Assets: {_fmt(filing.get('total_assets'))}\n"
            f"Total Liabilities: {_fmt(filing.get('total_liabilities'))}\n"
            f"Shareholders Equity: {_fmt(filing.get('shareholders_equity'))}\n"
            f"EBIT: {_fmt(filing.get('ebit'))}\n"
            f"Interest Expense: {_fmt(filing.get('interest_expense'))}\n"
            f"Retained Earnings: {_fmt(filing.get('retained_earnings'))}\n"
            f"Cash: {_fmt(filing.get('cash'))}\n"
            f"Market Cap: {_fmt(market_cap)}\n"
            f"FEDFUNDS Rate: {fred_snapshot.get('FEDFUNDS', 'N/A')}%\n"
            f"10Y-2Y Spread: {fred_snapshot.get('T10Y2Y', 'N/A')}%\n"
            f"CPI: {fred_snapshot.get('CPIAUCSL', 'N/A')}\n"
            f"Unemployment Rate: {fred_snapshot.get('UNRATE', 'N/A')}%\n"
        )
        embed_result = embed_text(
            filing_summary,
            metadata={"ticker": ticker, "year": year, "collection": "finsight_filings"},
        )
        if embed_result["error"] is not None:
            context["warnings"].append(f"Embed error for {ticker} {year}: {embed_result['error']}")

        # 6. Assemble IngestOutput
        ingest: Dict[str, Any] = {
            "revenue": filing.get("revenue"),
            "net_income": filing.get("net_income"),
            "total_assets": filing.get("total_assets"),
            "total_liabilities": filing.get("total_liabilities"),
            "shareholders_equity": filing.get("shareholders_equity"),
            "current_assets": filing.get("current_assets"),
            "current_liabilities": filing.get("current_liabilities"),
            "working_capital": filing.get("working_capital"),
            "ebit": filing.get("ebit"),
            "interest_expense": filing.get("interest_expense"),
            "retained_earnings": filing.get("retained_earnings"),
            "cash": filing.get("cash"),
            "shares_outstanding": filing.get("shares_outstanding"),
            "market_cap": market_cap,
            "fred_snapshot": fred_snapshot,
            "raw_text_id": raw_text_id,
            "error": None,
        }

        # 7. Validate required keys before handing off
        for key in INGEST_TO_ANALYSIS_REQUIRED:
            if key in ("ticker", "year"):
                continue  # these live on context, not ingest
            if ingest.get(key) is None:
                raise AgentExecutionError(
                    f"IngestAgent: required field '{key}' is None after ingestion"
                )

        # 8. Cache result
        write_to_db(raw_text_id, ingest)
        context["ingest"] = ingest

        log_agent_trace(
            run_id=context["run_id"],
            agent_name="IngestAgent",
            tool_name=None,
            input_summary=f"{ticker} {year}",
            output_summary=f"completed with {len(context['warnings'])} warnings",
            latency_ms=0.0,
        )

        return ingest
