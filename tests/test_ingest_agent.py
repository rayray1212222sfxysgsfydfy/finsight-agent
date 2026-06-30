"""Tests for IngestAgent — TDD per CLAUDE.md. Run RED before implementation.

All tool functions are mocked — no real API calls permitted.
"""

import pytest

from src.agents.ingest_agent import IngestAgent
from src.utils.exceptions import AgentExecutionError

# ---------------------------------------------------------------------------
# Helpers — canned tool responses
# ---------------------------------------------------------------------------

_SEC_SUCCESS = {
    "data": {
        "revenue": 22_000_000_000.0,
        "net_income": 5_400_000_000.0,
        "total_assets": 557_000_000_000.0,
        "total_liabilities": 504_000_000_000.0,
        "shareholders_equity": 53_000_000_000.0,
        "ebit": 6_100_000_000.0,
        "interest_expense": 4_200_000_000.0,
        "retained_earnings": 29_000_000_000.0,
        "cash": 31_000_000_000.0,
        "current_assets": None,
        "current_liabilities": None,
        "working_capital": None,
        "shares_outstanding": 400_000_000.0,
    },
    "error": None,
}

_MARKET_SUCCESS = {
    "data": {
        "market_cap": 55_000_000_000.0,
        "price_history": {
            "Close": {
                "2023-12-29": 137.50,
            }
        },
    },
    "error": None,
}

_FRED_SUCCESS = {
    "data": {
        "FEDFUNDS": 5.33,
        "CPIAUCSL": 314.17,
        "UNRATE": 3.7,
        "GS10": 3.97,
        "T10Y2Y": -0.35,
    },
    "error": None,
}

_EMBED_SUCCESS = {"data": {"doc_id": "PNC_2023_10k"}, "error": None}
_CACHE_MISS = {"data": None, "error": None}


def _mock_all(mocker, *, sec=_SEC_SUCCESS, market=_MARKET_SUCCESS,
              fred=_FRED_SUCCESS, embed=_EMBED_SUCCESS):
    """Patch all four tool functions used by IngestAgent."""
    mocker.patch("src.agents.ingest_agent.check_cache", return_value=_CACHE_MISS)
    mocker.patch("src.agents.ingest_agent.fetch_10k_filing", return_value=sec)
    mocker.patch("src.agents.ingest_agent.fetch_market_data", return_value=market)
    mocker.patch("src.agents.ingest_agent.get_macro_snapshot", return_value=fred)
    mocker.patch("src.agents.ingest_agent.embed_text", return_value=embed)
    mocker.patch("src.agents.ingest_agent.write_to_db", return_value={"data": True, "error": None})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ingest_agent_run_success_returns_required_keys(mocker):
    """Happy path: all tools succeed — output contains every required key."""
    _mock_all(mocker)
    ctx = {"ticker": "PNC", "year": 2023, "run_id": "abc123"}
    agent = IngestAgent()
    result = agent.run("PNC", 2023, ctx)

    required = ["revenue", "net_income", "total_assets", "fred_snapshot", "raw_text_id"]
    for key in required:
        assert key in result, f"Missing required key: {key}"
    assert result["error"] is None


def test_ingest_agent_run_sec_failure_raises_agent_execution_error(mocker):
    """fetch_10k_filing returning an error dict must raise AgentExecutionError."""
    _mock_all(mocker, sec={"data": None, "error": "EDGAR 404"})
    ctx = {"ticker": "PNC", "year": 2023, "run_id": "abc123"}
    agent = IngestAgent()

    with pytest.raises(AgentExecutionError, match="SEC"):
        agent.run("PNC", 2023, ctx)


def test_ingest_agent_run_fred_failure_continues_with_warning(mocker):
    """get_macro_snapshot failure is degradable — run continues, warning added."""
    _mock_all(mocker, fred={"data": None, "error": "FRED timeout"})
    ctx = {"ticker": "PNC", "year": 2023, "run_id": "abc123"}
    agent = IngestAgent()
    result = agent.run("PNC", 2023, ctx)

    assert result["error"] is None
    assert result["fred_snapshot"] == {}
    assert any("FRED" in w for w in ctx.get("warnings", []))


def test_ingest_agent_run_market_failure_sets_market_cap_none(mocker):
    """fetch_market_data failure is degradable — market_cap is None, run continues."""
    _mock_all(mocker, market={"data": None, "error": "yfinance error"})
    ctx = {"ticker": "PNC", "year": 2023, "run_id": "abc123"}
    agent = IngestAgent()
    result = agent.run("PNC", 2023, ctx)

    assert result["error"] is None
    assert result["market_cap"] is None


def test_ingest_agent_run_invalid_ticker_raises_value_error(mocker):
    """Empty ticker must raise ValueError before any tool is called."""
    fetch_mock = mocker.patch("src.agents.ingest_agent.fetch_10k_filing")
    ctx = {"ticker": "", "year": 2023, "run_id": "abc123"}
    agent = IngestAgent()

    with pytest.raises(ValueError, match="ticker"):
        agent.run("", 2023, ctx)

    fetch_mock.assert_not_called()
