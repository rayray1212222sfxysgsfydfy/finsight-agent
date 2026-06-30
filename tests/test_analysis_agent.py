"""Tests for AnalysisAgent — TDD per CLAUDE.md. Run RED before implementation.

All tool functions are mocked — no real API calls permitted.
"""

import pytest

from src.agents.analysis_agent import AnalysisAgent
from src.utils.exceptions import AgentExecutionError

# ---------------------------------------------------------------------------
# Canned tool responses
# ---------------------------------------------------------------------------

_RATIOS_SUCCESS = {
    "data": {
        "current_ratio": None,
        "debt_to_equity": 9.509,
        "return_on_assets": 0.0097,
        "net_margin": 0.245,
        "interest_coverage": 1.452,
    },
    "error": None,
}

_ANOMALIES_SUCCESS = {
    "data": [
        {
            "metric": "interest_coverage",
            "description": "Coverage 1.452 below 1.5",
            "severity": "high",
            "code": "low_interest_coverage",
        },
        {
            "metric": "T10Y2Y",
            "description": "Yield curve inverted",
            "severity": "high",
            "code": "yield_curve_inverted",
        },
    ],
    "error": None,
}

_CHART_SUCCESS = {"data": {"path": "reports/charts/PNC_2023_ratios.png"}, "error": None}

_BASE_CONTEXT = {
    "ticker": "PNC",
    "year": 2023,
    "run_id": "abc123",
    "ingest": {
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
        "current_assets": None,
        "current_liabilities": None,
        "working_capital": None,
        "shares_outstanding": 400_000_000.0,
        "fred_snapshot": {"FEDFUNDS": 5.33, "T10Y2Y": -0.35, "UNRATE": 3.7,
                          "CPIAUCSL": 314.17, "GS10": 3.97},
        "raw_text_id": "PNC_2023_10k",
        "error": None,
    },
}


def _fresh_context():
    """Return a deep copy of base context so tests don't share state."""
    import copy
    return copy.deepcopy(_BASE_CONTEXT)


def _mock_all(mocker, *, ratios=_RATIOS_SUCCESS, anomalies=_ANOMALIES_SUCCESS,
              chart=_CHART_SUCCESS):
    mocker.patch("src.agents.analysis_agent.calc_ratios", return_value=ratios)
    mocker.patch("src.agents.analysis_agent.detect_anomalies", return_value=anomalies)
    mocker.patch("src.agents.analysis_agent.plot_chart", return_value=chart)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_analysis_agent_run_success_returns_analysis_output(mocker):
    """Happy path: all tools succeed — output has ratios, anomalies, chart_paths."""
    _mock_all(mocker)
    ctx = _fresh_context()
    agent = AnalysisAgent()
    result = agent.run(ctx)

    assert "ratios" in result
    assert "anomalies" in result
    assert "chart_paths" in result
    assert isinstance(result["anomalies"], list)
    assert isinstance(result["chart_paths"], list)
    assert result["ratios"]["debt_to_equity"] == pytest.approx(9.509)


def test_analysis_agent_run_calc_ratios_failure_raises_agent_execution_error(mocker):
    """calc_ratios error is unrecoverable — must raise AgentExecutionError."""
    _mock_all(mocker, ratios={"data": None, "error": "division by zero"})
    ctx = _fresh_context()
    agent = AnalysisAgent()

    with pytest.raises(AgentExecutionError, match="ratios"):
        agent.run(ctx)


def test_analysis_agent_run_plot_chart_failure_continues_with_warning(mocker):
    """plot_chart failure is degradable — run continues, chart_paths is empty."""
    _mock_all(mocker, chart={"data": None, "error": "matplotlib error"})
    ctx = _fresh_context()
    agent = AnalysisAgent()
    result = agent.run(ctx)

    assert result["chart_paths"] == []
    assert any("chart" in w.lower() for w in ctx.get("warnings", []))


def test_analysis_agent_run_detect_anomalies_failure_continues_with_empty_list(mocker):
    """detect_anomalies failure is degradable — anomalies defaults to []."""
    _mock_all(mocker, anomalies={"data": None, "error": "anomaly detector error"})
    ctx = _fresh_context()
    agent = AnalysisAgent()
    result = agent.run(ctx)

    assert result["anomalies"] == []
    assert any("anomal" in w.lower() for w in ctx.get("warnings", []))


def test_analysis_agent_run_missing_ingest_raises_agent_execution_error(mocker):
    """Missing context['ingest'] must raise AgentExecutionError before any tool call."""
    calc_mock = mocker.patch("src.agents.analysis_agent.calc_ratios")
    ctx = {"ticker": "PNC", "year": 2023, "run_id": "abc123"}  # no 'ingest' key
    agent = AnalysisAgent()

    with pytest.raises(AgentExecutionError, match="ingest"):
        agent.run(ctx)

    calc_mock.assert_not_called()
