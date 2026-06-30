"""Tests for RiskAgent — TDD per CLAUDE.md. Run RED before implementation.

All tool functions are mocked — no real API calls permitted.
"""

import pytest

from src.agents.risk_agent import RiskAgent
from src.utils.exceptions import AgentExecutionError

# ---------------------------------------------------------------------------
# Canned tool responses
# ---------------------------------------------------------------------------

_ALTMAN_SUCCESS = {
    "data": {"z_score": 0.253, "z_zone": "distress"},
    "error": None,
}

_ML_SUCCESS = {
    "data": {"probability": 0.72, "label": "high"},
    "error": None,
}

_FLAGS_SUCCESS = {
    "data": [
        {"code": "yield_curve_inverted", "description": "T10Y2Y=-0.35", "severity": "high"},
        {"code": "low_interest_coverage", "description": "Coverage 1.452 < 1.5", "severity": "high"},
    ],
    "error": None,
}

_PEERS_SUCCESS = {"data": 0.55, "error": None}

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
    "analysis": {
        "ratios": {
            "current_ratio": None,
            "debt_to_equity": 9.509,
            "return_on_assets": 0.0097,
            "net_margin": 0.245,
            "interest_coverage": 1.452,
        },
        "anomalies": [
            {"metric": "interest_coverage", "description": "Coverage low",
             "severity": "high", "code": "low_interest_coverage"},
        ],
        "chart_paths": ["reports/charts/PNC_2023_ratios.png"],
    },
}


def _fresh_context():
    import copy
    return copy.deepcopy(_BASE_CONTEXT)


def _mock_all(mocker, *, altman=_ALTMAN_SUCCESS, ml=_ML_SUCCESS,
              flags=_FLAGS_SUCCESS, peers=_PEERS_SUCCESS):
    mocker.patch("src.agents.risk_agent.score_altman_z", return_value=altman)
    mocker.patch("src.agents.risk_agent.run_ml_risk_model", return_value=ml)
    mocker.patch("src.agents.risk_agent.flag_risks", return_value=flags)
    mocker.patch("src.agents.risk_agent.compare_to_peers", return_value=peers)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_risk_agent_run_success_returns_risk_output(mocker):
    """Happy path: all tools succeed — output matches RiskOutput schema."""
    _mock_all(mocker)
    ctx = _fresh_context()
    agent = RiskAgent()
    result = agent.run(ctx)

    assert result["z_score"] == pytest.approx(0.253)
    assert result["z_zone"] == "distress"
    assert result["ml_risk_prob"] == pytest.approx(0.72)
    assert result["ml_risk_label"] == "high"
    assert isinstance(result["risk_flags"], list)
    assert len(result["risk_flags"]) == 2
    assert result["peer_percentile"] == pytest.approx(0.55)


def test_risk_agent_run_altman_failure_raises_agent_execution_error(mocker):
    """score_altman_z error is unrecoverable — must raise AgentExecutionError."""
    _mock_all(mocker, altman={"data": None, "error": "zero total_assets"})
    ctx = _fresh_context()
    agent = RiskAgent()

    with pytest.raises(AgentExecutionError, match="[Zz][-_]?[Ss]core|altman"):
        agent.run(ctx)


def test_risk_agent_run_ml_failure_continues_with_warning(mocker):
    """run_ml_risk_model failure is degradable — run continues with unknown label."""
    _mock_all(mocker, ml={"data": None, "error": "pkl not found"})
    ctx = _fresh_context()
    agent = RiskAgent()
    result = agent.run(ctx)

    assert result["z_score"] == pytest.approx(0.253)
    assert result["ml_risk_label"] == "unknown"
    assert result["ml_risk_prob"] == pytest.approx(0.0)
    assert any("ml" in w.lower() or "model" in w.lower() for w in ctx.get("warnings", []))


def test_risk_agent_run_compare_to_peers_failure_continues_with_none(mocker):
    """compare_to_peers failure is degradable — peer_percentile defaults to None."""
    _mock_all(mocker, peers={"data": None, "error": "no peer data"})
    ctx = _fresh_context()
    agent = RiskAgent()
    result = agent.run(ctx)

    assert result["peer_percentile"] is None
    assert any("peer" in w.lower() for w in ctx.get("warnings", []))


def test_risk_agent_run_missing_analysis_raises_agent_execution_error(mocker):
    """Missing context['analysis'] must raise AgentExecutionError before any tool."""
    altman_mock = mocker.patch("src.agents.risk_agent.score_altman_z")
    ctx = _fresh_context()
    del ctx["analysis"]
    agent = RiskAgent()

    with pytest.raises(AgentExecutionError, match="analysis"):
        agent.run(ctx)

    altman_mock.assert_not_called()
