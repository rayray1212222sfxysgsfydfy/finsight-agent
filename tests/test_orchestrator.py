"""Tests for OrchestratorAgent — TDD per CLAUDE.md. Run RED before implementation.

All 5 agent classes are mocked — no real agents instantiated in tests.
"""

import pytest

from src.agents.orchestrator import OrchestratorAgent
from src.utils.exceptions import AgentExecutionError

# ---------------------------------------------------------------------------
# Canned agent return values
# ---------------------------------------------------------------------------

_INGEST_OUTPUT = {
    "revenue": 22_000_000_000.0,
    "net_income": 5_400_000_000.0,
    "total_assets": 557_000_000_000.0,
    "total_liabilities": 504_000_000_000.0,
    "shareholders_equity": 53_000_000_000.0,
    "fred_snapshot": {"FEDFUNDS": 5.33, "T10Y2Y": -0.35},
    "market_cap": 55_000_000_000.0,
    "raw_text_id": "PNC_2023_10k",
}

_ANALYSIS_OUTPUT = {
    "ratios": {"debt_to_equity": 9.509, "return_on_assets": 0.0097,
               "net_margin": 0.245, "interest_coverage": 1.452, "current_ratio": None},
    "anomalies": [],
    "chart_paths": [],
}

_RISK_OUTPUT = {
    "z_score": 0.253,
    "z_zone": "distress",
    "ml_risk_prob": 0.72,
    "ml_risk_label": "high",
    "risk_flags": [],
    "peer_percentile": 0.55,
}

_NARRATIVE_OUTPUT = {"narrative": "PNC 2023 analyst narrative."}

_REPORT_OUTPUT = {"report_path": "reports/PNC_2023_report.pdf"}


def _mock_all_agents(mocker, *, ingest=None, analysis=None, risk=None,
                     narrative=None, report=None):
    """Patch all 5 agent classes; each instance's run() returns the given value."""

    def _make_mock(return_value=None, side_effect=None):
        instance = mocker.MagicMock()
        if side_effect is not None:
            instance.run.side_effect = side_effect
        else:
            instance.run.return_value = return_value
        cls = mocker.MagicMock(return_value=instance)
        return cls

    mocker.patch("src.agents.orchestrator.IngestAgent",
                 _make_mock(ingest or _INGEST_OUTPUT))
    mocker.patch("src.agents.orchestrator.AnalysisAgent",
                 _make_mock(analysis or _ANALYSIS_OUTPUT))
    mocker.patch("src.agents.orchestrator.RiskAgent",
                 _make_mock(risk or _RISK_OUTPUT))
    mocker.patch("src.agents.orchestrator.NarrativeAgent",
                 _make_mock(narrative or _NARRATIVE_OUTPUT))
    mocker.patch("src.agents.orchestrator.ReportAgent",
                 _make_mock(report or _REPORT_OUTPUT))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_orchestrator_run_success_returns_complete_dict(mocker):
    """Happy path: all agents succeed — returns status='complete' with required keys."""
    _mock_all_agents(mocker)
    agent = OrchestratorAgent()
    result = agent.run("Analyze PNC 2023")

    assert result["status"] == "complete"
    assert "report_path" in result
    assert result["report_path"].endswith(".pdf")
    assert "risk" in result
    assert "run_id" in result


def test_orchestrator_run_ingest_failure_returns_failed_dict(mocker):
    """IngestAgent raising AgentExecutionError returns {'status': 'failed', 'error': ...}."""
    _mock_all_agents(mocker)
    mocker.patch("src.agents.orchestrator.IngestAgent",
                 mocker.MagicMock(return_value=mocker.MagicMock(
                     run=mocker.MagicMock(side_effect=AgentExecutionError("EDGAR 404"))
                 )))
    agent = OrchestratorAgent()
    result = agent.run("Analyze PNC 2023")

    assert result["status"] == "failed"
    assert "error" in result
    assert isinstance(result["error"], str)


def test_orchestrator_run_risk_failure_returns_failed_dict(mocker):
    """RiskAgent raising AgentExecutionError returns {'status': 'failed', 'error': ...}."""
    _mock_all_agents(mocker)
    mocker.patch("src.agents.orchestrator.RiskAgent",
                 mocker.MagicMock(return_value=mocker.MagicMock(
                     run=mocker.MagicMock(side_effect=AgentExecutionError("Z-score failed"))
                 )))
    agent = OrchestratorAgent()
    result = agent.run("Analyze PNC 2023")

    assert result["status"] == "failed"
    assert "error" in result


def test_orchestrator_parse_query_valid_returns_ticker_year(mocker):
    """'Analyze PNC 2023' parses to {'ticker': 'PNC', 'year': 2023}."""
    agent = OrchestratorAgent()
    result = agent._parse_query("Analyze PNC 2023")

    assert result["ticker"] == "PNC"
    assert result["year"] == 2023


def test_orchestrator_parse_query_future_year_raises_value_error(mocker):
    """A future year must raise ValueError before any agent is called."""
    agent = OrchestratorAgent()
    with pytest.raises(ValueError, match="[Yy]ear|future|[Ff]uture"):
        agent._parse_query("Analyze PNC 2099")
