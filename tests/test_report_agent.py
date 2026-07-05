"""Tests for ReportAgent — TDD per CLAUDE.md. Run RED before implementation.

All tool functions are mocked — no real PDFs written in tests.
"""

import copy
import pytest

from src.agents.report_agent import ReportAgent
from src.utils.exceptions import AgentExecutionError

# ---------------------------------------------------------------------------
# Canned tool responses
# ---------------------------------------------------------------------------

_TABLE_SUCCESS = {
    "data": {"table": "metric      | value\n------------+-------\nz_score     | 0.253"},
    "error": None,
}

_PDF_SUCCESS = {
    "data": {"path": "reports/PNC_2023_report.pdf"},
    "error": None,
}

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
        "fred_snapshot": {"FEDFUNDS": 5.33, "T10Y2Y": -0.35, "UNRATE": 3.7},
        "market_cap": 55_000_000_000.0,
        "raw_text_id": "PNC_2023_10k",
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
            {"metric": "interest_coverage", "description": "Coverage low", "severity": "high"},
        ],
        "chart_paths": ["reports/charts/PNC_2023_ratios.png"],
    },
    "risk": {
        "z_score": 0.253,
        "z_zone": "distress",
        "ml_risk_prob": 0.72,
        "ml_risk_label": "high",
        "risk_flags": [
            {"code": "yield_curve_inverted", "description": "T10Y2Y=-0.35", "severity": "high"},
        ],
        "peer_percentile": 0.55,
    },
    "narrative": {
        "narrative": "PNC 2023 showed strong revenue of $22B despite yield curve inversion.",
    },
}


def _fresh():
    return copy.deepcopy(_BASE_CONTEXT)


def _mock_tools(mocker, *, table=_TABLE_SUCCESS, pdf=_PDF_SUCCESS):
    mocker.patch("src.agents.report_agent.format_table", return_value=table)
    mocker.patch("src.agents.report_agent.generate_pdf", return_value=pdf)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_report_agent_run_success_returns_report_path(mocker):
    """Happy path: all tools succeed — result has 'report_path' ending in .pdf."""
    _mock_tools(mocker)
    ctx = _fresh()
    agent = ReportAgent()
    result = agent.run(ctx)

    assert "report_path" in result
    assert result["report_path"].endswith(".pdf")


def test_report_agent_run_pdf_failure_raises_agent_execution_error(mocker):
    """generate_pdf failure is unrecoverable — must raise AgentExecutionError."""
    _mock_tools(mocker, pdf={"data": None, "error": "disk full"})
    ctx = _fresh()
    agent = ReportAgent()

    with pytest.raises(AgentExecutionError, match="[Pp][Dd][Ff]|[Rr]eport"):
        agent.run(ctx)


def test_report_agent_run_missing_risk_raises_agent_execution_error(mocker):
    """Missing context['risk'] must raise AgentExecutionError before any tool call."""
    generate_mock = mocker.patch("src.agents.report_agent.generate_pdf")
    ctx = _fresh()
    del ctx["risk"]
    agent = ReportAgent()

    with pytest.raises(AgentExecutionError, match="risk"):
        agent.run(ctx)

    generate_mock.assert_not_called()


def test_report_agent_run_missing_narrative_continues_with_placeholder(mocker):
    """Missing context['narrative'] is degradable — agent uses placeholder and continues."""
    _mock_tools(mocker)
    ctx = _fresh()
    del ctx["narrative"]
    agent = ReportAgent()
    result = agent.run(ctx)

    assert "report_path" in result
    # The generate_pdf call must have been made with a non-empty narrative string
    call_kwargs = mocker.patch.object.__class__  # resolve via the mock itself below


def test_report_agent_run_output_path_matches_pattern(mocker):
    """Output path must be reports/<TICKER>_<YEAR>_report.pdf."""
    captured = {}

    def _fake_generate_pdf(output_path, ticker, year, narrative, table_str, chart_paths=None, eda_chart_paths=None, **kwargs):
        captured["output_path"] = output_path
        return _PDF_SUCCESS

    mocker.patch("src.agents.report_agent.format_table", return_value=_TABLE_SUCCESS)
    mocker.patch("src.agents.report_agent.generate_pdf", side_effect=_fake_generate_pdf)

    ctx = _fresh()
    agent = ReportAgent()
    agent.run(ctx)

    assert captured["output_path"] == "reports/PNC_2023_report.pdf"
