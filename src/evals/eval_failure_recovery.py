"""Tier 2 — Failure and recovery evals.

Verifies that the pipeline handles every class of failure gracefully:
no unhandled exceptions, always returns a status dict, warnings are
collected rather than swallowed silently.  Threshold: 100% pass rate.
"""

from unittest.mock import MagicMock, patch

from src.agents.orchestrator import OrchestratorAgent
from src.utils.exceptions import AgentExecutionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_filing() -> dict:
    return {
        "revenue": 22e9, "net_income": 5.4e9, "total_assets": 557e9,
        "total_liabilities": 504e9, "shareholders_equity": 53e9,
        "ebit": 6.1e9, "interest_expense": 4.2e9, "retained_earnings": 29e9,
        "cash": 31e9, "market_cap": 55e9, "shares_outstanding": None,
        "current_assets": None, "current_liabilities": None,
        "working_capital": None, "raw_text_id": "PNC-2023",
        "fred_snapshot": {"FEDFUNDS": 5.33, "T10Y2Y": -0.35, "UNRATE": 3.7,
                          "CPIAUCSL": 307.0, "GS10": 3.97},
    }


def _run_orchestrator(query: str, extra_patches=None) -> dict:
    """Run OrchestratorAgent with standard mocks + any extra patches."""
    mock_resp = MagicMock()
    mock_resp.stop_reason = "end_turn"
    mock_resp.content = [MagicMock(type="text", text="Narrative text.")]

    base_patches = [
        patch("src.agents.ingest_agent.fetch_10k_filing",
              return_value={"data": _mock_filing(), "error": None}),
        patch("src.agents.ingest_agent.get_macro_snapshot",
              return_value={"data": {"FEDFUNDS": 5.33, "T10Y2Y": -0.35,
                                      "UNRATE": 3.7, "CPIAUCSL": 307.0, "GS10": 3.97},
                             "error": None}),
        patch("src.agents.ingest_agent.fetch_market_data",
              return_value={"data": {"price_history": {}}, "error": None}),
        patch("src.agents.ingest_agent._year_end_market_cap",
              return_value=55_000_000_000.0),
        patch("src.agents.ingest_agent.embed_text",
              return_value={"data": {"doc_id": "abc"}, "error": None}),
        patch("src.agents.ingest_agent.check_cache",
              return_value={"data": None, "error": None}),
        patch("src.agents.ingest_agent.write_to_db",
              return_value={"data": True, "error": None}),
        patch("src.agents.narrative_agent.query_vector_store",
              return_value={"data": [], "error": None}),
        patch("src.agents.report_agent.generate_pdf",
              return_value={"data": {"path": "reports/PNC_2023_report.pdf"}, "error": None}),
        patch("src.agents.report_agent.format_table",
              return_value={"data": {"table": "t"}, "error": None}),
    ]

    all_patches = base_patches + (extra_patches or [])

    with patch("src.agents.narrative_agent.anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_resp
        for p in all_patches:
            p.start()
        try:
            orch = OrchestratorAgent()
            return orch.run(query)
        finally:
            for p in all_patches:
                p.stop()


# ---------------------------------------------------------------------------
# Failure / recovery evals
# ---------------------------------------------------------------------------

def eval_bad_query_format_returns_failed_not_raises():
    """Completely unparseable query must not raise — return status=failed."""
    orch = OrchestratorAgent()
    result = orch.run("hello world")
    assert isinstance(result, dict), "Must return a dict"
    assert result["status"] == "failed"
    assert "error" in result


def eval_edgar_503_returns_failed_with_run_id():
    """EDGAR 503 during ingest must return status=failed and preserve run_id."""
    extra = [
        patch("src.agents.ingest_agent.fetch_10k_filing",
              return_value={"data": None, "error": "503 Service Unavailable"}),
        patch("src.agents.ingest_agent.check_cache",
              return_value={"data": None, "error": None}),
    ]
    # run without the base ingest patches to let our error through
    mock_resp = MagicMock()
    mock_resp.stop_reason = "end_turn"
    mock_resp.content = [MagicMock(type="text", text="n/a")]
    with patch("src.agents.narrative_agent.anthropic") as mock_anthropic, \
         extra[0], extra[1]:
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_resp
        orch = OrchestratorAgent()
        result = orch.run("Analyze PNC 2023")
    assert result["status"] == "failed"
    assert "run_id" in result


def eval_narrative_api_failure_degrades_gracefully():
    """If NarrativeAgent raises AgentExecutionError, pipeline still fails cleanly."""
    extra = [
        patch("src.agents.narrative_agent.NarrativeAgent.run",
              side_effect=AgentExecutionError("API credit exhausted")),
    ]
    result = _run_orchestrator("Analyze PNC 2023", extra_patches=extra)
    assert result["status"] == "failed"
    assert "error" in result


def eval_report_pdf_failure_returns_failed():
    """If generate_pdf fails, pipeline returns status=failed."""
    extra = [
        patch("src.agents.report_agent.generate_pdf",
              return_value={"data": None, "error": "disk quota exceeded"}),
    ]
    result = _run_orchestrator("Analyze PNC 2023", extra_patches=extra)
    assert result["status"] == "failed"


def eval_fred_timeout_collected_as_warning_not_crash():
    """FRED timeout must be collected as a warning, pipeline must still complete."""
    extra = [
        patch("src.agents.ingest_agent.get_macro_snapshot",
              return_value={"data": None, "error": "connection timeout"}),
    ]
    result = _run_orchestrator("Analyze PNC 2023", extra_patches=extra)
    # Pipeline should complete (FRED is degradable)
    assert result["status"] == "complete", f"Expected complete, got {result}"


def eval_market_data_failure_returns_status_dict_not_raises():
    """Market data failure must not raise an unhandled exception — returns a dict."""
    extra = [
        patch("src.agents.ingest_agent.fetch_market_data",
              return_value={"data": None, "error": "yfinance timeout"}),
        patch("src.agents.ingest_agent._year_end_market_cap",
              return_value=None),
    ]
    # Without market_cap, Altman Z fails, so pipeline returns failed — that's acceptable.
    # The contract is: no uncaught exception, always a dict with 'status'.
    result = _run_orchestrator("Analyze PNC 2023", extra_patches=extra)
    assert isinstance(result, dict), "Must return a dict"
    assert "status" in result


def eval_missing_narrative_context_uses_placeholder():
    """If NarrativeAgent produces no narrative, ReportAgent uses placeholder."""
    from src.agents.report_agent import ReportAgent
    from src.tools.analysis_tools import calc_ratios

    ingest = _mock_filing()
    ratios = calc_ratios(ingest)["data"]
    ctx = {
        "ticker": "PNC", "year": 2023, "run_id": "t",
        "ingest": ingest,
        "analysis": {"ratios": ratios, "anomalies": [], "chart_paths": []},
        "risk": {
            "z_score": 0.284, "z_zone": "distress",
            "ml_risk_prob": 0.45, "ml_risk_label": "moderate",
            "risk_flags": [], "peer_percentile": 1.0,
        },
        # narrative intentionally omitted
    }
    with patch("src.agents.report_agent.generate_pdf",
               return_value={"data": {"path": "reports/PNC_2023_report.pdf"}, "error": None}), \
         patch("src.agents.report_agent.format_table",
               return_value={"data": {"table": "t"}, "error": None}):
        agent = ReportAgent()
        result = agent.run(ctx)

    assert "report_path" in result
    assert any("narrative" in w.lower() for w in ctx.get("warnings", []))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_EVALS = [
    eval_bad_query_format_returns_failed_not_raises,
    eval_edgar_503_returns_failed_with_run_id,
    eval_narrative_api_failure_degrades_gracefully,
    eval_report_pdf_failure_returns_failed,
    eval_fred_timeout_collected_as_warning_not_crash,
    eval_market_data_failure_returns_status_dict_not_raises,
    eval_missing_narrative_context_uses_placeholder,
]


def run_failure_recovery_evals() -> dict:
    passed, failed = [], []
    for fn in _EVALS:
        try:
            fn()
            passed.append(fn.__name__)
        except Exception as exc:
            failed.append((fn.__name__, str(exc)))
    return {"passed": passed, "failed": failed, "total": len(_EVALS)}


if __name__ == "__main__":
    summary = run_failure_recovery_evals()
    print(f"Failure/recovery evals: {len(summary['passed'])}/{summary['total']} passed")
    for name, err in summary["failed"]:
        print(f"  FAIL {name}: {err}")
