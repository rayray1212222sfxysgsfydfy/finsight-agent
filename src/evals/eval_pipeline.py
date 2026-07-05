"""Tier 2 — Pipeline integration evals.

Tests the full Ingest→Analysis→Risk→Narrative→Report sequence through
OrchestratorAgent with all external I/O mocked.  Threshold: 80% pass rate.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agents.orchestrator import OrchestratorAgent
from src.utils.exceptions import AgentExecutionError


# ---------------------------------------------------------------------------
# Shared mock helpers
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


def _mock_narrative_response() -> MagicMock:
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [MagicMock(type="text",
                               text="PNC demonstrated resilience in 2023 despite yield curve inversion.")]
    return resp


def _full_pipeline_patches():
    """Return a list of (target, kwargs) for patch() to mock all external I/O."""
    return [
        ("src.agents.ingest_agent.fetch_10k_filing",
         {"return_value": {"data": _mock_filing(), "error": None}}),
        ("src.agents.ingest_agent.get_macro_snapshot",
         {"return_value": {"data": {"FEDFUNDS": 5.33, "T10Y2Y": -0.35,
                                     "UNRATE": 3.7, "CPIAUCSL": 307.0, "GS10": 3.97},
                            "error": None}}),
        ("src.agents.ingest_agent.fetch_market_data",
         {"return_value": {"data": {"price_history": {}}, "error": None}}),
        # _year_end_market_cap returns None when price_history is empty; patch it
        # to return PNC 2023 fixture market cap so score_altman_z has what it needs.
        ("src.agents.ingest_agent._year_end_market_cap",
         {"return_value": 55_000_000_000.0}),
        ("src.agents.ingest_agent.embed_text",
         {"return_value": {"data": {"doc_id": "abc"}, "error": None}}),
        ("src.agents.ingest_agent.check_cache",
         {"return_value": {"data": None, "error": None}}),
        ("src.agents.ingest_agent.write_to_db",
         {"return_value": {"data": True, "error": None}}),
        ("src.agents.narrative_agent.query_vector_store",
         {"return_value": {"data": [], "error": None}}),
    ]


def _run_with_mocks(query: str, narrative_text: str = "Analysis complete.") -> dict:
    """Run the full pipeline with all external I/O mocked."""
    mock_resp = MagicMock()
    mock_resp.stop_reason = "end_turn"
    mock_resp.content = [MagicMock(type="text", text=narrative_text)]

    patches = _full_pipeline_patches()
    with patch("src.agents.narrative_agent.anthropic") as mock_anthropic, \
         patch("src.agents.report_agent.generate_pdf",
               return_value={"data": {"path": "reports/PNC_2023_report.pdf"}, "error": None}), \
         patch("src.agents.report_agent.format_table",
               return_value={"data": {"table": "Metric | Value"}, "error": None}):
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_resp
        ctx_patches = [patch(target, **kwargs) for target, kwargs in patches]
        for p in ctx_patches:
            p.start()
        try:
            orch = OrchestratorAgent()
            result = orch.run(query)
        finally:
            for p in ctx_patches:
                p.stop()
    return result


# ---------------------------------------------------------------------------
# Integration evals
# ---------------------------------------------------------------------------

def eval_pipeline_pnc_2023_completes():
    """Full pipeline for PNC 2023 must return status=complete."""
    result = _run_with_mocks("Analyze PNC 2023")
    assert result["status"] == "complete", f"Expected complete, got: {result}"


def eval_pipeline_result_has_report_path():
    """Completed pipeline must include a report_path."""
    result = _run_with_mocks("Analyze PNC 2023")
    assert "report_path" in result, "Pipeline result missing report_path"


def eval_pipeline_result_has_risk_summary():
    """Completed pipeline must include a risk dict with z_score and ml_risk_label."""
    result = _run_with_mocks("Analyze PNC 2023")
    assert "risk" in result, "Pipeline result missing risk dict"
    assert "z_score" in result["risk"], "risk dict missing z_score"
    assert "ml_risk_label" in result["risk"], "risk dict missing ml_risk_label"


def eval_pipeline_invalid_query_returns_failed():
    """An unparseable query must return status=failed, not raise."""
    orch = OrchestratorAgent()
    result = orch.run("just some random text")
    assert result["status"] == "failed"
    assert "error" in result


def eval_pipeline_future_year_returns_failed():
    """A future year must return status=failed with a clear error message."""
    orch = OrchestratorAgent()
    result = orch.run("Analyze PNC 2099")
    assert result["status"] == "failed"
    assert "error" in result


def eval_pipeline_ingest_failure_returns_failed():
    """If IngestAgent fails, the pipeline must return status=failed gracefully."""
    with patch("src.agents.ingest_agent.fetch_10k_filing",
               return_value={"data": None, "error": "EDGAR 500"}), \
         patch("src.agents.ingest_agent.check_cache",
               return_value={"data": None, "error": None}):
        orch = OrchestratorAgent()
        result = orch.run("Analyze PNC 2023")
    assert result["status"] == "failed"
    assert "run_id" in result


def eval_pipeline_run_id_present():
    """Every pipeline result must have a run_id for traceability."""
    result = _run_with_mocks("Analyze PNC 2023")
    assert "run_id" in result and result["run_id"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_EVALS = [
    eval_pipeline_pnc_2023_completes,
    eval_pipeline_result_has_report_path,
    eval_pipeline_result_has_risk_summary,
    eval_pipeline_invalid_query_returns_failed,
    eval_pipeline_future_year_returns_failed,
    eval_pipeline_ingest_failure_returns_failed,
    eval_pipeline_run_id_present,
]

PASS_THRESHOLD = 0.80


def run_pipeline_evals() -> dict:
    passed, failed = [], []
    for fn in _EVALS:
        try:
            fn()
            passed.append(fn.__name__)
        except Exception as exc:
            failed.append((fn.__name__, str(exc)))
    rate = len(passed) / len(_EVALS)
    return {
        "passed": passed, "failed": failed,
        "total": len(_EVALS), "pass_rate": rate,
        "threshold_met": rate >= PASS_THRESHOLD,
    }


if __name__ == "__main__":
    summary = run_pipeline_evals()
    print(f"Pipeline evals: {len(summary['passed'])}/{summary['total']} "
          f"({summary['pass_rate']:.0%}) — "
          f"{'PASS' if summary['threshold_met'] else 'FAIL'} (threshold {PASS_THRESHOLD:.0%})")
    for name, err in summary["failed"]:
        print(f"  FAIL {name}: {err}")
