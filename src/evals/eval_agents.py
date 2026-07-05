"""Tier 2 — Agent contract evals.

Checks that each agent honours its input/output contract without making real
API calls.  All external I/O is mocked at the tool layer so these run fast
and deterministically.
"""

from unittest.mock import MagicMock, patch

from src.utils.exceptions import AgentExecutionError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pnc_ingest() -> dict:
    return {
        "revenue": 22e9, "net_income": 5.4e9, "total_assets": 557e9,
        "total_liabilities": 504e9, "shareholders_equity": 53e9,
        "ebit": 6.1e9, "interest_expense": 4.2e9, "retained_earnings": 29e9,
        "cash": 31e9, "market_cap": 55e9,
        "current_assets": None, "current_liabilities": None,
        "working_capital": None, "raw_text_id": "PNC-2023",
        "fred_snapshot": {"FEDFUNDS": 5.33, "T10Y2Y": -0.35, "UNRATE": 3.7,
                          "CPIAUCSL": 307.0, "GS10": 3.97},
        "shares_outstanding": None,
    }


def _base_context(ingest=None) -> dict:
    return {
        "ticker": "PNC", "year": 2023, "run_id": "test-run-001",
        "ingest": ingest or _pnc_ingest(),
    }


def _analysis_output() -> dict:
    return {
        "ratios": {
            "debt_to_equity": 9.509, "return_on_assets": 0.0097,
            "net_margin": 0.245, "interest_coverage": 1.452,
            "current_ratio": None,
        },
        "anomalies": [],
        "chart_paths": [],
    }


def _risk_output() -> dict:
    return {
        "z_score": 0.284, "z_zone": "distress",
        "ml_risk_prob": 0.45, "ml_risk_label": "moderate",
        "risk_flags": [], "peer_percentile": 1.0,
    }


# ---------------------------------------------------------------------------
# IngestAgent
# ---------------------------------------------------------------------------

def eval_ingest_agent_run_returns_required_keys():
    """IngestAgent.run must populate ingest output with required keys."""
    from src.agents.ingest_agent import IngestAgent
    agent = IngestAgent()
    ctx = {"ticker": "PNC", "year": 2023, "run_id": "t1"}

    mock_filing = {**_pnc_ingest(), "shares_outstanding": None}
    mock_fred = {"FEDFUNDS": 5.33, "T10Y2Y": -0.35, "UNRATE": 3.7,
                 "CPIAUCSL": 307.0, "GS10": 3.97}

    with patch("src.agents.ingest_agent.fetch_10k_filing",
               return_value={"data": mock_filing, "error": None}), \
         patch("src.agents.ingest_agent.get_macro_snapshot",
               return_value={"data": mock_fred, "error": None}), \
         patch("src.agents.ingest_agent.fetch_market_data",
               return_value={"data": {"price_history": {}}, "error": None}), \
         patch("src.agents.ingest_agent._year_end_market_cap",
               return_value=55_000_000_000.0), \
         patch("src.agents.ingest_agent.embed_text",
               return_value={"data": {"doc_id": "abc"}, "error": None}), \
         patch("src.agents.ingest_agent.check_cache",
               return_value={"data": None, "error": None}), \
         patch("src.agents.ingest_agent.write_to_db",
               return_value={"data": True, "error": None}):
        result = agent.run("PNC", 2023, ctx)

    for key in ("revenue", "net_income", "total_assets", "fred_snapshot"):
        assert key in result, f"IngestAgent output missing key: {key}"


def eval_ingest_agent_missing_ticker_raises():
    from src.agents.ingest_agent import IngestAgent
    agent = IngestAgent()
    try:
        agent.run("", 2023, {"run_id": "t"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def eval_ingest_agent_future_year_raises():
    from src.agents.ingest_agent import IngestAgent
    agent = IngestAgent()
    try:
        agent.run("PNC", 2099, {"run_id": "t"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def eval_ingest_agent_sec_failure_raises_agent_error():
    from src.agents.ingest_agent import IngestAgent
    agent = IngestAgent()
    ctx = {"ticker": "PNC", "year": 2023, "run_id": "t"}
    with patch("src.agents.ingest_agent.fetch_10k_filing",
               return_value={"data": None, "error": "EDGAR 404"}), \
         patch("src.agents.ingest_agent.check_cache",
               return_value={"data": None, "error": None}):
        try:
            agent.run("PNC", 2023, ctx)
            assert False, "Should have raised AgentExecutionError"
        except AgentExecutionError:
            pass


def eval_ingest_agent_fred_failure_degrades_gracefully():
    from src.agents.ingest_agent import IngestAgent
    agent = IngestAgent()
    ctx = {"ticker": "PNC", "year": 2023, "run_id": "t"}
    mock_filing = {**_pnc_ingest(), "shares_outstanding": None}
    with patch("src.agents.ingest_agent.fetch_10k_filing",
               return_value={"data": mock_filing, "error": None}), \
         patch("src.agents.ingest_agent.get_macro_snapshot",
               return_value={"data": None, "error": "FRED timeout"}), \
         patch("src.agents.ingest_agent.fetch_market_data",
               return_value={"data": {"price_history": {}}, "error": None}), \
         patch("src.agents.ingest_agent._year_end_market_cap",
               return_value=55_000_000_000.0), \
         patch("src.agents.ingest_agent.embed_text",
               return_value={"data": {"doc_id": "x"}, "error": None}), \
         patch("src.agents.ingest_agent.check_cache",
               return_value={"data": None, "error": None}), \
         patch("src.agents.ingest_agent.write_to_db",
               return_value={"data": True, "error": None}):
        result = agent.run("PNC", 2023, ctx)
    assert "warnings" in ctx and any("FRED" in w for w in ctx["warnings"])
    assert "revenue" in result  # still completed


# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------

def eval_analysis_agent_run_returns_required_keys():
    from src.agents.analysis_agent import AnalysisAgent
    agent = AnalysisAgent()
    ctx = _base_context()
    result = agent.run(ctx)
    for key in ("ratios", "anomalies", "chart_paths"):
        assert key in result, f"AnalysisAgent output missing: {key}"


def eval_analysis_agent_missing_ingest_raises():
    from src.agents.analysis_agent import AnalysisAgent
    agent = AnalysisAgent()
    try:
        agent.run({"ticker": "PNC", "year": 2023, "run_id": "t"})
        assert False, "Should raise AgentExecutionError"
    except AgentExecutionError:
        pass


def eval_analysis_agent_writes_to_context():
    from src.agents.analysis_agent import AnalysisAgent
    agent = AnalysisAgent()
    ctx = _base_context()
    agent.run(ctx)
    assert "analysis" in ctx, "AnalysisAgent must write context['analysis']"


def eval_analysis_agent_ratios_non_empty():
    from src.agents.analysis_agent import AnalysisAgent
    agent = AnalysisAgent()
    ctx = _base_context()
    result = agent.run(ctx)
    assert result["ratios"], "ratios dict must not be empty"


def eval_analysis_agent_chart_failure_degrades():
    from src.agents.analysis_agent import AnalysisAgent
    agent = AnalysisAgent()
    ctx = _base_context()
    with patch("src.agents.analysis_agent.plot_chart",
               return_value={"data": None, "error": "disk full"}):
        result = agent.run(ctx)
    assert result["chart_paths"] == []
    assert any("Chart" in w for w in ctx.get("warnings", []))


# ---------------------------------------------------------------------------
# RiskAgent
# ---------------------------------------------------------------------------

def eval_risk_agent_run_returns_required_keys():
    from src.agents.risk_agent import RiskAgent
    agent = RiskAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    result = agent.run(ctx)
    for key in ("z_score", "z_zone", "ml_risk_label", "ml_risk_prob",
                "risk_flags", "peer_percentile"):
        assert key in result, f"RiskAgent output missing: {key}"


def eval_risk_agent_missing_analysis_raises():
    from src.agents.risk_agent import RiskAgent
    agent = RiskAgent()
    try:
        agent.run(_base_context())
        assert False, "Should raise AgentExecutionError"
    except AgentExecutionError:
        pass


def eval_risk_agent_z_score_distress_for_pnc():
    from src.agents.risk_agent import RiskAgent
    agent = RiskAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    result = agent.run(ctx)
    assert result["z_zone"] == "distress"


def eval_risk_agent_ml_failure_degrades_to_unknown():
    from src.agents.risk_agent import RiskAgent
    agent = RiskAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    with patch("src.agents.risk_agent.run_ml_risk_model",
               return_value={"data": None, "error": "model missing"}):
        result = agent.run(ctx)
    assert result["ml_risk_label"] == "unknown"
    assert result["ml_risk_prob"] == 0.0


def eval_risk_agent_writes_to_context():
    from src.agents.risk_agent import RiskAgent
    agent = RiskAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    agent.run(ctx)
    assert "risk" in ctx


# ---------------------------------------------------------------------------
# NarrativeAgent
# ---------------------------------------------------------------------------

def eval_narrative_agent_run_returns_narrative_key():
    from src.agents.narrative_agent import NarrativeAgent
    agent = NarrativeAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    ctx["risk"] = _risk_output()

    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(type="text", text="PNC showed strong performance.")]

    with patch("src.agents.narrative_agent.anthropic") as mock_anthropic, \
         patch("src.agents.narrative_agent.query_vector_store",
               return_value={"data": [], "error": None}):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_response
        result = agent.run(ctx)

    assert "narrative" in result
    assert isinstance(result["narrative"], str) and result["narrative"]


def eval_narrative_agent_missing_risk_raises():
    from src.agents.narrative_agent import NarrativeAgent
    agent = NarrativeAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    try:
        agent.run(ctx)
        assert False, "Should raise AgentExecutionError"
    except AgentExecutionError:
        pass


def eval_narrative_agent_rag_failure_degrades():
    from src.agents.narrative_agent import NarrativeAgent
    agent = NarrativeAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    ctx["risk"] = _risk_output()

    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(type="text", text="Narrative text here.")]

    with patch("src.agents.narrative_agent.anthropic") as mock_anthropic, \
         patch("src.agents.narrative_agent.query_vector_store",
               return_value={"data": None, "error": "chroma down"}):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_response
        result = agent.run(ctx)

    assert "narrative" in result
    assert any("RAG" in w or "vector" in w.lower() for w in ctx.get("warnings", []))


def eval_narrative_agent_writes_to_context():
    from src.agents.narrative_agent import NarrativeAgent
    agent = NarrativeAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    ctx["risk"] = _risk_output()

    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(type="text", text="Analysis complete.")]

    with patch("src.agents.narrative_agent.anthropic") as mock_anthropic, \
         patch("src.agents.narrative_agent.query_vector_store",
               return_value={"data": [], "error": None}):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_response
        agent.run(ctx)

    assert "narrative" in ctx


def eval_narrative_agent_missing_ingest_still_runs():
    """NarrativeAgent only requires risk context — missing ingest must not crash it."""
    from src.agents.narrative_agent import NarrativeAgent
    from unittest.mock import MagicMock, patch as _patch
    agent = NarrativeAgent()
    ctx = {"ticker": "PNC", "year": 2023, "run_id": "t", "risk": _risk_output()}

    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(type="text", text="Narrative without ingest.")]

    with _patch("src.agents.narrative_agent.anthropic") as mock_anthropic, \
         _patch("src.agents.narrative_agent.query_vector_store",
                return_value={"data": [], "error": None}):
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response
        result = agent.run(ctx)

    assert "narrative" in result


# ---------------------------------------------------------------------------
# ReportAgent
# ---------------------------------------------------------------------------

def eval_report_agent_run_returns_report_path():
    from src.agents.report_agent import ReportAgent
    agent = ReportAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    ctx["risk"] = _risk_output()
    ctx["narrative"] = {"narrative": "Strong performance in 2023."}
    with patch("src.agents.report_agent.generate_pdf",
               return_value={"data": {"path": "reports/PNC_2023_report.pdf"}, "error": None}), \
         patch("src.agents.report_agent.format_table",
               return_value={"data": {"table": "Metric | Value"}, "error": None}):
        result = agent.run(ctx)
    assert "report_path" in result


def eval_report_agent_missing_risk_raises():
    from src.agents.report_agent import ReportAgent
    agent = ReportAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    try:
        agent.run(ctx)
        assert False, "Should raise AgentExecutionError"
    except AgentExecutionError:
        pass


def eval_report_agent_missing_narrative_uses_placeholder():
    from src.agents.report_agent import ReportAgent
    agent = ReportAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    ctx["risk"] = _risk_output()
    captured = {}
    with patch("src.agents.report_agent.generate_pdf",
               side_effect=lambda **kw: captured.update(kw) or
               {"data": {"path": "reports/PNC_2023_report.pdf"}, "error": None}), \
         patch("src.agents.report_agent.format_table",
               return_value={"data": {"table": "t"}, "error": None}):
        agent.run(ctx)
    assert "unavailable" in captured.get("narrative", "").lower()


def eval_report_agent_pdf_failure_raises():
    from src.agents.report_agent import ReportAgent
    agent = ReportAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    ctx["risk"] = _risk_output()
    ctx["narrative"] = {"narrative": "text"}
    with patch("src.agents.report_agent.generate_pdf",
               return_value={"data": None, "error": "disk full"}), \
         patch("src.agents.report_agent.format_table",
               return_value={"data": {"table": "t"}, "error": None}):
        try:
            agent.run(ctx)
            assert False, "Should raise AgentExecutionError"
        except AgentExecutionError:
            pass


def eval_report_agent_writes_to_context():
    from src.agents.report_agent import ReportAgent
    agent = ReportAgent()
    ctx = _base_context()
    ctx["analysis"] = _analysis_output()
    ctx["risk"] = _risk_output()
    ctx["narrative"] = {"narrative": "text"}
    with patch("src.agents.report_agent.generate_pdf",
               return_value={"data": {"path": "reports/PNC_2023_report.pdf"}, "error": None}), \
         patch("src.agents.report_agent.format_table",
               return_value={"data": {"table": "t"}, "error": None}):
        agent.run(ctx)
    assert "report" in ctx


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_EVALS = [
    eval_ingest_agent_run_returns_required_keys,
    eval_ingest_agent_missing_ticker_raises,
    eval_ingest_agent_future_year_raises,
    eval_ingest_agent_sec_failure_raises_agent_error,
    eval_ingest_agent_fred_failure_degrades_gracefully,
    eval_analysis_agent_run_returns_required_keys,
    eval_analysis_agent_missing_ingest_raises,
    eval_analysis_agent_writes_to_context,
    eval_analysis_agent_ratios_non_empty,
    eval_analysis_agent_chart_failure_degrades,
    eval_risk_agent_run_returns_required_keys,
    eval_risk_agent_missing_analysis_raises,
    eval_risk_agent_z_score_distress_for_pnc,
    eval_risk_agent_ml_failure_degrades_to_unknown,
    eval_risk_agent_writes_to_context,
    eval_narrative_agent_run_returns_narrative_key,
    eval_narrative_agent_missing_risk_raises,
    eval_narrative_agent_rag_failure_degrades,
    eval_narrative_agent_writes_to_context,
    eval_narrative_agent_missing_ingest_still_runs,
    eval_report_agent_run_returns_report_path,
    eval_report_agent_missing_risk_raises,
    eval_report_agent_missing_narrative_uses_placeholder,
    eval_report_agent_pdf_failure_raises,
    eval_report_agent_writes_to_context,
]


def run_agent_evals() -> dict:
    passed, failed = [], []
    for fn in _EVALS:
        try:
            fn()
            passed.append(fn.__name__)
        except Exception as exc:
            failed.append((fn.__name__, str(exc)))
    return {"passed": passed, "failed": failed, "total": len(_EVALS)}


if __name__ == "__main__":
    summary = run_agent_evals()
    print(f"Agent evals: {len(summary['passed'])}/{summary['total']} passed")
    for name, err in summary["failed"]:
        print(f"  FAIL {name}: {err}")
