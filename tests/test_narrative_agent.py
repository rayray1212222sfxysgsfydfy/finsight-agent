"""Tests for NarrativeAgent — TDD per CLAUDE.md. Run RED before implementation.

Claude API and all tool functions are mocked — no real API calls permitted.
"""

import pytest

from src.agents.narrative_agent import NarrativeAgent
from src.utils.exceptions import AgentExecutionError

# ---------------------------------------------------------------------------
# Helpers — fake Anthropic SDK objects
# ---------------------------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeToolUseBlock:
    def __init__(self, tool_use_id: str = "tu_1", name: str = "query_vector_store",
                 input: dict = None) -> None:
        self.type = "tool_use"
        self.id = tool_use_id
        self.name = name
        self.input = input or {"query": "PNC 2023 revenue", "n_results": 3}


class _FakeResponse:
    def __init__(self, content, stop_reason: str = "end_turn",
                 input_tokens: int = 100, output_tokens: int = 200) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = type("Usage", (), {"input_tokens": input_tokens,
                                        "output_tokens": output_tokens})()


_RAG_SUCCESS = {
    "data": ["PNC reported $22B revenue in 2023.", "Net income was $5.4B."],
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
        "fred_snapshot": {"FEDFUNDS": 5.33, "T10Y2Y": -0.35, "UNRATE": 3.7,
                          "CPIAUCSL": 314.17, "GS10": 3.97},
        "market_cap": 55_000_000_000.0,
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
    "risk": {
        "z_score": 0.253,
        "z_zone": "distress",
        "ml_risk_prob": 0.72,
        "ml_risk_label": "high",
        "risk_flags": [
            {"code": "yield_curve_inverted", "description": "T10Y2Y=-0.35",
             "severity": "high"},
        ],
        "peer_percentile": 0.55,
    },
}


def _fresh_context():
    import copy
    return copy.deepcopy(_BASE_CONTEXT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_narrative_agent_run_success_returns_narrative_string(mocker):
    """Happy path: RAG and API succeed — returns dict with non-empty narrative."""
    mocker.patch("src.agents.narrative_agent.query_vector_store",
                 return_value=_RAG_SUCCESS)
    mock_create = mocker.patch(
        "src.agents.narrative_agent.anthropic.Anthropic"
    )
    mock_create.return_value.messages.create.return_value = _FakeResponse(
        content=[_FakeTextBlock("PNC 2023 showed strong revenue of $22B...")]
    )

    ctx = _fresh_context()
    agent = NarrativeAgent()
    result = agent.run(ctx)

    assert "narrative" in result
    assert isinstance(result["narrative"], str)
    assert len(result["narrative"]) > 0


def test_narrative_agent_run_implements_agentic_loop(mocker):
    """If first response is tool_use, agent loops until end_turn."""
    mocker.patch("src.agents.narrative_agent.query_vector_store",
                 return_value=_RAG_SUCCESS)

    tool_response = _FakeResponse(
        content=[_FakeToolUseBlock()],
        stop_reason="tool_use",
    )
    final_response = _FakeResponse(
        content=[_FakeTextBlock("Final narrative after tool use.")],
        stop_reason="end_turn",
    )

    mock_client = mocker.patch("src.agents.narrative_agent.anthropic.Anthropic")
    mock_client.return_value.messages.create.side_effect = [tool_response, final_response]

    ctx = _fresh_context()
    agent = NarrativeAgent()
    result = agent.run(ctx)

    # API must have been called twice (tool_use loop + final)
    assert mock_client.return_value.messages.create.call_count == 2
    assert result["narrative"] == "Final narrative after tool use."


def test_narrative_agent_run_rag_failure_continues_with_warning(mocker):
    """query_vector_store failure is degradable — narrative generates without RAG."""
    mocker.patch("src.agents.narrative_agent.query_vector_store",
                 return_value={"data": None, "error": "ChromaDB unavailable"})
    mock_create = mocker.patch("src.agents.narrative_agent.anthropic.Anthropic")
    mock_create.return_value.messages.create.return_value = _FakeResponse(
        content=[_FakeTextBlock("Narrative without RAG context.")]
    )

    ctx = _fresh_context()
    agent = NarrativeAgent()
    result = agent.run(ctx)

    assert result["narrative"] == "Narrative without RAG context."
    assert any("rag" in w.lower() or "chroma" in w.lower() or "vector" in w.lower()
               for w in ctx.get("warnings", []))


def test_narrative_agent_run_api_exception_raises_agent_execution_error(mocker):
    """Claude API exception is unrecoverable — raises AgentExecutionError."""
    mocker.patch("src.agents.narrative_agent.query_vector_store",
                 return_value=_RAG_SUCCESS)
    mock_create = mocker.patch("src.agents.narrative_agent.anthropic.Anthropic")
    mock_create.return_value.messages.create.side_effect = Exception("API timeout")

    ctx = _fresh_context()
    agent = NarrativeAgent()

    with pytest.raises(AgentExecutionError, match="[Aa][Pp][Ii]|[Nn]arrative"):
        agent.run(ctx)


def test_narrative_agent_run_missing_risk_raises_agent_execution_error(mocker):
    """Missing context['risk'] must raise AgentExecutionError before any API call."""
    api_mock = mocker.patch("src.agents.narrative_agent.anthropic.Anthropic")
    ctx = _fresh_context()
    del ctx["risk"]
    agent = NarrativeAgent()

    with pytest.raises(AgentExecutionError, match="risk"):
        agent.run(ctx)

    api_mock.return_value.messages.create.assert_not_called()
