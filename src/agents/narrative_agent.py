"""NarrativeAgent — generates analyst narrative using Claude API with RAG context.

Loads skills/narrative_agent/SKILL.md as its system prompt at runtime.
Reads from context['risk'] and context['analysis'], writes under context['narrative'].
Makes LLM calls via the Anthropic SDK agentic loop.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import anthropic

from src.tools.rag_tools import query_vector_store
from src.utils.exceptions import AgentExecutionError
from src.utils.logger import log_agent_trace

_SKILL_PATH = Path(__file__).resolve().parents[2] / "skills" / "narrative_agent" / "SKILL.md"

_QUERY_TOOL = {
    "name": "query_vector_store",
    "description": "Query the ChromaDB vector store for relevant filing text chunks.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text"},
            "n_results": {"type": "integer", "description": "Number of results"},
        },
        "required": ["query"],
    },
}


def _load_skill() -> str:
    """Load the narrative_agent SKILL.md system prompt."""
    return _SKILL_PATH.read_text()


class NarrativeAgent:
    """Generate analyst narrative from risk/analysis context using the Claude API."""

    def __init__(self) -> None:
        self.skill = _load_skill()
        self.client = anthropic.Anthropic()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Read risk context, call Claude with RAG, return {'narrative': str}.

        Raises AgentExecutionError for missing risk context or Claude API failures.
        Degrades gracefully for RAG failures, appending warnings to context['warnings'].
        """
        context.setdefault("warnings", [])
        ticker = context.get("ticker", "UNKNOWN")
        year = context.get("year", 0)

        # Validate upstream context
        risk = context.get("risk")
        if not risk:
            raise AgentExecutionError(
                "NarrativeAgent: context['risk'] is missing — RiskAgent must run first"
            )

        analysis = context.get("analysis", {})
        ingest = context.get("ingest", {})

        # RAG query — degradable
        rag_query = f"{ticker} {year} financial performance revenue income"
        rag_result = query_vector_store(rag_query, n_results=3)
        rag_chunks: List[str] = []
        if rag_result["error"] is not None:
            context["warnings"].append(
                f"RAG/vector store unavailable for {ticker} {year}: {rag_result['error']}"
            )
        else:
            raw = rag_result["data"] or []
            rag_chunks = [
                c["text"] if isinstance(c, dict) else str(c) for c in raw
            ]

        # Build user message
        rag_section = ""
        if rag_chunks:
            rag_section = "\n\nRelevant filing excerpts:\n" + "\n---\n".join(rag_chunks)

        user_message = (
            f"Generate an analyst narrative for {ticker} {year}.\n\n"
            f"Risk data: {json.dumps(risk)}\n"
            f"Analysis data: {json.dumps(analysis)}\n"
            f"Ingest summary: revenue={ingest.get('revenue')}, "
            f"net_income={ingest.get('net_income')}, "
            f"total_assets={ingest.get('total_assets')}"
            f"{rag_section}"
        )

        messages = [{"role": "user", "content": user_message}]

        # Claude agentic loop
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system=self.skill,
                tools=[_QUERY_TOOL],
                messages=messages,
            )

            while response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_input = block.input or {}
                        tr = query_vector_store(
                            tool_input.get("query", ""),
                            n_results=tool_input.get("n_results", 3),
                        )
                        result_content = (
                            json.dumps(tr["data"]) if tr["error"] is None
                            else f"Error: {tr['error']}"
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_content,
                        })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                response = self.client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2000,
                    system=self.skill,
                    tools=[_QUERY_TOOL],
                    messages=messages,
                )

        except Exception as exc:
            raise AgentExecutionError(
                f"NarrativeAgent: Claude API call failed for {ticker} {year}: {exc}"
            ) from exc

        # Extract narrative text
        narrative = ""
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                narrative = block.text
                break

        result = {"narrative": narrative}
        context["narrative"] = result

        log_agent_trace(
            run_id=context.get("run_id", ""),
            agent_name="NarrativeAgent",
            tool_name=None,
            input_summary=f"{ticker} {year}",
            output_summary=f"narrative length={len(narrative)}",
            latency_ms=0.0,
        )

        return result
