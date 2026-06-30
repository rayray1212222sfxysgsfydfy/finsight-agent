"""OrchestratorAgent — routes queries through the full FinSight pipeline.

Loads skills/orchestrator/SKILL.md as its system prompt at runtime.
Calls agents in sequence: Ingest → Analysis → Risk → Narrative → Report.
Never lets agents call each other directly. Catches AgentExecutionError from
any agent and returns {'status': 'failed', 'error': str} rather than raising.
"""

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.agents.analysis_agent import AnalysisAgent
from src.agents.ingest_agent import IngestAgent
from src.agents.narrative_agent import NarrativeAgent
from src.agents.report_agent import ReportAgent
from src.agents.risk_agent import RiskAgent
from src.utils.exceptions import AgentExecutionError
from src.utils.logger import log_agent_trace

_SKILL_PATH = Path(__file__).resolve().parents[2] / "skills" / "orchestrator" / "SKILL.md"

# Earliest fiscal year supported per CLAUDE.md validation rule.
_MIN_YEAR = 1993


def _load_skill() -> str:
    """Load the orchestrator SKILL.md system prompt."""
    return _SKILL_PATH.read_text()


class OrchestratorAgent:
    """Route a natural-language query through the full FinSight agent pipeline."""

    def __init__(self) -> None:
        self.skill = _load_skill()

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse and validate 'Analyze <TICKER> <YEAR>' query string.

        Returns {'ticker': str, 'year': int}.
        Raises ValueError with a clear message for invalid input.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")

        # Accept: "Analyze PNC 2023", "PNC 2023", "analyze pnc 2023", etc.
        match = re.search(r"\b([A-Za-z]{1,6})\b.*?\b((?:19|20)\d{2})\b", query)
        if not match:
            raise ValueError(
                f"Could not parse ticker and year from query: '{query}'. "
                "Expected format: 'Analyze <TICKER> <YEAR>'"
            )

        ticker = match.group(1).upper()
        year = int(match.group(2))

        if not ticker:
            raise ValueError("Ticker must be a non-empty string.")

        current_year = datetime.now().year
        if year > current_year:
            raise ValueError(
                f"Year {year} is in the future. Must be {current_year} or earlier."
            )
        if year < _MIN_YEAR:
            raise ValueError(
                f"Year {year} is before the minimum supported year {_MIN_YEAR}."
            )

        return {"ticker": ticker, "year": year}

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the full pipeline for the given query string.

        Returns {'status': 'complete', 'report_path': str, 'risk': dict, 'run_id': str}
        on success, or {'status': 'failed', 'error': str} on any AgentExecutionError.
        Never raises — always returns a dict.
        """
        run_id = str(uuid.uuid4())

        try:
            parsed = self._parse_query(query)
        except ValueError as exc:
            return {"status": "failed", "error": str(exc), "run_id": run_id}

        ticker = parsed["ticker"]
        year = parsed["year"]

        context: Dict[str, Any] = {
            "ticker": ticker,
            "year": year,
            "run_id": run_id,
            "warnings": [],
        }

        pipeline = [
            ("AnalysisAgent", AnalysisAgent, "analysis"),
            ("RiskAgent", RiskAgent, "risk"),
            ("NarrativeAgent", NarrativeAgent, "narrative"),
            ("ReportAgent", ReportAgent, "report"),
        ]

        try:
            # IngestAgent has signature run(ticker, year, context)
            ingest_agent = IngestAgent()
            ingest_output = ingest_agent.run(ticker, year, context)
            if "ingest" not in context and ingest_output is not None:
                context["ingest"] = ingest_output
            log_agent_trace(
                run_id=run_id,
                agent_name="IngestAgent",
                tool_name=None,
                input_summary=f"{ticker} {year}",
                output_summary=f"keys={list((ingest_output or {}).keys())}",
                latency_ms=0.0,
            )

            for agent_name, AgentClass, context_key in pipeline:
                agent = AgentClass()
                output = agent.run(context)
                # Agents write their own key into context, but also return the dict;
                # ensure context is populated even if the agent skips self-writing.
                if context_key not in context and output is not None:
                    context[context_key] = output

                log_agent_trace(
                    run_id=run_id,
                    agent_name=agent_name,
                    tool_name=None,
                    input_summary=f"{ticker} {year}",
                    output_summary=f"keys={list((output or {}).keys())}",
                    latency_ms=0.0,
                )

        except AgentExecutionError as exc:
            return {"status": "failed", "error": str(exc), "run_id": run_id}

        report = context.get("report", {})
        risk = context.get("risk", {})

        return {
            "status": "complete",
            "report_path": report.get("report_path", ""),
            "risk": risk,
            "run_id": run_id,
        }
