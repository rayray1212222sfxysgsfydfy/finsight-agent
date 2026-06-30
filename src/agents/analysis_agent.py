"""AnalysisAgent — computes ratios, detects anomalies, generates charts.

Loads skills/analysis_agent/SKILL.md as its system prompt at runtime.
Reads from context['ingest'], writes output under context['analysis'].
Never calls other agents. Never makes LLM calls.
"""

from pathlib import Path
from typing import Any, Dict, List

from src.agents.context_schema import ANALYSIS_TO_RISK_REQUIRED
from src.tools.analysis_tools import calc_ratios, detect_anomalies, plot_chart
from src.utils.exceptions import AgentExecutionError
from src.utils.logger import log_agent_trace

_SKILL_PATH = Path(__file__).resolve().parents[2] / "skills" / "analysis_agent" / "SKILL.md"


def _load_skill() -> str:
    """Load the analysis_agent SKILL.md system prompt."""
    return _SKILL_PATH.read_text()


class AnalysisAgent:
    """Compute financial ratios, detect anomalies, and generate charts."""

    def __init__(self) -> None:
        self.skill = _load_skill()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Read ingest context, run analysis tools, return AnalysisOutput dict.

        Raises AgentExecutionError for unrecoverable failures (missing ingest,
        calc_ratios error). Degrades gracefully for anomaly detection and chart
        failures, appending warnings to context['warnings'].
        """
        context.setdefault("warnings", [])

        # Validate upstream context
        ingest = context.get("ingest")
        if not ingest:
            raise AgentExecutionError(
                "AnalysisAgent: context['ingest'] is missing — IngestAgent must run first"
            )
        for key in ("revenue", "net_income", "total_assets"):
            if ingest.get(key) is None:
                raise AgentExecutionError(
                    f"AnalysisAgent: required ingest field '{key}' is None"
                )

        # 1. Compute ratios — unrecoverable
        ratios_result = calc_ratios(ingest)
        if ratios_result["error"] is not None:
            raise AgentExecutionError(
                f"AnalysisAgent: calc_ratios failed: {ratios_result['error']}"
            )
        ratios = ratios_result["data"]

        # 2. Detect anomalies — degradable
        fred_snapshot = ingest.get("fred_snapshot", {})
        anomalies_result = detect_anomalies(ratios, fred_snapshot)
        if anomalies_result["error"] is not None:
            context["warnings"].append(
                f"Anomaly detection error: {anomalies_result['error']}"
            )
            anomalies: List[Dict] = []
        else:
            anomalies = anomalies_result["data"]

        # 3. Generate charts — degradable
        chart_paths: List[str] = []
        ticker = context.get("ticker", "UNKNOWN")
        year = context.get("year", 0)
        import pandas as pd
        chart_df = pd.DataFrame({
            "ratio": list(ratios.keys()),
            "value": [v if v is not None else 0.0 for v in ratios.values()],
        }).set_index("ratio")
        chart_result = plot_chart(
            chart_df,
            chart_type="bar",
            title=f"{ticker} {year} Financial Ratios",
            output_path=f"reports/charts/{ticker}_{year}_ratios.png",
        )
        if chart_result["error"] is not None:
            context["warnings"].append(
                f"Chart generation error: {chart_result['error']}"
            )
        else:
            chart_paths.append(chart_result["data"]["path"])

        # 4. Assemble AnalysisOutput
        analysis: Dict[str, Any] = {
            "ratios": ratios,
            "anomalies": anomalies,
            "chart_paths": chart_paths,
        }

        # 5. Validate required keys for downstream
        for key in ANALYSIS_TO_RISK_REQUIRED:
            if key in ("ticker", "year"):
                continue
            if analysis.get(key) is None:
                raise AgentExecutionError(
                    f"AnalysisAgent: required output field '{key}' is None"
                )

        context["analysis"] = analysis

        log_agent_trace(
            run_id=context["run_id"],
            agent_name="AnalysisAgent",
            tool_name=None,
            input_summary=f"{ticker} {year}",
            output_summary=f"{len(anomalies)} anomalies, {len(chart_paths)} charts",
            latency_ms=0.0,
        )

        return analysis
