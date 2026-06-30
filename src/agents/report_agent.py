"""ReportAgent — assembles the final PDF analyst report from pipeline context.

Loads skills/report_agent/SKILL.md as its system prompt at runtime.
Reads from context['ingest'], context['analysis'], context['risk'],
context['narrative']. Writes the PDF to reports/<TICKER>_<YEAR>_report.pdf
and returns {'report_path': str}.
"""

from pathlib import Path
from typing import Any, Dict

from src.agents.context_schema import RISK_TO_REPORT_REQUIRED
from src.tools.report_tools import format_table, generate_pdf
from src.utils.exceptions import AgentExecutionError
from src.utils.logger import log_agent_trace

_SKILL_PATH = Path(__file__).resolve().parents[2] / "skills" / "report_agent" / "SKILL.md"

_NARRATIVE_PLACEHOLDER = (
    "Narrative unavailable — NarrativeAgent output was missing from context. "
    "Please review pipeline logs for details."
)


def _load_skill() -> str:
    """Load the report_agent SKILL.md system prompt."""
    return _SKILL_PATH.read_text()


class ReportAgent:
    """Assemble and write the final PDF analyst report from pipeline context."""

    def __init__(self) -> None:
        self.skill = _load_skill()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Read pipeline context, generate PDF report, return {'report_path': str}.

        Raises AgentExecutionError for missing risk context or PDF generation failure.
        Degrades gracefully for missing narrative, using a placeholder string.
        """
        context.setdefault("warnings", [])
        ticker = context.get("ticker", "UNKNOWN")
        year = context.get("year", 0)

        # Validate upstream context — risk is required
        risk = context.get("risk")
        if not risk:
            raise AgentExecutionError(
                "ReportAgent: context['risk'] is missing — RiskAgent must run first"
            )
        for key in RISK_TO_REPORT_REQUIRED:
            if key not in risk:
                raise AgentExecutionError(
                    f"ReportAgent: required risk field '{key}' is missing"
                )

        ingest = context.get("ingest", {})
        analysis = context.get("analysis", {})
        ratios = analysis.get("ratios", {})
        anomalies = analysis.get("anomalies", [])
        chart_paths = analysis.get("chart_paths", [])

        # Narrative — degradable
        narrative_block = context.get("narrative")
        if narrative_block and isinstance(narrative_block, dict):
            narrative = narrative_block.get("narrative") or _NARRATIVE_PLACEHOLDER
        elif isinstance(narrative_block, str) and narrative_block:
            narrative = narrative_block
        else:
            context["warnings"].append(
                "ReportAgent: context['narrative'] missing — using placeholder text"
            )
            narrative = _NARRATIVE_PLACEHOLDER

        # Build summary rows for the metrics table
        rows = [
            {"Metric": "Revenue ($B)", "Value": f"{(ingest.get('revenue') or 0) / 1e9:.1f}"},
            {"Metric": "Net Income ($B)", "Value": f"{(ingest.get('net_income') or 0) / 1e9:.1f}"},
            {"Metric": "Total Assets ($B)", "Value": f"{(ingest.get('total_assets') or 0) / 1e9:.0f}"},
            {"Metric": "Debt / Equity", "Value": f"{ratios.get('debt_to_equity') or 0:.3f}"},
            {"Metric": "Return on Assets", "Value": f"{ratios.get('return_on_assets') or 0:.4f}"},
            {"Metric": "Net Margin", "Value": f"{ratios.get('net_margin') or 0:.3f}"},
            {"Metric": "Interest Coverage", "Value": f"{ratios.get('interest_coverage') or 0:.3f}"},
            {"Metric": "Altman Z-Score", "Value": f"{risk.get('z_score') or 0:.3f}"},
            {"Metric": "Z-Zone", "Value": risk.get("z_zone", "")},
            {"Metric": "ML Risk Label", "Value": risk.get("ml_risk_label", "")},
            {"Metric": "ML Risk Prob", "Value": f"{risk.get('ml_risk_prob') or 0:.2f}"},
            {"Metric": "Peer Percentile", "Value": f"{risk.get('peer_percentile') or 0:.2f}"},
        ]

        table_result = format_table(rows, columns=["Metric", "Value"])
        table_str = (
            table_result["data"]["table"]
            if table_result["error"] is None
            else "(table formatting failed)"
        )
        if table_result["error"] is not None:
            context["warnings"].append(f"ReportAgent: format_table error: {table_result['error']}")

        # Ensure reports/ directory exists
        Path("reports").mkdir(parents=True, exist_ok=True)
        output_path = f"reports/{ticker}_{year}_report.pdf"

        pdf_result = generate_pdf(
            output_path=output_path,
            ticker=ticker,
            year=year,
            narrative=narrative,
            table_str=table_str,
            chart_paths=chart_paths,
        )
        if pdf_result["error"] is not None:
            raise AgentExecutionError(
                f"ReportAgent: PDF generation failed for {ticker} {year}: {pdf_result['error']}"
            )

        report_path: str = pdf_result["data"]["path"]
        result = {"report_path": report_path}
        context["report"] = result

        log_agent_trace(
            run_id=context.get("run_id", ""),
            agent_name="ReportAgent",
            tool_name=None,
            input_summary=f"{ticker} {year}",
            output_summary=f"report_path={report_path}",
            latency_ms=0.0,
        )

        return result
