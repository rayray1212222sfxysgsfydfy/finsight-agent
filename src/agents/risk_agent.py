"""RiskAgent — scores credit risk via Altman Z, ML model, and rule-based flags.

Loads skills/risk_agent/SKILL.md as its system prompt at runtime.
Reads from context['ingest'] and context['analysis'], writes under context['risk'].
Never calls other agents. Never makes LLM calls.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.context_schema import RISK_TO_REPORT_REQUIRED
from src.tools.risk_tools import compare_to_peers, flag_risks, run_ml_risk_model, score_altman_z
from src.utils.exceptions import AgentExecutionError
from src.utils.logger import log_agent_trace

_SKILL_PATH = Path(__file__).resolve().parents[2] / "skills" / "risk_agent" / "SKILL.md"

# Sector peer Z-Scores used as the comparison benchmark (2023 dataset averages).
# These are computed from the 15-bank dataset built by build_features.py.
_PEER_Z_SCORES: List[float] = [
    0.187, 0.211, 0.213, 0.216, 0.224, 0.244,
    0.155, 0.171, 0.179, 0.193, 0.207,
    0.198, 0.202, 0.219, 0.231,
]


def _load_skill() -> str:
    """Load the risk_agent SKILL.md system prompt."""
    return _SKILL_PATH.read_text()


def _build_ml_features(ratios: Dict[str, Any], fred_snapshot: Dict[str, Any],
                       ingest: Dict[str, Any]) -> Dict[str, float]:
    """Assemble the 7-feature dict expected by run_ml_risk_model."""
    prev_revenue = ingest.get("prev_revenue")
    revenue = ingest.get("revenue", 0.0) or 0.0
    if prev_revenue and prev_revenue > 0:
        revenue_growth = (revenue - prev_revenue) / prev_revenue
    else:
        revenue_growth = 0.0

    return {
        "debt_to_equity": ratios.get("debt_to_equity", 0.0) or 0.0,
        "return_on_assets": ratios.get("return_on_assets", 0.0) or 0.0,
        "interest_coverage": ratios.get("interest_coverage", 0.0) or 0.0,
        "revenue_growth": revenue_growth,
        "net_margin": ratios.get("net_margin", 0.0) or 0.0,
        "fedfunds": fred_snapshot.get("FEDFUNDS", 0.0) or 0.0,
        "t10y2y": fred_snapshot.get("T10Y2Y", 0.0) or 0.0,
    }


class RiskAgent:
    """Score credit risk for a bank-year using Z-Score, ML model, and risk flags."""

    def __init__(self) -> None:
        self.skill = _load_skill()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Read analysis context, run risk tools, return RiskOutput dict.

        Raises AgentExecutionError for unrecoverable failures (missing analysis,
        score_altman_z error). Degrades gracefully for ML model and peer comparison
        failures, appending warnings to context['warnings'].
        """
        context.setdefault("warnings", [])
        ticker = context.get("ticker", "UNKNOWN")
        year = context.get("year", 0)

        # Validate upstream context
        analysis = context.get("analysis")
        if not analysis:
            raise AgentExecutionError(
                "RiskAgent: context['analysis'] is missing — AnalysisAgent must run first"
            )
        ratios = analysis.get("ratios")
        if not ratios:
            raise AgentExecutionError(
                "RiskAgent: context['analysis']['ratios'] is missing"
            )

        ingest = context.get("ingest", {})
        fred_snapshot = ingest.get("fred_snapshot", {})

        # 1. Altman Z-Score — unrecoverable
        z_result = score_altman_z(ingest)
        if z_result["error"] is not None:
            raise AgentExecutionError(
                f"RiskAgent: Z-Score computation failed for {ticker} {year}: {z_result['error']}"
            )
        z_score: float = z_result["data"]["z_score"]
        z_zone: str = z_result["data"]["z_zone"]

        # 2. ML risk model — degradable
        features = _build_ml_features(ratios, fred_snapshot, ingest)
        ml_result = run_ml_risk_model(features)
        if ml_result["error"] is not None:
            context["warnings"].append(
                f"ML model error for {ticker} {year}: {ml_result['error']}"
            )
            ml_risk_prob: float = 0.0
            ml_risk_label: str = "unknown"
        else:
            ml_risk_prob = ml_result["data"]["probability"]
            ml_risk_label = ml_result["data"]["label"]

        # 3. Rule-based risk flags — degradable
        flags_result = flag_risks(ratios, fred_snapshot)
        if flags_result["error"] is not None:
            context["warnings"].append(
                f"Flag risks error for {ticker} {year}: {flags_result['error']}"
            )
            risk_flags: List[Dict] = []
        else:
            risk_flags = flags_result["data"]

        # 4. Peer comparison — degradable
        peers_result = compare_to_peers(z_score, _PEER_Z_SCORES)
        if peers_result["error"] is not None:
            context["warnings"].append(
                f"Peer comparison error for {ticker} {year}: {peers_result['error']}"
            )
            peer_percentile: Optional[float] = None
        else:
            peer_percentile = peers_result["data"]

        # 5. Assemble RiskOutput
        risk: Dict[str, Any] = {
            "z_score": z_score,
            "z_zone": z_zone,
            "ml_risk_prob": ml_risk_prob,
            "ml_risk_label": ml_risk_label,
            "risk_flags": risk_flags,
            "peer_percentile": peer_percentile,
        }

        # 6. Validate required keys for downstream
        for key in RISK_TO_REPORT_REQUIRED:
            if key not in risk:
                raise AgentExecutionError(
                    f"RiskAgent: required output field '{key}' is missing"
                )

        context["risk"] = risk

        log_agent_trace(
            run_id=context["run_id"],
            agent_name="RiskAgent",
            tool_name=None,
            input_summary=f"{ticker} {year}",
            output_summary=f"z={z_score:.3f} zone={z_zone} ml={ml_risk_label} flags={len(risk_flags)}",
            latency_ms=0.0,
        )

        return risk
