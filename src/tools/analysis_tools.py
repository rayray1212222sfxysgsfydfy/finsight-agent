"""Financial analysis tools.

Pure functions (no LLM calls) that turn IngestAgent financials into named
ratios, detect anomalies, and render charts. Follows the CLAUDE.md error
contract: never raise (except ValueError for invalid input before any I/O);
always return {'data': ..., 'error': None} or {'data': None, 'error': 'msg'}.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib

# Non-interactive backend: this module runs headless (tools, CI, agents).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)

from src.agents.context_schema import Anomaly, Ratios  # noqa: E402

# Required IngestOutput numeric fields for ratio computation. current_assets /
# current_liabilities / working_capital are intentionally absent: banks file
# unclassified balance sheets, so current_ratio is dropped (see Session Log).
# interest_expense is handled separately (None is coerced via max(_, 0.001)).
_REQUIRED_INGEST_FIELDS: List[str] = [
    "revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "shareholders_equity",
    "ebit",
]

# Anomaly thresholds.
_MIN_INTEREST_COVERAGE: float = 1.5
_SUPPORTED_CHART_TYPES = {"line", "bar", "barh", "area"}


def calc_ratios(ingest: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the named financial ratios from an IngestOutput dict.

    Args:
        ingest: IngestAgent output (financial fields keyed per IngestOutput).

    Returns:
        {'data': Ratios, 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    for field in _REQUIRED_INGEST_FIELDS:
        if ingest.get(field) is None:
            return {"data": None, "error": f"Missing required field: {field}"}

    try:
        revenue = float(ingest["revenue"])
        net_income = float(ingest["net_income"])
        total_assets = float(ingest["total_assets"])
        total_liabilities = float(ingest["total_liabilities"])
        shareholders_equity = float(ingest["shareholders_equity"])
        ebit = float(ingest["ebit"])

        # interest_expense may be None for some banks; coerce then floor at
        # 0.001 to avoid division by zero (CLAUDE.md caveat).
        raw_ie = ingest.get("interest_expense")
        interest_expense = float(raw_ie) if raw_ie is not None else 0.0
        coverage_denom = max(interest_expense, 0.001)

        if total_assets == 0 or shareholders_equity == 0 or revenue == 0:
            return {
                "data": None,
                "error": "Zero denominator (total_assets/shareholders_equity/revenue)",
            }

        ratios: Ratios = {
            # Dropped for banks (unclassified balance sheets) — key kept, null.
            "current_ratio": None,
            "debt_to_equity": total_liabilities / shareholders_equity,
            "return_on_assets": net_income / total_assets,
            "net_margin": net_income / revenue,
            "interest_coverage": ebit / coverage_denom,
        }
        return {"data": ratios, "error": None}
    except (TypeError, ValueError) as exc:
        return {"data": None, "error": f"Ratio computation failed: {exc}"}


def detect_anomalies(
    ratios: Dict[str, Any], fred_snapshot: Dict[str, Any]
) -> Dict[str, Any]:
    """Flag anomalous ratios and macro conditions.

    Args:
        ratios: a Ratios dict (output of calc_ratios).
        fred_snapshot: macro snapshot keyed by FRED series ID.

    Returns:
        {'data': List[Anomaly], 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    try:
        anomalies: List[Anomaly] = []

        coverage = ratios.get("interest_coverage")
        if coverage is not None and coverage < _MIN_INTEREST_COVERAGE:
            anomalies.append(
                {
                    "metric": "interest_coverage",
                    "code": "low_interest_coverage",
                    "description": (
                        f"Interest coverage {coverage:.2f} is below "
                        f"{_MIN_INTEREST_COVERAGE} — earnings barely cover interest."
                    ),
                    "severity": "high",
                }
            )

        t10y2y = fred_snapshot.get("T10Y2Y") if fred_snapshot else None
        if t10y2y is not None and t10y2y < 0:
            anomalies.append(
                {
                    "metric": "T10Y2Y",
                    "code": "yield_curve_inverted",
                    "description": (
                        f"Yield curve inverted (T10Y2Y={t10y2y:.2f}) — a recession "
                        "signal that pressures bank net interest margins."
                    ),
                    "severity": "high",
                }
            )

        return {"data": anomalies, "error": None}
    except (TypeError, ValueError, AttributeError) as exc:
        return {"data": None, "error": f"Anomaly detection failed: {exc}"}


def plot_chart(
    df: Any, chart_type: str, title: str, output_path: str
) -> Dict[str, Any]:
    """Render a DataFrame to a PNG chart saved at output_path.

    Args:
        df: pandas DataFrame holding the series to plot.
        chart_type: one of 'line', 'bar', 'barh', 'area'.
        title: chart title.
        output_path: destination PNG path (typically under reports/charts/).

    Returns:
        {'data': {'path': str}, 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    try:
        if df is None or not hasattr(df, "empty") or df.empty:
            return {"data": None, "error": "plot_chart requires a non-empty DataFrame"}
        if chart_type not in _SUPPORTED_CHART_TYPES:
            return {"data": None, "error": f"Unsupported chart_type: {chart_type}"}

        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind=chart_type, ax=ax)
        ax.set_title(title)
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return {"data": {"path": output_path}, "error": None}
    except Exception as exc:  # noqa: BLE001 — tool must never raise
        return {"data": None, "error": f"Chart generation failed: {exc}"}


__all__ = ["calc_ratios", "detect_anomalies", "plot_chart"]
