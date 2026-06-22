"""Context schema definitions.

This module defines the TypedDict schemas and the required-key lists used by
agents when handing off context between pipeline stages. Keep this file limited
to type definitions and constants — no runtime logic.
"""

from typing import Any, Dict, List, Optional, TypedDict


class BaseContext(TypedDict):
	"""Minimal orchestrator context passed to every agent.

	Fields:
	- `ticker`: bank ticker string
	- `year`: fiscal year as 4-digit integer
	- `run_id`: unique run identifier
	"""

	ticker: str
	year: int
	run_id: str


# Expected keys inside `fred_snapshot`. These are FRED series IDs as documented
# in CLAUDE.md's "Data Sources" section. Values are the last observation on or
# before the filing's fiscal year end date.
FRED_SNAPSHOT_KEYS: List[str] = [
	"FEDFUNDS",
	"CPIAUCSL",
	"UNRATE",
	"GS10",
	"T10Y2Y",
]


class IngestOutput(TypedDict):
	"""Structured financials and metadata produced by the IngestAgent.

	Types follow the shape described in CLAUDE.md. Numeric finance fields use
	`float`. `fred_snapshot` is keyed by the FRED series IDs in
	`FRED_SNAPSHOT_KEYS`, each mapping to a float value (or None if unavailable).
	"""

	revenue: float
	net_income: float
	total_assets: float
	total_liabilities: float
	shareholders_equity: float
	# Optional: banks file unclassified balance sheets, so AssetsCurrent /
	# LiabilitiesCurrent are absent from XBRL. These three are structurally None
	# for our bank dataset; `current_ratio` was dropped from the ML features and
	# the Altman WC/TA term is computed with WC=0 when working_capital is None.
	current_assets: Optional[float]
	current_liabilities: Optional[float]
	working_capital: Optional[float]
	ebit: float
	interest_expense: float
	retained_earnings: float
	cash: float
	# Per-fiscal-year shares (dei:EntityCommonStockSharesOutstanding), used with
	# year-end price to compute historical market_cap. None if unavailable.
	shares_outstanding: Optional[float]
	market_cap: float
	fred_snapshot: Dict[str, Any]
	raw_text_id: str


# Minimum keys IngestAgent's output must contain before AnalysisAgent can run.
# Per CLAUDE.md's Agent Context Schema section.
INGEST_TO_ANALYSIS_REQUIRED: List[str] = [
	"ticker",
	"year",
	"revenue",
	"net_income",
	"total_assets",
]


class Ratios(TypedDict):
	"""The five named financial ratios computed by AnalysisAgent.

	Formulas (see CLAUDE.md "Financial Domain Reference"):
	- `current_ratio` = current_assets / current_liabilities
	- `debt_to_equity` = total_liabilities / shareholders_equity
	- `return_on_assets` = net_income / total_assets
	- `net_margin` = net_income / revenue
	- `interest_coverage` = ebit / max(interest_expense, 0.001)
	"""

	current_ratio: float
	debt_to_equity: float
	return_on_assets: float
	net_margin: float
	interest_coverage: float


class Anomaly(TypedDict):
	"""A single detected anomaly flagged by AnalysisAgent.

	Fields:
	- `metric`: name of the metric or ratio that triggered the anomaly
	- `description`: human-readable explanation of what was detected
	- `severity`: one of "low", "moderate", "high"
	"""

	metric: str
	description: str
	severity: str


class AnalysisOutput(TypedDict):
	"""Structured output produced by AnalysisAgent, added under context["analysis"].

	`ratios` holds the five named ratios (see `Ratios`). `anomalies` is a list
	of detected anomalies (see `Anomaly`). `chart_paths` lists file paths to
	generated chart PNGs (see CLAUDE.md "Output" table — reports/charts/).
	"""

	ratios: Ratios
	anomalies: List[Anomaly]
	chart_paths: List[str]


# Minimum keys AnalysisAgent's output must contain before RiskAgent can run.
# Per CLAUDE.md's Agent Context Schema section.
ANALYSIS_TO_RISK_REQUIRED: List[str] = [
	"ratios",
	"anomalies",
	"ticker",
	"year",
]


class RiskFlag(TypedDict):
	"""A single risk flag surfaced by RiskAgent.

	Fields:
	- `code`: short identifier for the flag (e.g. "yield_curve_inverted")
	- `description`: human-readable explanation
	- `severity`: one of "low", "moderate", "high"
	"""

	code: str
	description: str
	severity: str


class RiskOutput(TypedDict):
	"""Structured output produced by RiskAgent, added under context["risk"].

	Fields:
	- `z_score`: full Altman Z-Score (see CLAUDE.md "Altman Z-Score")
	- `z_zone`: one of "safe", "grey", "distress" based on `z_score`
	- `ml_risk_prob`: probability output from the trained sklearn classifier (0-1)
	- `ml_risk_label`: one of "low", "moderate", "high" from the classifier
	- `risk_flags`: list of flags (see `RiskFlag`), e.g. yield curve inversion,
	  rising unemployment, low interest coverage
	- `peer_percentile`: this bank's Z-Score percentile rank among sector peers,
	  used to contextualize bank Z-Scores per CLAUDE.md's caveat that banks run
	  structurally low Z-Scores
	"""

	z_score: float
	z_zone: str
	ml_risk_prob: float
	ml_risk_label: str
	risk_flags: List[RiskFlag]
	peer_percentile: float


# Minimum keys RiskAgent's output must contain before ReportAgent can run.
# Per CLAUDE.md's Agent Context Schema section.
RISK_TO_REPORT_REQUIRED: List[str] = [
	"z_score",
	"ml_risk_prob",
	"risk_flags",
	"peer_percentile",
]


__all__ = [
	"BaseContext",
	"FRED_SNAPSHOT_KEYS",
	"IngestOutput",
	"INGEST_TO_ANALYSIS_REQUIRED",
	"Ratios",
	"Anomaly",
	"AnalysisOutput",
	"ANALYSIS_TO_RISK_REQUIRED",
	"RiskFlag",
	"RiskOutput",
	"RISK_TO_REPORT_REQUIRED",
]