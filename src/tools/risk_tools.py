"""Risk scoring tools.

Pure functions (no LLM calls) that score financial distress. Follows the
CLAUDE.md error contract: never raise (except ValueError for invalid input
before any I/O); always return {'data': ..., 'error': None} or
{'data': None, 'error': 'message'}.

Altman Z-Score (CLAUDE.md):
    Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MktCap/TL) + 1.0*(Rev/TA)
    Zones: > 2.99 safe, 1.81-2.99 grey, < 1.81 distress
Banks run structurally low Z-Scores (high leverage); the WC/TA term uses WC=0
when working_capital is None (unclassified bank balance sheets). PNC 2023 ground
truth: 0.253 (fixture values w/ WC=18B) and 0.284 (real EDGAR data) — both
distress. Never assert 1.847 or 0.328.
"""

import json
import os
from typing import Any, Dict, List, Optional

import joblib

# Fields score_altman_z requires (and must be non-zero where they're denominators).
# working_capital is intentionally absent: it is Optional and treated as 0 when None.
_REQUIRED_ALTMAN_FIELDS: List[str] = [
    "total_assets",
    "total_liabilities",
    "retained_earnings",
    "ebit",
    "market_cap",
    "revenue",
]

# Distress thresholds.
_LOW_INTEREST_COVERAGE: float = 1.5
_ELEVATED_UNRATE: float = 5.0

# ML model artifact locations and the canonical feature order.
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MODEL_PATH = os.path.join(_MODELS_DIR, "risk_classifier.pkl")
_PIPELINE_PATH = os.path.join(_MODELS_DIR, "feature_pipeline.pkl")
_FEATURE_NAMES_PATH = os.path.join(_MODELS_DIR, "feature_names.json")


def _zone_for(z_score: float) -> str:
    """Map an Altman Z-Score to its zone label."""
    if z_score > 2.99:
        return "safe"
    if z_score >= 1.81:
        return "grey"
    return "distress"


def score_altman_z(ingest: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the Altman Z-Score and zone from an IngestOutput dict.

    Args:
        ingest: IngestAgent output (financial fields keyed per IngestOutput).

    Returns:
        {'data': {'z_score': float, 'z_zone': str}, 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    for field in _REQUIRED_ALTMAN_FIELDS:
        if ingest.get(field) is None:
            return {"data": None, "error": f"Missing required field: {field}"}

    try:
        total_assets = float(ingest["total_assets"])
        total_liabilities = float(ingest["total_liabilities"])
        retained_earnings = float(ingest["retained_earnings"])
        ebit = float(ingest["ebit"])
        market_cap = float(ingest["market_cap"])
        revenue = float(ingest["revenue"])

        # WC/TA term uses 0 when working_capital is None (banks file
        # unclassified balance sheets — see CLAUDE.md).
        raw_wc = ingest.get("working_capital")
        working_capital = float(raw_wc) if raw_wc is not None else 0.0

        if total_assets == 0:
            return {"data": None, "error": "total_assets is zero (cannot divide)"}
        if total_liabilities == 0:
            return {"data": None, "error": "total_liabilities is zero (cannot divide)"}

        z_score = (
            1.2 * (working_capital / total_assets)
            + 1.4 * (retained_earnings / total_assets)
            + 3.3 * (ebit / total_assets)
            + 0.6 * (market_cap / total_liabilities)
            + 1.0 * (revenue / total_assets)
        )

        return {
            "data": {"z_score": z_score, "z_zone": _zone_for(z_score)},
            "error": None,
        }
    except (TypeError, ValueError) as exc:
        return {"data": None, "error": f"Altman Z computation failed: {exc}"}


def _load_feature_names() -> List[str]:
    """Load the canonical ordered feature list (source of truth for model input)."""
    with open(_FEATURE_NAMES_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_model() -> Any:
    """Load the trained sklearn risk classifier from disk (joblib, see train.py)."""
    return joblib.load(_MODEL_PATH)


def _load_pipeline() -> Any:
    """Load the fitted feature pipeline from disk (joblib, see train.py)."""
    return joblib.load(_PIPELINE_PATH)


def _label_for(probability: float) -> str:
    """Map a high-risk probability to a coarse label."""
    if probability < 0.34:
        return "low"
    if probability < 0.67:
        return "moderate"
    return "high"


def run_ml_risk_model(features: Dict[str, Any]) -> Dict[str, Any]:
    """Score a feature vector with the trained risk classifier.

    Args:
        features: dict keyed by the names in feature_names.json. Every feature
            must be present and non-None.

    Returns:
        {'data': {'probability': float, 'label': str}, 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    try:
        feature_names = _load_feature_names()
    except (OSError, ValueError) as exc:
        return {"data": None, "error": f"Could not load feature names: {exc}"}

    missing = [name for name in feature_names if features.get(name) is None]
    if missing:
        return {"data": None, "error": f"Missing features: {', '.join(missing)}"}

    try:
        vector = [float(features[name]) for name in feature_names]
    except (TypeError, ValueError) as exc:
        return {"data": None, "error": f"Non-numeric feature: {exc}"}

    try:
        # train.py saves a single combined Pipeline (scaler + clf) to risk_classifier.pkl.
        # Call predict_proba directly — no separate transform step needed.
        pipeline = _load_model()
        proba_rows = pipeline.predict_proba([vector])
        # Last column = probability of the highest-risk class.
        probability = float(proba_rows[0][-1])
        return {
            "data": {"probability": probability, "label": _label_for(probability)},
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — tool must never raise
        return {"data": None, "error": f"Model scoring failed: {exc}"}


def flag_risks(
    ratios: Dict[str, Any], fred_snapshot: Dict[str, Any]
) -> Dict[str, Any]:
    """Surface qualitative risk flags from ratios and macro conditions.

    Per CLAUDE.md, always flag yield-curve inversion (T10Y2Y < 0) and elevated
    unemployment; low interest coverage is flagged as high severity. Banks run
    structurally low Z-Scores, so distress zones are read relative to peers
    (see compare_to_peers) rather than flagged in isolation here.

    Args:
        ratios: a Ratios dict (output of analysis_tools.calc_ratios).
        fred_snapshot: macro snapshot keyed by FRED series ID.

    Returns:
        {'data': List[RiskFlag], 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    try:
        flags: List[Dict[str, str]] = []

        coverage = ratios.get("interest_coverage") if ratios else None
        if coverage is not None and coverage < _LOW_INTEREST_COVERAGE:
            flags.append(
                {
                    "code": "low_interest_coverage",
                    "description": (
                        f"Interest coverage {coverage:.2f} is below "
                        f"{_LOW_INTEREST_COVERAGE} — earnings barely cover interest."
                    ),
                    "severity": "high",
                }
            )

        t10y2y = fred_snapshot.get("T10Y2Y") if fred_snapshot else None
        if t10y2y is not None and t10y2y < 0:
            flags.append(
                {
                    "code": "yield_curve_inverted",
                    "description": (
                        f"Yield curve inverted (T10Y2Y={t10y2y:.2f}) — a recession "
                        "signal that compresses bank net interest margins."
                    ),
                    "severity": "high",
                }
            )

        unrate = fred_snapshot.get("UNRATE") if fred_snapshot else None
        if unrate is not None and unrate > _ELEVATED_UNRATE:
            flags.append(
                {
                    "code": "elevated_unemployment",
                    "description": (
                        f"Unemployment at {unrate:.1f}% is elevated — a leading "
                        "indicator of rising credit losses for lenders."
                    ),
                    "severity": "moderate",
                }
            )

        return {"data": flags, "error": None}
    except (TypeError, ValueError, AttributeError) as exc:
        return {"data": None, "error": f"Risk flagging failed: {exc}"}


def compare_to_peers(
    z_score: float, peer_z_scores: List[float]
) -> Dict[str, Any]:
    """Rank a Z-Score against sector peers as a percentile in [0, 1].

    Banks run structurally low Z-Scores, so absolute zones mislead; the
    percentile contextualizes a bank against its peer group (CLAUDE.md caveat).

    Args:
        z_score: this bank's Altman Z-Score.
        peer_z_scores: Z-Scores of sector peers to rank against.

    Returns:
        {'data': float, 'error': None} on success — fraction of peers at or
            below z_score.
        {'data': None, 'error': 'message'} on failure
    """
    try:
        peers: List[float] = [
            float(p) for p in (peer_z_scores or []) if p is not None
        ]
    except (TypeError, ValueError) as exc:
        return {"data": None, "error": f"Invalid peer Z-Scores: {exc}"}

    if not peers:
        return {"data": None, "error": "No peer Z-Scores provided"}

    try:
        z = float(z_score)
    except (TypeError, ValueError) as exc:
        return {"data": None, "error": f"Invalid z_score: {exc}"}

    at_or_below = sum(1 for p in peers if p <= z)
    percentile = at_or_below / len(peers)
    return {"data": percentile, "error": None}


__all__ = [
    "score_altman_z",
    "run_ml_risk_model",
    "flag_risks",
    "compare_to_peers",
]
