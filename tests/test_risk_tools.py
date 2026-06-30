"""Tests for src/tools/risk_tools.py.

Written before the implementation (TDD per CLAUDE.md). Covers score_altman_z,
run_ml_risk_model, flag_risks, and compare_to_peers against the error-handling
contract: tools never raise (except ValueError for invalid input at the top)
and always return {'data': ..., 'error': None} / {'data': None, 'error': 'msg'}.

Altman Z ground truth (PNC 2023): the documented CLAUDE.md fixture values WITH
working_capital=18B give Z=0.253, zone='distress'. (Real EDGAR data gives 0.284
via a historical market cap and the real line items, which are NOT the fixture
values — see Session Log. Never assert 1.847 or 0.328.)
"""

import pytest

from src.tools.risk_tools import (
    compare_to_peers,
    flag_risks,
    run_ml_risk_model,
    score_altman_z,
)


# Valid ML feature vector matching src/models/feature_names.json order.
def _valid_features() -> dict:
    return {
        "debt_to_equity": 9.51,
        "return_on_assets": 0.0097,
        "interest_coverage": 1.45,
        "revenue_growth": 0.03,
        "net_margin": 0.245,
        "fedfunds": 5.33,
        "t10y2y": -0.35,
    }


class _FakeModel:
    """Stand-in for the sklearn classifier; predict_proba -> high-risk class."""

    def predict_proba(self, X):
        # Last column = probability of the highest-risk class.
        return [[0.2, 0.8]]


class _FakePipeline:
    """Stand-in for the feature pipeline; identity transform."""

    def transform(self, X):
        return X


# ---------------------------------------------------------------------------
# score_altman_z
# ---------------------------------------------------------------------------


def test_score_altman_z_pnc_2023_returns_distress(pnc_2023_ingest):
    # Documented fixture values WITH working_capital=18B -> Z=0.253.
    ingest = dict(pnc_2023_ingest)
    ingest["working_capital"] = 18_000_000_000.0

    result = score_altman_z(ingest)

    assert result["error"] is None
    assert result["data"]["z_score"] == pytest.approx(0.253, abs=0.002)
    assert result["data"]["z_zone"] == "distress"


def test_score_altman_z_none_market_cap_returns_error(pnc_2023_ingest):
    ingest = dict(pnc_2023_ingest)
    ingest["market_cap"] = None

    result = score_altman_z(ingest)

    assert result["data"] is None
    assert isinstance(result["error"], str)
    assert "market_cap" in result["error"]


def test_score_altman_z_zero_total_assets_returns_error(pnc_2023_ingest):
    ingest = dict(pnc_2023_ingest)
    ingest["total_assets"] = 0

    result = score_altman_z(ingest)

    # Must be a graceful error dict, NOT a ZeroDivisionError.
    assert result["data"] is None
    assert isinstance(result["error"], str)


# ---------------------------------------------------------------------------
# run_ml_risk_model
# ---------------------------------------------------------------------------


def test_run_ml_risk_model_success_returns_probability_and_label(mocker):
    mocker.patch("src.tools.risk_tools._load_model", return_value=_FakeModel())
    mocker.patch("src.tools.risk_tools._load_pipeline", return_value=_FakePipeline())

    result = run_ml_risk_model(_valid_features())

    assert result["error"] is None
    assert isinstance(result["data"]["probability"], float)
    assert result["data"]["label"] in {"low", "moderate", "high"}
    # predict_proba high-risk column = 0.8 -> high label.
    assert result["data"]["probability"] == pytest.approx(0.8)
    assert result["data"]["label"] == "high"


def test_run_ml_risk_model_missing_features_returns_error(mocker):
    mocker.patch("src.tools.risk_tools._load_model", return_value=_FakeModel())
    mocker.patch("src.tools.risk_tools._load_pipeline", return_value=_FakePipeline())

    features = _valid_features()
    del features["net_margin"]

    result = run_ml_risk_model(features)

    assert result["data"] is None
    assert isinstance(result["error"], str)
    assert "net_margin" in result["error"]


# ---------------------------------------------------------------------------
# flag_risks
# ---------------------------------------------------------------------------


def test_flag_risks_inverted_yield_curve_high_severity():
    ratios = {"interest_coverage": 5.0}  # healthy coverage
    fred_snapshot = {"T10Y2Y": -0.35, "UNRATE": 3.8}

    result = flag_risks(ratios, fred_snapshot)

    assert result["error"] is None
    inverted = [f for f in result["data"] if f["code"] == "yield_curve_inverted"]
    assert len(inverted) == 1
    assert inverted[0]["severity"] == "high"


def test_flag_risks_low_interest_coverage_high_severity():
    ratios = {"interest_coverage": 1.45}  # < 1.5
    fred_snapshot = {"T10Y2Y": 1.10, "UNRATE": 3.8}  # healthy macro

    result = flag_risks(ratios, fred_snapshot)

    assert result["error"] is None
    coverage = [f for f in result["data"] if "coverage" in f["code"]]
    assert len(coverage) == 1
    assert coverage[0]["severity"] == "high"


# ---------------------------------------------------------------------------
# compare_to_peers
# ---------------------------------------------------------------------------


def test_compare_to_peers_returns_percentile_between_zero_and_one():
    result = compare_to_peers(0.284, [0.10, 0.20, 0.30, 0.40, 0.50])

    assert result["error"] is None
    pct = result["data"]
    assert isinstance(pct, float)
    assert 0.0 <= pct <= 1.0
    # 0.284 exceeds 0.10 and 0.20 of five peers -> 0.4
    assert pct == pytest.approx(0.4)
