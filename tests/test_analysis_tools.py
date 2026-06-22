"""Tests for src/tools/analysis_tools.py.

Written before the implementation (TDD per CLAUDE.md). Covers calc_ratios,
detect_anomalies, and plot_chart against the error-handling contract:
tools never raise (except ValueError for invalid input at the top) and always
return {'data': ..., 'error': None} / {'data': None, 'error': 'message'}.

PNC 2023 fixture (conftest.pnc_2023_ingest) yields interest_coverage =
6.1B / 4.2B = 1.452, which is < 1.5 on purpose so the anomaly path is exercised
with the documented ground-truth values.
"""

import pandas as pd
import pytest

from src.tools.analysis_tools import calc_ratios, detect_anomalies, plot_chart


# ---------------------------------------------------------------------------
# calc_ratios
# ---------------------------------------------------------------------------


def test_calc_ratios_pnc_2023_returns_correct_values(pnc_2023_ingest):
    result = calc_ratios(pnc_2023_ingest)

    assert result["error"] is None
    ratios = result["data"]
    # debt_to_equity = 504B / 53B
    assert ratios["debt_to_equity"] == pytest.approx(504_000_000_000.0 / 53_000_000_000.0)
    # return_on_assets = 5.4B / 557B
    assert ratios["return_on_assets"] == pytest.approx(5_400_000_000.0 / 557_000_000_000.0)
    # net_margin = 5.4B / 22B
    assert ratios["net_margin"] == pytest.approx(5_400_000_000.0 / 22_000_000_000.0)
    # interest_coverage = 6.1B / max(4.2B, 0.001) = 1.452...
    assert ratios["interest_coverage"] == pytest.approx(6_100_000_000.0 / 4_200_000_000.0)
    assert ratios["interest_coverage"] < 1.5
    # current_ratio is dropped for banks (unclassified balance sheets) -> None,
    # but the key is still present (contract: key present, null ok).
    assert ratios["current_ratio"] is None


def test_calc_ratios_none_interest_expense_does_not_crash(pnc_2023_ingest):
    # max(interest_expense, 0.001) caveat: None must be coerced, not crash.
    ingest = dict(pnc_2023_ingest)
    ingest["interest_expense"] = None

    result = calc_ratios(ingest)

    assert result["error"] is None
    # ebit / 0.001 -> a very large but finite, real number.
    assert result["data"]["interest_coverage"] == pytest.approx(6_100_000_000.0 / 0.001)


def test_calc_ratios_missing_required_field_returns_error(pnc_2023_ingest):
    ingest = dict(pnc_2023_ingest)
    del ingest["shareholders_equity"]

    result = calc_ratios(ingest)

    assert result["data"] is None
    assert isinstance(result["error"], str)
    assert "shareholders_equity" in result["error"]


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------


def test_detect_anomalies_flags_low_coverage_and_inverted_curve():
    ratios = {
        "current_ratio": None,
        "debt_to_equity": 9.5,
        "return_on_assets": 0.0097,
        "net_margin": 0.245,
        "interest_coverage": 1.45,  # < 1.5
    }
    fred_snapshot = {"T10Y2Y": -0.35, "UNRATE": 3.8, "FEDFUNDS": 5.33}

    result = detect_anomalies(ratios, fred_snapshot)

    assert result["error"] is None
    anomalies = result["data"]
    assert isinstance(anomalies, list)

    # interest_coverage < 1.5 -> high severity
    coverage = [a for a in anomalies if "interest_coverage" in a["metric"]]
    assert len(coverage) == 1
    assert coverage[0]["severity"] == "high"

    # T10Y2Y < 0 -> high severity, code yield_curve_inverted
    inverted = [a for a in anomalies if a.get("code") == "yield_curve_inverted"]
    assert len(inverted) == 1
    assert inverted[0]["severity"] == "high"


def test_detect_anomalies_healthy_data_returns_empty_list():
    ratios = {
        "current_ratio": None,
        "debt_to_equity": 5.0,
        "return_on_assets": 0.02,
        "net_margin": 0.30,
        "interest_coverage": 8.0,  # healthy
    }
    fred_snapshot = {"T10Y2Y": 1.10, "UNRATE": 3.5, "FEDFUNDS": 2.0}

    result = detect_anomalies(ratios, fred_snapshot)

    assert result["error"] is None
    assert result["data"] == []


# ---------------------------------------------------------------------------
# plot_chart
# ---------------------------------------------------------------------------


def test_plot_chart_saves_png_and_returns_path(mocker):
    # Mock file I/O so no real PNG is written during the test.
    save = mocker.patch("src.tools.analysis_tools.plt.savefig")
    mocker.patch("src.tools.analysis_tools.os.makedirs")

    df = pd.DataFrame({"year": [2021, 2022, 2023], "roa": [0.011, 0.010, 0.0097]})
    out = "reports/charts/test_roa.png"

    result = plot_chart(df, "line", "ROA Trend", out)

    assert result["error"] is None
    assert result["data"]["path"] == out
    save.assert_called_once()


def test_plot_chart_invalid_data_returns_error():
    result = plot_chart(pd.DataFrame(), "line", "Empty", "reports/charts/x.png")

    assert result["data"] is None
    assert isinstance(result["error"], str)
